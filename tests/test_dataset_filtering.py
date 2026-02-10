"""Tests for dataset filtering, SpecAugment, and cache versioning."""

import json
import tempfile
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from beat_weaver.model.config import ModelConfig
from beat_weaver.model.dataset import BeatSaberDataset, _cache_version_key


def _make_test_data(tmp_path, notes_data, metadata_list):
    """Helper to create minimal Parquet + metadata + manifest for testing."""
    # Write notes.parquet
    processed = tmp_path / "processed"
    processed.mkdir()

    from beat_weaver.storage.writer import NOTES_SCHEMA

    table = pa.table(notes_data, schema=NOTES_SCHEMA)
    pq.write_table(table, processed / "notes.parquet")

    # Write metadata.json
    (processed / "metadata.json").write_text(json.dumps(metadata_list))

    # Write audio manifest — point to dummy files (won't be loaded in tests
    # since we'll use mel cache)
    manifest = {}
    for m in metadata_list:
        manifest[m["hash"]] = str(tmp_path / "dummy.wav")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    return processed, manifest_path


def _make_simple_dataset(tmp_path, difficulties, characteristics, bpms):
    """Create a dataset with one note per combination for filtering tests."""
    song_hashes = []
    diff_list = []
    char_list = []
    beats = []
    times = []
    xs = []
    ys = []
    colors = []
    cut_dirs = []
    angle_offsets = []
    bpm_col = []
    sources = []

    metadata = []

    for i, (diff, char, bpm) in enumerate(zip(difficulties, characteristics, bpms)):
        h = f"hash_{i:04d}"
        song_hashes.extend([h] * 4)
        diff_list.extend([diff] * 4)
        char_list.extend([char] * 4)
        beats.extend([0.0, 1.0, 2.0, 3.0])
        times.extend([0.0, 0.5, 1.0, 1.5])
        xs.extend([0, 1, 2, 3])
        ys.extend([0, 0, 0, 0])
        colors.extend([0, 1, 0, 1])
        cut_dirs.extend([1, 1, 1, 1])
        angle_offsets.extend([0, 0, 0, 0])
        bpm_col.extend([bpm] * 4)
        sources.extend(["beatsaver"] * 4)

        metadata.append({
            "hash": h, "source": "beatsaver", "source_id": h,
            "song_name": f"Song {i}", "bpm": bpm,
        })

    notes_data = {
        "song_hash": song_hashes,
        "source": sources,
        "difficulty": diff_list,
        "characteristic": char_list,
        "bpm": bpm_col,
        "beat": beats,
        "time_seconds": times,
        "x": xs,
        "y": ys,
        "color": colors,
        "cut_direction": cut_dirs,
        "angle_offset": angle_offsets,
    }

    processed, manifest_path = _make_test_data(tmp_path, notes_data, metadata)

    # Pre-create mel cache files so dataset doesn't need actual audio
    cache_dir = processed / "mel_cache"
    cache_dir.mkdir()
    for m in metadata:
        mel = np.zeros((80, 100), dtype=np.float32)
        np.save(cache_dir / f"{m['hash']}_{m['bpm']}.npy", mel)

    return processed, manifest_path


class TestFilterByDifficulty:
    def test_only_expert_and_above(self, tmp_path):
        """Only Expert/ExpertPlus samples when min_difficulty='Expert'."""
        processed, manifest = _make_simple_dataset(
            tmp_path,
            difficulties=["Easy", "Normal", "Hard", "Expert", "ExpertPlus"],
            characteristics=["Standard"] * 5,
            bpms=[120.0] * 5,
        )
        config = ModelConfig(min_difficulty="Expert", max_seq_len=64)
        ds = BeatSaberDataset(processed, manifest, config, split="train")
        diffs = {s["difficulty"] for s in ds.samples}
        assert "Easy" not in diffs
        assert "Normal" not in diffs
        assert "Hard" not in diffs

    def test_all_when_easy(self, tmp_path):
        """All difficulties included when min_difficulty='Easy'."""
        processed, manifest = _make_simple_dataset(
            tmp_path,
            difficulties=["Easy", "Hard", "ExpertPlus"] * 4,  # 12 songs to ensure train split has several
            characteristics=["Standard"] * 12,
            bpms=[120.0] * 12,
        )
        config = ModelConfig(min_difficulty="Easy", max_seq_len=64)
        ds = BeatSaberDataset(processed, manifest, config, split="train")
        # With 12 songs and 80/10/10 split, train should have ~9-10
        assert len(ds.samples) >= 5
        diffs = {s["difficulty"] for s in ds.samples}
        # Should include multiple difficulty levels
        assert len(diffs) >= 2


class TestFilterByCharacteristic:
    def test_standard_only(self, tmp_path):
        """Only Standard samples when characteristics=['Standard']."""
        processed, manifest = _make_simple_dataset(
            tmp_path,
            difficulties=["Expert", "Expert", "Expert"],
            characteristics=["Standard", "OneSaber", "NoArrows"],
            bpms=[120.0] * 3,
        )
        config = ModelConfig(
            characteristics=["Standard"], max_seq_len=64,
        )
        ds = BeatSaberDataset(processed, manifest, config, split="train")
        chars = {s["characteristic"] for s in ds.samples}
        assert chars <= {"Standard"}


class TestFilterByBpm:
    def test_bpm_range(self, tmp_path):
        """Samples outside [50, 300] excluded."""
        processed, manifest = _make_simple_dataset(
            tmp_path,
            difficulties=["Expert"] * 4,
            characteristics=["Standard"] * 4,
            bpms=[30.0, 100.0, 200.0, 400.0],
        )
        config = ModelConfig(min_bpm=50.0, max_bpm=300.0, max_seq_len=64)
        ds = BeatSaberDataset(processed, manifest, config, split="train")
        for s in ds.samples:
            assert 50.0 <= s["bpm"] <= 300.0


class TestSpecAugment:
    def test_training_only(self):
        """SpecAugment modifies mel in train split."""
        mel = np.ones((80, 100), dtype=np.float32)
        augmented = BeatSaberDataset._spec_augment(mel)
        # Should have some zeros from masking
        assert np.any(augmented == 0.0)

    def test_shape_preserved(self):
        """Output shape unchanged after augmentation."""
        mel = np.random.randn(80, 200).astype(np.float32)
        augmented = BeatSaberDataset._spec_augment(mel)
        assert augmented.shape == mel.shape

    def test_does_not_mutate_input(self):
        """SpecAugment should not modify the input array."""
        mel = np.ones((80, 100), dtype=np.float32)
        _ = BeatSaberDataset._spec_augment(mel)
        assert np.all(mel == 1.0)


class TestCacheVersioning:
    def test_version_key_changes_with_onset(self):
        """Cache version changes when use_onset_features changes."""
        config1 = ModelConfig(use_onset_features=False)
        config2 = ModelConfig(use_onset_features=True)
        assert _cache_version_key(config1) != _cache_version_key(config2)

    def test_version_key_changes_with_n_mels(self):
        """Cache version changes when n_mels changes."""
        config1 = ModelConfig(n_mels=80)
        config2 = ModelConfig(n_mels=128)
        assert _cache_version_key(config1) != _cache_version_key(config2)

    def test_version_key_stable(self):
        """Same config produces same version key."""
        config1 = ModelConfig()
        config2 = ModelConfig()
        assert _cache_version_key(config1) == _cache_version_key(config2)
