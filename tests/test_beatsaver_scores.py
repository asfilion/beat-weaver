"""Tests for BeatSaver score persistence in the pipeline."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from beat_weaver.sources.beatsaver import load_beatsaver_meta


class TestBeatsaverMeta:
    def test_load_meta_exists(self, tmp_path):
        """Load _beatsaver_meta.json from a map directory."""
        meta = {
            "id": "abc123",
            "stats": {"score": 0.95, "upvotes": 100, "downvotes": 5},
        }
        meta_path = tmp_path / "_beatsaver_meta.json"
        meta_path.write_text(json.dumps(meta))

        loaded = load_beatsaver_meta(tmp_path)
        assert loaded is not None
        assert loaded["stats"]["score"] == 0.95
        assert loaded["stats"]["upvotes"] == 100

    def test_load_meta_missing(self, tmp_path):
        """Returns None when no _beatsaver_meta.json exists."""
        result = load_beatsaver_meta(tmp_path)
        assert result is None


class TestScoreInjection:
    def test_scores_injected_into_metadata(self, tmp_path):
        """Scores from _beatsaver_meta.json are injected into processed beatmaps."""
        from beat_weaver.cli import _process_single_folder

        # Create a minimal map folder with Info.dat + _beatsaver_meta.json
        map_folder = tmp_path / "test_map"
        map_folder.mkdir()

        info = {
            "_version": "2.0.0",
            "_songName": "Test Song",
            "_songSubName": "",
            "_songAuthorName": "Author",
            "_levelAuthorName": "Mapper",
            "_beatsPerMinute": 120,
            "_songFilename": "song.ogg",
            "_difficultyBeatmapSets": [{
                "_beatmapCharacteristicName": "Standard",
                "_difficultyBeatmaps": [{
                    "_difficulty": "Expert",
                    "_difficultyRank": 7,
                    "_beatmapFilename": "ExpertStandard.dat",
                    "_noteJumpMovementSpeed": 16,
                    "_noteJumpStartBeatOffset": 0,
                }],
            }],
        }
        (map_folder / "Info.dat").write_text(json.dumps(info))

        # Create a v2 difficulty file with a single note
        diff_data = {
            "_version": "2.0.0",
            "_notes": [
                {"_time": 1.0, "_lineIndex": 1, "_lineLayer": 0,
                 "_type": 0, "_cutDirection": 1},
            ],
            "_events": [],
            "_obstacles": [],
        }
        (map_folder / "ExpertStandard.dat").write_text(json.dumps(diff_data))

        # Create _beatsaver_meta.json with score data
        meta = {
            "id": "abc",
            "stats": {"score": 0.92, "upvotes": 50, "downvotes": 3},
        }
        (map_folder / "_beatsaver_meta.json").write_text(json.dumps(meta))

        # Process the folder
        beatmaps = _process_single_folder(map_folder, source="beatsaver")
        assert len(beatmaps) >= 1
        assert beatmaps[0].metadata.score == 0.92
        assert beatmaps[0].metadata.upvotes == 50
        assert beatmaps[0].metadata.downvotes == 3


def _make_backfill_test_data(tmp_path, scores_in_metadata):
    """Create test data for score backfill tests.

    Args:
        scores_in_metadata: dict mapping hash -> score to put in metadata.json.
            Use None for missing scores (the bug scenario).
    """
    from beat_weaver.storage.writer import NOTES_SCHEMA

    processed = tmp_path / "processed"
    processed.mkdir()

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

    for i, (h, score) in enumerate(scores_in_metadata.items()):
        song_hashes.extend([h] * 4)
        diff_list.extend(["Expert"] * 4)
        char_list.extend(["Standard"] * 4)
        beats.extend([0.0, 1.0, 2.0, 3.0])
        times.extend([0.0, 0.5, 1.0, 1.5])
        xs.extend([0, 1, 2, 3])
        ys.extend([0, 0, 0, 0])
        colors.extend([0, 1, 0, 1])
        cut_dirs.extend([1, 1, 1, 1])
        angle_offsets.extend([0, 0, 0, 0])
        bpm_col.extend([120.0] * 4)
        sources.extend(["beatsaver"] * 4)

        metadata.append({
            "hash": h, "source": "beatsaver", "source_id": h,
            "song_name": f"Song {i}", "bpm": 120.0, "score": score,
        })

    notes_data = {
        "song_hash": song_hashes, "source": sources,
        "difficulty": diff_list, "characteristic": char_list,
        "bpm": bpm_col, "beat": beats, "time_seconds": times,
        "x": xs, "y": ys, "color": colors,
        "cut_direction": cut_dirs, "angle_offset": angle_offsets,
    }
    table = pa.table(notes_data, schema=NOTES_SCHEMA)
    pq.write_table(table, processed / "notes.parquet")
    (processed / "metadata.json").write_text(json.dumps(metadata))

    return processed


class TestScoreBackfill:
    """Regression tests for BeatSaver score back-filling from raw metadata."""

    def _make_raw_folders(self, tmp_path, score_map):
        """Create raw map folders with _beatsaver_meta.json and audio manifest."""
        manifest = {}
        for h, score in score_map.items():
            raw_dir = tmp_path / "raw" / h
            raw_dir.mkdir(parents=True)
            dummy_audio = raw_dir / "song.egg"
            dummy_audio.write_bytes(b"fake audio")
            manifest[h] = str(dummy_audio)
            if score is not None:
                bs_meta = {"stats": {"score": score, "upvotes": 10, "downvotes": 1}}
                (raw_dir / "_beatsaver_meta.json").write_text(json.dumps(bs_meta))

        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        return manifest_path

    def test_backfill_patches_none_scores(self, tmp_path):
        """Scores missing from metadata.json are loaded from _beatsaver_meta.json."""
        torch = pytest.importorskip("torch")
        from beat_weaver.model.config import ModelConfig
        from beat_weaver.model.dataset import BeatSaberDataset

        # 12 songs so train split has enough after 80/10/10
        hashes = {f"hash_{i:04d}": None for i in range(12)}
        raw_scores = {f"hash_{i:04d}": 0.85 + i * 0.01 for i in range(12)}

        processed = _make_backfill_test_data(tmp_path, hashes)
        manifest_path = self._make_raw_folders(tmp_path, raw_scores)

        # Pre-cache dummy mel spectrograms
        mel_cache = processed / "mel_cache"
        mel_cache.mkdir()
        import numpy as np
        config = ModelConfig()
        for h in hashes:
            mel = np.zeros((config.n_mels, 100), dtype=np.float32)
            np.save(mel_cache / f"{h}_120.0.npy", mel)

        ds = BeatSaberDataset(processed, manifest_path, config, split="train")

        # Verify scores were back-filled into samples
        for sample in ds.samples:
            assert sample["score"] is not None, (
                f"Score not back-filled for {sample['song_hash']}"
            )
            expected = raw_scores[sample["song_hash"]]
            assert sample["score"] == expected

    def test_no_backfill_when_scores_present(self, tmp_path):
        """No backfill attempted when metadata.json already has scores."""
        torch = pytest.importorskip("torch")
        from beat_weaver.model.config import ModelConfig
        from beat_weaver.model.dataset import BeatSaberDataset

        hashes = {f"hash_{i:04d}": 0.95 for i in range(12)}
        raw_scores = {f"hash_{i:04d}": 0.50 for i in range(12)}  # different!

        processed = _make_backfill_test_data(tmp_path, hashes)
        manifest_path = self._make_raw_folders(tmp_path, raw_scores)

        mel_cache = processed / "mel_cache"
        mel_cache.mkdir()
        import numpy as np
        config = ModelConfig()
        for h in hashes:
            mel = np.zeros((config.n_mels, 100), dtype=np.float32)
            np.save(mel_cache / f"{h}_120.0.npy", mel)

        ds = BeatSaberDataset(processed, manifest_path, config, split="train")

        # Scores should remain at 0.95, not overwritten with 0.50
        for sample in ds.samples:
            assert sample["score"] == 0.95

    def test_backfill_handles_missing_meta_file(self, tmp_path):
        """Backfill gracefully skips songs without _beatsaver_meta.json."""
        torch = pytest.importorskip("torch")
        from beat_weaver.model.config import ModelConfig
        from beat_weaver.model.dataset import BeatSaberDataset

        hashes = {f"hash_{i:04d}": None for i in range(12)}
        # Only provide raw meta for half the songs
        raw_scores = {f"hash_{i:04d}": 0.90 for i in range(6)}

        processed = _make_backfill_test_data(tmp_path, hashes)
        manifest_path = self._make_raw_folders(tmp_path, raw_scores)

        mel_cache = processed / "mel_cache"
        mel_cache.mkdir()
        import numpy as np
        config = ModelConfig()
        for h in hashes:
            mel = np.zeros((config.n_mels, 100), dtype=np.float32)
            np.save(mel_cache / f"{h}_120.0.npy", mel)

        ds = BeatSaberDataset(processed, manifest_path, config, split="train")

        for sample in ds.samples:
            h = sample["song_hash"]
            idx = int(h.split("_")[1])
            if idx < 6:
                assert sample["score"] == 0.90
            else:
                assert sample["score"] is None

    def test_backfill_scores_affect_weighted_sampler(self, tmp_path):
        """Back-filled scores produce non-uniform weights in the sampler."""
        torch = pytest.importorskip("torch")
        from beat_weaver.model.config import ModelConfig
        from beat_weaver.model.dataset import BeatSaberDataset, build_weighted_sampler
        from beat_weaver.storage.writer import NOTES_SCHEMA

        # Need both official and custom sources for weighted sampler to activate.
        # Build data manually to include both sources.
        processed = tmp_path / "processed"
        processed.mkdir()

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
        source_col = []
        metadata = []

        all_hashes = {}
        # 10 beatsaver songs with None scores + 2 official songs
        for i in range(12):
            h = f"hash_{i:04d}"
            src = "official" if i >= 10 else "beatsaver"
            score = None if src == "beatsaver" else None
            all_hashes[h] = (src, score)
            song_hashes.extend([h] * 4)
            diff_list.extend(["Expert"] * 4)
            char_list.extend(["Standard"] * 4)
            beats.extend([0.0, 1.0, 2.0, 3.0])
            times.extend([0.0, 0.5, 1.0, 1.5])
            xs.extend([0, 1, 2, 3])
            ys.extend([0, 0, 0, 0])
            colors.extend([0, 1, 0, 1])
            cut_dirs.extend([1, 1, 1, 1])
            angle_offsets.extend([0, 0, 0, 0])
            bpm_col.extend([120.0] * 4)
            source_col.extend([src] * 4)
            metadata.append({
                "hash": h, "source": src, "source_id": h,
                "song_name": f"Song {i}", "bpm": 120.0, "score": score,
            })

        notes_data = {
            "song_hash": song_hashes, "source": source_col,
            "difficulty": diff_list, "characteristic": char_list,
            "bpm": bpm_col, "beat": beats, "time_seconds": times,
            "x": xs, "y": ys, "color": colors,
            "cut_direction": cut_dirs, "angle_offset": angle_offsets,
        }
        table = pa.table(notes_data, schema=NOTES_SCHEMA)
        pq.write_table(table, processed / "notes.parquet")
        (processed / "metadata.json").write_text(json.dumps(metadata))

        # Create raw folders with varying scores for beatsaver songs
        manifest = {}
        raw_scores = {}
        import numpy as np
        for h, (src, _) in all_hashes.items():
            raw_dir = tmp_path / "raw" / h
            raw_dir.mkdir(parents=True)
            dummy_audio = raw_dir / "song.egg"
            dummy_audio.write_bytes(b"fake")
            manifest[h] = str(dummy_audio)
            if src == "beatsaver":
                idx = int(h.split("_")[1])
                score = 0.50 + idx * 0.05  # varying scores
                raw_scores[h] = score
                bs_meta = {"stats": {"score": score, "upvotes": 10, "downvotes": 1}}
                (raw_dir / "_beatsaver_meta.json").write_text(json.dumps(bs_meta))
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        # Pre-cache dummy mels
        mel_cache = processed / "mel_cache"
        mel_cache.mkdir()
        config = ModelConfig()
        for h in all_hashes:
            mel = np.zeros((config.n_mels, 100), dtype=np.float32)
            np.save(mel_cache / f"{h}_120.0.npy", mel)

        ds = BeatSaberDataset(processed, manifest_path, config, split="train")
        sampler = build_weighted_sampler(ds, official_ratio=0.2)
        assert sampler is not None, "Sampler should be created with mixed sources"

        # With back-filled scores, custom weights should NOT all be 1.0
        weights = [w.item() if hasattr(w, 'item') else w for w in sampler.weights]
        custom_weights = [w for w in weights if w > 0 and w != weights[0]]
        unique_custom = set(round(w, 4) for w in custom_weights)
        assert len(unique_custom) > 1, (
            f"All custom weights are identical ({unique_custom}), scores not affecting sampler"
        )
