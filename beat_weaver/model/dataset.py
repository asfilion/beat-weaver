"""PyTorch Dataset for Beat Saber map training.

Produces (mel_spectrogram, token_ids, token_mask) tuples from Parquet data + audio.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from beat_weaver.model.audio import (
    beat_align_spectrogram,
    compute_mel_spectrogram,
    compute_mel_with_onset,
    load_audio,
    load_manifest,
)
from beat_weaver.model.config import ModelConfig
from beat_weaver.model.tokenizer import encode_beatmap
from beat_weaver.schemas.normalized import (
    DifficultyInfo,
    Note,
    NormalizedBeatmap,
    SongMetadata,
)

logger = logging.getLogger(__name__)


def _compute_one_mel(
    audio_path: str, song_hash: str, bpm: float, cache_path: str,
    sr: int, n_mels: int, n_fft: int, hop_length: int,
    use_onset: bool = False,
) -> str | None:
    """Compute and cache one mel spectrogram. Returns song_hash on success, None on error."""
    try:
        audio, actual_sr = load_audio(Path(audio_path), sr=sr)
        if use_onset:
            mel = compute_mel_with_onset(
                audio, sr=actual_sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
            )
        else:
            mel = compute_mel_spectrogram(
                audio, sr=actual_sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
            )
        mel = beat_align_spectrogram(mel, sr=actual_sr, hop_length=hop_length, bpm=bpm)
        np.save(cache_path, mel)
        return song_hash
    except Exception as e:
        logger.warning("Failed to compute mel for %s: %s", song_hash, e)
        return None


def _cache_version_key(config: ModelConfig) -> str:
    """Compute a version string from feature-relevant config fields."""
    import hashlib
    key = f"{config.n_mels}_{config.n_fft}_{config.hop_length}_{config.sample_rate}_{config.use_onset_features}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def warm_mel_cache(
    processed_dir: Path,
    audio_manifest_path: Path,
    config: ModelConfig,
    max_workers: int | None = None,
) -> int:
    """Pre-compute mel spectrograms in parallel for all songs in the manifest.

    Skips songs that already have a cached mel file. Returns the number of
    newly computed spectrograms.
    """
    cache_dir = processed_dir / "mel_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check cache version — invalidate if feature config changed
    version_file = cache_dir / "VERSION"
    expected_version = _cache_version_key(config)
    needs_clear = False
    if version_file.exists():
        current_version = version_file.read_text().strip()
        if current_version != expected_version:
            needs_clear = True
    else:
        # No VERSION file but cache files exist — legacy cache, must clear
        if any(cache_dir.glob("*.npy")):
            needs_clear = True
    if needs_clear:
        n_stale = sum(1 for _ in cache_dir.glob("*.npy"))
        logger.warning(
            "Mel cache version mismatch (need %s). "
            "Clearing %d stale files and recomputing.",
            expected_version, n_stale,
        )
        for npy_file in cache_dir.glob("*.npy"):
            npy_file.unlink()
    version_file.write_text(expected_version)

    manifest = load_manifest(audio_manifest_path)

    # Load BPMs from metadata
    meta_path = processed_dir / "metadata.json"
    with open(meta_path) as f:
        raw_meta = json.load(f)
    bpm_lookup: dict[str, float] = {}
    if isinstance(raw_meta, list):
        for m in raw_meta:
            bpm_lookup[m["hash"]] = m["bpm"]
    else:
        for h, m in raw_meta.items():
            bpm_lookup[h] = m["bpm"]

    # Find songs that need mel computation
    todo: list[tuple[str, str, float, str]] = []  # (audio_path, hash, bpm, cache_path)
    for song_hash, audio_path in manifest.items():
        bpm = bpm_lookup.get(song_hash)
        if bpm is None:
            continue
        cache_path = str(cache_dir / f"{song_hash}_{bpm}.npy")
        if not Path(cache_path).exists():
            todo.append((str(audio_path), song_hash, bpm, cache_path))

    if not todo:
        logger.info("Mel cache is warm: all %d songs already cached", len(manifest))
        return 0

    logger.info(
        "Warming mel cache: %d songs to compute (%d already cached)",
        len(todo), len(manifest) - len(todo),
    )

    computed = 0
    import os
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 8)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _compute_one_mel, audio_path, song_hash, bpm, cache_path,
                config.sample_rate, config.n_mels, config.n_fft, config.hop_length,
                config.use_onset_features,
            ): song_hash
            for audio_path, song_hash, bpm, cache_path in todo
        }
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result is not None:
                computed += 1
            if i % 500 == 0 or i == len(futures):
                logger.info("Mel cache progress: %d/%d computed", i, len(futures))

    logger.info("Mel cache warm: %d newly computed", computed)
    return computed


def _split_hashes(
    hashes: list[str], split: str, seed: int = 42,
) -> list[str]:
    """Deterministically split song hashes into train/val/test (80/10/10)."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(hashes))
    n_val = max(1, len(hashes) // 10)
    n_test = max(1, len(hashes) // 10)

    if split == "train":
        return [hashes[i] for i in indices[: len(hashes) - n_val - n_test]]
    elif split == "val":
        return [hashes[i] for i in indices[len(hashes) - n_val - n_test : len(hashes) - n_test]]
    elif split == "test":
        return [hashes[i] for i in indices[len(hashes) - n_test :]]
    else:
        raise ValueError(f"Unknown split: {split!r}")


class BeatSaberDataset(Dataset):
    """Dataset that loads Parquet note data + audio for training.

    Each item is one (song_hash, difficulty, characteristic) combination.
    Returns (mel_spectrogram, token_ids, token_mask) tensors.
    """

    # Difficulty rank for filtering (matches DifficultyInfo.difficulty_rank)
    _DIFF_RANK = {
        "Easy": 1, "Normal": 3, "Hard": 5, "Expert": 7, "ExpertPlus": 9,
    }

    def __init__(
        self,
        processed_dir: Path,
        audio_manifest_path: Path,
        config: ModelConfig,
        split: str = "train",
    ) -> None:
        self.config = config
        self.split = split
        self.processed_dir = Path(processed_dir)
        self.audio_manifest = load_manifest(audio_manifest_path)

        # Load metadata (writer produces a list; convert to dict keyed by hash)
        meta_path = self.processed_dir / "metadata.json"
        with open(meta_path) as f:
            raw_meta = json.load(f)
        if isinstance(raw_meta, list):
            self.metadata: dict[str, dict] = {m["hash"]: m for m in raw_meta}
        else:
            self.metadata = raw_meta

        # Back-fill BeatSaver scores from raw _beatsaver_meta.json files
        # when metadata.json was generated before score injection existed.
        self._backfill_beatsaver_scores()

        # Mel spectrogram cache directory
        self.mel_cache_dir = self.processed_dir / "mel_cache"
        self.mel_cache_dir.mkdir(parents=True, exist_ok=True)

        # Load notes from Parquet using pandas groupby (vectorized)
        from beat_weaver.storage.writer import read_notes_parquet

        table = read_notes_parquet(self.processed_dir)
        df = table.to_pandas()

        # Ensure angle_offset column exists
        if "angle_offset" not in df.columns:
            df["angle_offset"] = 0

        # Group notes by (song_hash, difficulty, characteristic)
        self.samples: list[dict] = []
        note_cols = ["beat", "time_seconds", "x", "y", "color",
                     "cut_direction", "angle_offset", "bpm"]
        grouped = df.groupby(["song_hash", "difficulty", "characteristic"])

        # Collect all unique hashes for splitting
        all_hashes = sorted(df["song_hash"].unique())
        split_hashes = set(_split_hashes(all_hashes, split))

        # Prepare filtering thresholds from config
        min_diff_rank = self._DIFF_RANK.get(config.min_difficulty, 1)
        allowed_chars = set(config.characteristics) if config.characteristics else None
        filtered_counts = {"difficulty": 0, "characteristic": 0, "bpm": 0}

        for (song_hash, difficulty, characteristic), group in grouped:
            if song_hash not in split_hashes:
                continue
            if song_hash not in self.audio_manifest:
                continue
            # Apply config-driven filters
            if self._DIFF_RANK.get(difficulty, 0) < min_diff_rank:
                filtered_counts["difficulty"] += 1
                continue
            if allowed_chars and characteristic not in allowed_chars:
                filtered_counts["characteristic"] += 1
                continue
            note_dicts = group[note_cols].to_dict("records")
            bpm = note_dicts[0]["bpm"]
            if bpm < config.min_bpm or bpm > config.max_bpm:
                filtered_counts["bpm"] += 1
                continue
            meta = self.metadata.get(song_hash, {})
            self.samples.append({
                "song_hash": song_hash,
                "difficulty": difficulty,
                "characteristic": characteristic,
                "notes": note_dicts,
                "bpm": bpm,
                "source": meta.get("source", "unknown"),
                "score": meta.get("score"),
            })

        for reason, count in filtered_counts.items():
            if count > 0:
                logger.info("Filtered %d samples by %s", count, reason)

        # Free the large DataFrame and Arrow table now that samples are built
        del df, table, grouped

        # Pre-tokenize all samples (deterministic — no need to repeat per epoch)
        skipped_samples = 0
        for sample in self.samples:
            # Filter out notes with coordinates outside the standard 4x3 grid
            # (mapping extension maps can have x=1000, y=3000, etc.)
            notes = [
                Note(
                    beat=n["beat"],
                    time_seconds=n["time_seconds"],
                    x=n["x"],
                    y=n["y"],
                    color=n["color"],
                    cut_direction=n["cut_direction"],
                    angle_offset=n.get("angle_offset", 0),
                )
                for n in sample["notes"]
                if 0 <= n["x"] < 4 and 0 <= n["y"] < 3
                and 0 <= n["cut_direction"] < 9 and n["color"] in (0, 1)
            ]
            if not notes:
                skipped_samples += 1
                sample["_skip"] = True
                continue
            meta = self.metadata.get(sample["song_hash"], {})
            beatmap = NormalizedBeatmap(
                metadata=SongMetadata(
                    source=meta.get("source", "unknown"),
                    source_id=meta.get("source_id", sample["song_hash"]),
                    hash=sample["song_hash"],
                    bpm=sample["bpm"],
                ),
                difficulty_info=DifficultyInfo(
                    characteristic=sample["characteristic"],
                    difficulty=sample["difficulty"],
                    difficulty_rank=0,
                    note_jump_speed=0.0,
                    note_jump_offset=0.0,
                ),
                notes=notes,
            )
            token_ids = encode_beatmap(beatmap)

            # Truncate or pad tokens
            max_len = self.config.max_seq_len
            if len(token_ids) > max_len:
                token_ids = token_ids[:max_len]
            mask = [True] * len(token_ids) + [False] * (max_len - len(token_ids))
            token_ids = token_ids + [0] * (max_len - len(token_ids))

            sample["token_ids"] = token_ids
            sample["token_mask"] = mask

        # Remove samples that had no valid notes after filtering
        if skipped_samples > 0:
            self.samples = [s for s in self.samples if not s.get("_skip")]
            logger.warning(
                "Filtered out %d samples with no standard-grid notes", skipped_samples,
            )

        # Free raw note dicts — only token_ids/token_mask are needed for training
        for sample in self.samples:
            sample.pop("notes", None)

        logger.info(
            "BeatSaberDataset(%s): %d samples from %d songs",
            split, len(self.samples), len(split_hashes),
        )

    def _backfill_beatsaver_scores(self) -> None:
        """Load BeatSaver scores from raw _beatsaver_meta.json files.

        When metadata.json was generated before score injection existed,
        all beatsaver scores will be None. This method reads scores from
        the original _beatsaver_meta.json files (located next to audio files
        referenced in the audio manifest) and patches the in-memory metadata.
        """
        # Check if any beatsaver entries are missing scores
        needs_backfill = any(
            m.get("source") == "beatsaver" and m.get("score") is None
            for m in self.metadata.values()
        )
        if not needs_backfill:
            return

        filled = 0
        for song_hash, meta in self.metadata.items():
            if meta.get("source") != "beatsaver" or meta.get("score") is not None:
                continue
            audio_path = self.audio_manifest.get(song_hash)
            if audio_path is None:
                continue
            meta_path = Path(audio_path).parent / "_beatsaver_meta.json"
            if not meta_path.exists():
                continue
            try:
                bs_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                stats = bs_meta.get("stats", {})
                score = stats.get("score")
                if score is not None:
                    meta["score"] = score
                    filled += 1
            except (json.JSONDecodeError, OSError):
                continue

        if filled > 0:
            logger.info("Back-filled BeatSaver scores for %d songs from raw metadata", filled)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        song_hash = sample["song_hash"]
        bpm = sample["bpm"]

        # Use pre-computed tokens
        token_ids = sample["token_ids"]
        mask = sample["token_mask"]

        # Load mel spectrogram from cache or compute and cache
        cache_path = self.mel_cache_dir / f"{song_hash}_{bpm}.npy"
        if cache_path.exists():
            mel = np.load(cache_path)
        else:
            audio_path = self.audio_manifest[song_hash]
            audio, sr = load_audio(Path(audio_path), sr=self.config.sample_rate)
            if self.config.use_onset_features:
                mel = compute_mel_with_onset(
                    audio, sr=sr,
                    n_mels=self.config.n_mels,
                    n_fft=self.config.n_fft,
                    hop_length=self.config.hop_length,
                )
            else:
                mel = compute_mel_spectrogram(
                    audio, sr=sr,
                    n_mels=self.config.n_mels,
                    n_fft=self.config.n_fft,
                    hop_length=self.config.hop_length,
                )
            mel = beat_align_spectrogram(
                mel, sr=sr, hop_length=self.config.hop_length, bpm=bpm,
            )
            np.save(cache_path, mel)

        # Truncate audio to max_audio_len to fit in VRAM
        if mel.shape[1] > self.config.max_audio_len:
            mel = mel[:, : self.config.max_audio_len]

        # SpecAugment: random time/frequency masking (training only)
        if self.split == "train":
            mel = self._spec_augment(mel)

        return (
            torch.from_numpy(mel),                          # (n_mels, T_audio)
            torch.tensor(token_ids, dtype=torch.long),      # (max_seq_len,)
            torch.tensor(mask, dtype=torch.bool),           # (max_seq_len,)
        )

    @staticmethod
    def _spec_augment(mel: np.ndarray) -> np.ndarray:
        """Apply SpecAugment: random time and frequency masking.

        Time mask width scales with sequence length so the masking fraction
        stays meaningful for long sequences (e.g. 4096 frames).
        """
        mel = mel.copy()  # Don't mutate cached data
        n_mels, n_frames = mel.shape
        if n_frames == 0:
            return mel
        # Frequency masking: 2 bands, width up to 10 bins
        for _ in range(2):
            f = np.random.randint(1, min(11, n_mels))
            f0 = np.random.randint(0, n_mels - f + 1)
            mel[f0:f0 + f, :] = 0.0
        # Time masking: scale width with sequence length (~2% of frames, min 20, max 100)
        max_t = max(20, min(100, n_frames // 50))
        for _ in range(2):
            t = np.random.randint(1, min(max_t + 1, n_frames))
            t0 = np.random.randint(0, n_frames - t + 1)
            mel[:, t0:t0 + t] = 0.0
        return mel


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate batch, padding mel spectrograms to the longest in the batch.

    Returns (mel, mel_mask, tokens, token_mask).
    """
    mels, tokens, masks = zip(*batch)

    # Pad mel spectrograms to max length in batch
    max_mel_len = max(m.shape[1] for m in mels)
    n_mels = mels[0].shape[0]

    mel_padded = torch.zeros(len(mels), n_mels, max_mel_len)
    mel_mask = torch.zeros(len(mels), max_mel_len, dtype=torch.bool)
    for i, m in enumerate(mels):
        length = m.shape[1]
        mel_padded[i, :, :length] = m
        mel_mask[i, :length] = True

    tokens_stacked = torch.stack(tokens)
    masks_stacked = torch.stack(masks)

    return mel_padded, mel_mask, tokens_stacked, masks_stacked


def build_weighted_sampler(
    dataset: BeatSaberDataset, official_ratio: float = 0.2,
) -> WeightedRandomSampler | None:
    """Build a WeightedRandomSampler that oversamples official maps.

    Official maps are weighted to fill ``official_ratio`` of each batch.
    Custom maps are weighted by their BeatSaver score (higher-rated maps
    sampled more often).

    Returns ``None`` if all samples come from a single source (no
    rebalancing needed).
    """
    official_indices = []
    custom_indices = []
    custom_scores: list[float] = []

    for i, sample in enumerate(dataset.samples):
        if sample["source"] == "official":
            official_indices.append(i)
        else:
            custom_indices.append(i)
            # Default to 1.0 if score is missing
            custom_scores.append(sample.get("score") or 1.0)

    n_official = len(official_indices)
    n_custom = len(custom_indices)

    # No rebalancing needed if only one source present
    if n_official == 0 or n_custom == 0:
        return None

    # Compute weights so official samples collectively account for
    # ``official_ratio`` of the total sampling probability:
    #   sum(w_official) / (sum(w_official) + sum(w_custom)) = official_ratio
    # Within custom maps, weight by score.
    sum_custom_scores = sum(custom_scores)
    w_official = (official_ratio * sum_custom_scores) / (n_official * (1.0 - official_ratio))

    weights = [0.0] * len(dataset)
    for i in official_indices:
        weights[i] = w_official
    for i, idx in enumerate(custom_indices):
        weights[idx] = custom_scores[i]

    logger.info(
        "Weighted sampler: %d official (w=%.4f), %d custom (mean_score=%.4f), "
        "target official_ratio=%.0f%%",
        n_official, w_official, n_custom,
        sum_custom_scores / n_custom, official_ratio * 100,
    )

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True,
    )
