"""PyTorch Dataset for Beat Saber map training.

Produces (mel_spectrogram, token_ids, token_mask) tuples from Parquet data + audio.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from beat_weaver.model.audio import (
    beat_align_spectrogram,
    compute_mel_spectrogram,
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

    def __init__(
        self,
        processed_dir: Path,
        audio_manifest_path: Path,
        config: ModelConfig,
        split: str = "train",
    ) -> None:
        self.config = config
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

        # Mel spectrogram cache directory
        self.mel_cache_dir = self.processed_dir / "mel_cache"
        self.mel_cache_dir.mkdir(parents=True, exist_ok=True)

        # Load notes from Parquet using pandas groupby (vectorized)
        import pyarrow.parquet as pq

        notes_path = self.processed_dir / "notes.parquet"
        table = pq.read_table(notes_path)
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

        for (song_hash, difficulty, characteristic), group in grouped:
            if song_hash not in split_hashes:
                continue
            if song_hash not in self.audio_manifest:
                continue
            note_dicts = group[note_cols].to_dict("records")
            meta = self.metadata.get(song_hash, {})
            self.samples.append({
                "song_hash": song_hash,
                "difficulty": difficulty,
                "characteristic": characteristic,
                "notes": note_dicts,
                "bpm": note_dicts[0]["bpm"],
                "source": meta.get("source", "unknown"),
                "score": meta.get("score"),
            })

        # Pre-tokenize all samples (deterministic — no need to repeat per epoch)
        for sample in self.samples:
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
            ]
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

        logger.info(
            "BeatSaberDataset(%s): %d samples from %d songs",
            split, len(self.samples), len(split_hashes),
        )

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

        return (
            torch.from_numpy(mel),                          # (n_mels, T_audio)
            torch.tensor(token_ids, dtype=torch.long),      # (max_seq_len,)
            torch.tensor(mask, dtype=torch.bool),           # (max_seq_len,)
        )


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
