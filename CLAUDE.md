# Beat Saber Track Generator

An AI-powered mod for Beat Saber that generates custom note maps from audio files.

## Project Overview

Given an audio file as input, the system generates block positions and orientations (for both left and right sabers) at a regular sample rate (~20 Hz). The goal is a machine learning model that produces playable, enjoyable Beat Saber tracks automatically.

## Architecture

1. **Data Pipeline** (complete) — Extract, parse, normalize Beat Saber maps into Parquet
2. **ML Model** (complete) — Encoder-decoder transformer with Conformer audio encoder for audio → token sequence generation
3. **Baseline Training** (complete) — 16 epochs on 23K songs, 60.6% token accuracy, generates playable maps
4. **Model Improvements** (complete) — Filtering, SpecAugment, onset features, RoPE, color balance loss, medium config
5. **Conformer Encoder** (complete) — Conformer blocks (conv + attention) replace pure transformer encoder, now default
6. **Full-Song Generation** (complete) — Windowed inference with overlap stitching for songs of any length
7. **Feedback System** (future) — In-game mechanism for player feedback to improve the model

## Beat Saber Map Format Quick Reference

Maps are **JSON files** in folders. See [RESEARCH.md](RESEARCH.md) for full format details. Implementation plans are in [plans/](plans/).

- **Schemas:** v2 (most custom maps), v3, v4 (latest official). All JSON-based, `.dat` extension.
- **Entry point:** `Info.dat` — song metadata, references difficulty files
- **Grid:** 4 columns (x: 0-3) x 3 rows (y: 0-2)
- **Colors:** 0 = Red/Left, 1 = Blue/Right
- **Cut directions:** 0=Up, 1=Down, 2=Left, 3=Right, 4=UpLeft, 5=UpRight, 6=DownLeft, 7=DownRight, 8=Any
- **Timing:** All in beats (float). Convert: `seconds = beat * 60.0 / BPM`
- **Custom maps location:** `<Beat Saber>/Beat Saber_Data/CustomLevels/`
- **Official maps:** Extracted from Unity bundles via `UnityPy` — 214 levels (65 base + 149 DLC) with WAV audio
- **DLC maps location:** `<Beat Saber>/DLC/Levels/<LevelName>/<bundlefile>` — individual bundles per level
- **Training data sources:** Custom maps from [BeatSaver](https://beatsaver.com/) (v2 JSON, score>=0.9/upvotes>=5) + 214 official levels (v4 gzip JSON + WAV audio)
- **Local install:** `C:\Program Files (x86)\Steam\steamapps\common\Beat Saber`

## CLI (`beat-weaver`)

Install: `pip install -e .` (core) or `pip install -e ".[ml]"` (with ML dependencies)

| Command | Description |
|---------|-------------|
| `beat-weaver download` | Download custom maps from BeatSaver API |
| `beat-weaver extract-official` | Extract official maps from Unity bundles |
| `beat-weaver build-manifest` | Build audio manifest from raw map folders |
| `beat-weaver process` | Normalize raw maps to Parquet |
| `beat-weaver run` | Full pipeline (all sources) |
| `beat-weaver train` | Train the ML model |
| `beat-weaver generate` | Generate a Beat Saber map from audio |
| `beat-weaver evaluate` | Evaluate model on test data |

**Key modules:**
- `beat_weaver.parsers.beatmap_parser.parse_map_folder(path)` — parse any map folder
- `beat_weaver.sources.beatsaver` — BeatSaver API client + downloader
- `beat_weaver.sources.unity_extractor` — official map + audio extraction from Unity bundles (base + DLC)
- `beat_weaver.storage.writer` — Parquet output (notes/bombs/obstacles)
- `beat_weaver.model.tokenizer` — encode/decode beatmaps ↔ token sequences (291 vocab)
- `beat_weaver.model.audio` — mel spectrogram extraction, beat-aligned framing, BPM auto-detection
- `beat_weaver.model.transformer` — AudioEncoder (Conformer or Transformer) + TokenDecoder + BeatWeaverModel (RoPE or sinusoidal PE)
- `beat_weaver.model.inference` — autoregressive generation with grammar mask + windowed full-song generation
- `beat_weaver.model.exporter` — token sequence or note list → playable v2 map folder
- `beat_weaver.model.evaluate` — onset F1, NPS accuracy, parity, diversity metrics

**Output format:** `data/processed/notes_NNNN.parquet` (one row group per song, split at 1 GB) with columns: song_hash, source, difficulty, characteristic, bpm, beat, time_seconds, x, y, color, cut_direction, angle_offset. Reader (`read_notes_parquet`) handles both multi-file and legacy single-file layouts.

**Model configs:** `configs/small.json` (1M params, batch_size=32) for fast iteration. `configs/medium.json` (6.5M params, 4L/256d, seq_len=4096, Expert+ only, onset features) for standard transformer training. `configs/medium_conformer.json` (9.4M params, Conformer encoder, LR=3e-5) for Conformer training on 8GB VRAM. Default config (44.5M params) for full training.

**Tests:** `python -m pytest tests/ -v` (178 tests; ML tests skipped without `.[ml]` deps)

**Training data:** 23,588 songs (23,375 BeatSaver + 213 official), 42,542 training samples, 40.3M notes total. Mel spectrograms pre-cached to `data/processed/mel_cache/` (~23K `.npy` files, ~30GB). Cache auto-invalidates when audio feature config changes (VERSION file).

## ML Model

See [RESEARCH.md](RESEARCH.md) for research details, [plans/002-ml-model.md](plans/002-ml-model.md) for implementation plan.

- **Architecture:** Encoder-decoder transformer (44.5M default, 9.4M medium conformer, 6.5M medium standard, 1M small)
- **Audio encoder:** Conformer (default) or standard Transformer. Conformer blocks use FFN/2 + Self-Attention + DepthwiseConv + FFN/2 + LayerNorm (Gulati et al., 2020). Config: `use_conformer=True` (default), `conformer_kernel_size=31`.
- **Audio input:** Log-mel spectrogram (80 bins, sr=22050, hop=512), beat-aligned to 1/16th note grid. Optional onset strength channel (+1 bin).
- **Positional encoding:** RoPE (default) or sinusoidal (config.use_rope=False). RoPE applied to self-attention Q/K only (not cross-attention).
- **Scope:** Color notes only (no bombs, walls, arcs, chains until core model performs well)
- **Output:** Beat-quantized compound tokens (291 vocab) → v2 Beat Saber JSON
- **Token format:** `START DIFF BAR POS LEFT RIGHT ... BAR ... END`
- **Training:** Cross-entropy with label smoothing + optional color balance auxiliary loss, AdamW + cosine LR, mixed-precision, early stopping, weighted sampling (official 20% of batch, custom weighted by BeatSaver score)
- **Data augmentation:** SpecAugment (time/frequency masking, training split only)
- **Data filtering:** By difficulty (min_difficulty), characteristic, BPM range. Out-of-grid notes filtered during pre-tokenization.
- **Inference:** Autoregressive with grammar-constrained decoding (strictly increasing POS per bar), temperature/top-k/top-p sampling, windowed full-song generation with overlap stitching
- **Evaluation:** Onset F1, parity violation rate, NPS accuracy, beat alignment, pattern diversity
- **Mel pre-caching:** `warm_mel_cache()` computes all spectrograms in parallel before training starts (ProcessPoolExecutor, ~25min for 14K songs). Cache versioned and auto-invalidated on config change.
- **Full-song generation:** `generate_full_song()` processes audio in overlapping windows (25% overlap, capped at 1024 frames), decodes each window's tokens to notes with beat offsets, and merges using midpoint ownership to eliminate duplicates. Songs of any length are supported. `export_notes()` writes a merged note list directly to v2 map format.

## Training Results

### Baseline (small config, 1M params, 23K songs)

Best model after 16 epochs: **val_loss=2.055, 60.6% token accuracy**. Model plateaued — 1M params saturated on 42K samples. Generates playable maps in-game. Known issues: color imbalance (skews red), NPS sometimes too high/low.

### Medium Standard Transformer (6.5M params)

Training diverged after epoch 4 with `learning_rate=1e-4`: train loss dropped (4.26 → 2.0) but val loss exploded (5.26 → 13.4). Root cause: LR too aggressive for the larger model — not overfitting but optimization instability.

### Medium Conformer (9.4M params) — current best

Training with `configs/medium_conformer.json` (LR=3e-5, warmup=1000, batch_size=8). First 7 epochs:

| Epoch | Train Loss | Val Loss | Val Acc | Time/epoch |
|-------|-----------|----------|---------|------------|
| 1 | 4.046 | 3.277 | 37.8% | ~39 min |
| 4 | 2.341 | 2.530 | 52.1% | ~37 min |
| 7 | 2.189 | 2.396 | 53.6% | ~39 min |

Val loss steadily decreasing with no divergence. Already surpassing the small model's accuracy (53.6% at epoch 7 vs 60.6% at epoch 16) and on track to exceed it. Training is ongoing.

**Key lesson:** Medium+ models need lower LR (3e-5 vs 1e-4) and longer warmup (1000 vs 500 steps). The Conformer's conv module helps capture local audio patterns (onsets, transients) that pure attention misses.

**Checkpoint resume:** Must save ALL training state (model, optimizer, scheduler, GradScaler). Missing GradScaler was root cause of NaN on resume (fresh scaler at scale=65536 causes overflow). Now saves `scheduler.pt` + `scaler.pt`; fallbacks handle old checkpoints.

## Open Questions

- **Conformer training completion:** Currently training — expect val_loss < 2.0 and accuracy > 60% based on trajectory
- **Feedback capture system:** In-game mechanism to collect player feedback (later phase)
- **RL fine-tuning:** After supervised pretraining, fine-tune with player feedback reward model
