# Beat Saber Track Generator

An AI-powered mod for Beat Saber that generates custom note maps from audio files.

## Project Overview

Given an audio file as input, the system generates block positions and orientations (for both left and right sabers) at a regular sample rate (~20 Hz). The goal is a machine learning model that produces playable, enjoyable Beat Saber tracks automatically.

## Architecture

1. **Data Pipeline** (complete) — Extract, parse, normalize Beat Saber maps into Parquet
2. **ML Model** (complete) — Encoder-decoder transformer for audio → token sequence generation
3. **Baseline Training** (complete) — 13 epochs on 23K songs, 60.6% token accuracy, generates playable maps
4. **Feedback System** (future) — In-game mechanism for player feedback to improve the model

## Beat Saber Map Format Quick Reference

Maps are **JSON files** in folders. See [LEARNINGS.md](LEARNINGS.md) for full format details. Implementation plans are in [plans/](plans/).

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
- `beat_weaver.model.transformer` — AudioEncoder + TokenDecoder + BeatWeaverModel
- `beat_weaver.model.inference` — autoregressive generation with grammar mask
- `beat_weaver.model.exporter` — token sequence → playable v2 map folder
- `beat_weaver.model.evaluate` — onset F1, NPS accuracy, parity, diversity metrics

**Output format:** `data/processed/notes_NNNN.parquet` (one row group per song, split at 1 GB) with columns: song_hash, source, difficulty, characteristic, bpm, beat, time_seconds, x, y, color, cut_direction, angle_offset. Reader (`read_notes_parquet`) handles both multi-file and legacy single-file layouts.

**Model configs:** `configs/small.json` (1M params, batch_size=32, ~10min/epoch on full dataset) for fast iteration. Default config (44.5M params) for full training.

**Tests:** `python -m pytest tests/ -v` (131 tests; ML tests skipped without `.[ml]` deps)

**Training data:** 23,588 songs (23,375 BeatSaver + 213 official), 42,542 training samples, 40.3M notes total. Mel spectrograms pre-cached to `data/processed/mel_cache/` (~23K `.npy` files, ~30GB).

## ML Model

See [LEARNINGS.md](LEARNINGS.md) for research details, [plans/002-ml-model.md](plans/002-ml-model.md) for implementation plan.

- **Architecture:** Encoder-decoder transformer (44.5M params default, 1M params small config)
- **Audio input:** Log-mel spectrogram (80 bins, sr=22050, hop=512), beat-aligned to 1/16th note grid
- **Scope:** Color notes only (no bombs, walls, arcs, chains until core model performs well)
- **Output:** Beat-quantized compound tokens (291 vocab) → v2 Beat Saber JSON
- **Token format:** `START DIFF BAR POS LEFT RIGHT ... BAR ... END`
- **Training:** Cross-entropy with label smoothing, AdamW + cosine LR, mixed-precision, early stopping, weighted sampling (official 20% of batch, custom weighted by score)
- **Inference:** Autoregressive with grammar-constrained decoding (strictly increasing POS per bar), temperature/top-k/top-p sampling
- **Evaluation:** Onset F1, parity violation rate, NPS accuracy, beat alignment, pattern diversity
- **Mel pre-caching:** `warm_mel_cache()` computes all spectrograms in parallel before training starts (ProcessPoolExecutor, ~25min for 14K songs)
- **Data filtering:** Out-of-grid notes (mapping extensions) filtered during pre-tokenization; difficulty names case-insensitive with `Expert+` alias

## Baseline Training Results (small config, 23K songs)

Best model after 16 epochs: **val_loss=2.055, 60.6% token accuracy**. Model plateaued — 1M params saturated on 42K samples. Generates playable maps in-game. Known issues: color imbalance (skews red), NPS sometimes too high/low.

**Checkpoint resume:** Must save ALL training state (model, optimizer, scheduler, GradScaler). Missing GradScaler was root cause of NaN on resume (fresh scaler at scale=65536 causes overflow). Now saves `scheduler.pt` + `scaler.pt`; fallbacks handle old checkpoints.

## Open Questions

- **Larger model:** Try 4-layer/256d (~10M params) now that data pipeline is proven at scale
- **Color balance:** Model generates ~70% red notes; may need loss weighting or data augmentation
- **Full-song generation:** Audio truncated to max_audio_len=4096 frames (~95s at 22050Hz/512hop); need windowed generation for full songs
- **Feedback capture system:** In-game mechanism to collect player feedback (later phase)
- **RL fine-tuning:** After supervised pretraining, fine-tune with player feedback reward model
