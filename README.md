[![CI](https://github.com/asfilion/beat-weaver/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/asfilion/beat-weaver/actions/workflows/ci.yml)

# beat-weaver

AI-powered Beat Saber track generator — feed in a song, get a playable custom map.

## What is this?

Beat Weaver uses machine learning to automatically generate [Beat Saber](https://beatsaber.com/) note maps from audio files. Instead of manually placing blocks, you provide a song and the model outputs block positions, orientations, and timing for both sabers.

## Features

- **Audio-to-map generation** — provide an audio file, get a playable v2 Beat Saber map (BPM auto-detected or manual)
- **Difficulty selection** — generate for Easy, Normal, Hard, Expert, or ExpertPlus
- **Seeded generation** — use a fixed seed for repeatable tracks, or randomize for variety
- **Grammar-constrained decoding** — generated maps always follow valid Beat Saber structure
- **Quality metrics** — onset F1, parity violations, NPS accuracy, beat alignment, pattern diversity

## Installation

```bash
# Core (data pipeline only)
pip install -e .

# With ML model dependencies
pip install -e ".[ml]"

# Development
pip install -e ".[ml,dev]"
```

## Usage

### Data Pipeline

```bash
# Download community maps from BeatSaver (resumable, parallel)
beat-weaver download --min-score 0.9

# Extract official maps + DLC from Unity bundles
beat-weaver extract-official

# Process all raw maps into training-ready Parquet
beat-weaver process

# Or run the full pipeline at once
beat-weaver run
```

### Model Training

```bash
# Train the model
beat-weaver train --audio-manifest data/audio_manifest.json --data data/processed --output output/training

# Resume training from checkpoint
beat-weaver train --audio-manifest data/audio_manifest.json --resume output/training/checkpoints/epoch_050
```

### Map Generation

```bash
# Generate a Beat Saber map from audio (BPM auto-detected)
beat-weaver generate --checkpoint output/training/checkpoints/best --audio song.ogg --difficulty Expert --output my_map/

# With explicit BPM and seed for reproducibility
beat-weaver generate --checkpoint output/training/checkpoints/best --audio song.ogg --bpm 128 --seed 42
```

### Evaluation

```bash
# Evaluate model on test set
beat-weaver evaluate --checkpoint output/training/checkpoints/best --audio-manifest data/audio_manifest.json
```

## Architecture

An encoder-decoder model that takes a log-mel spectrogram as input and generates a sequence of beat-quantized tokens representing note placements.

```
Audio (mel spectrogram + onset) → [Audio Encoder] → [Token Decoder] → Token Sequence → v2 Beat Saber Map
```

- **Tokenizer:** 291-token vocabulary encoding difficulty, bar structure, beat positions, and compound note placements (position + direction per hand)
- **Encoder:** Linear projection + positional encoding (RoPE or sinusoidal) + Conformer encoder (conv + self-attention, default) or Transformer encoder
- **Decoder:** Token embedding + positional encoding + Transformer decoder with cross-attention
- **Audio features:** Log-mel spectrogram (80 bins) with optional onset strength channel
- **Training:** SpecAugment, color balance loss, dataset filtering by difficulty/characteristic/BPM
- **Inference:** Autoregressive generation with grammar constraints ensuring valid map structure
- **Configs:** Small (1M params), medium (6.5M params), medium conformer (9.4M params, 8GB VRAM), default (44.5M params)

See [RESEARCH.md](RESEARCH.md) for research details and [plans/](plans/) for implementation plans.

## Project Status

- **Data pipeline** — complete (parsers for v2/v3/v4 maps, BeatSaver downloader, Unity extractor, Parquet storage)
- **ML model** — complete (tokenizer, audio preprocessing, Conformer/Transformer encoder, training loop, inference, exporter, evaluation)
- **Baseline training** — complete (small model: 16 epochs, 23K songs, 60.6% token accuracy, generates playable maps)
- **Model improvements** — complete (dataset filtering, SpecAugment, onset features, RoPE, color balance loss, Conformer encoder)
- **Conformer training** — in progress (9.4M params, 53.6% accuracy at epoch 7, val loss still decreasing)

## Tests

```bash
# Run all tests (178 total; ML tests auto-skip without ML deps)
python -m pytest tests/ -v
```

## License

[MIT](LICENSE)
