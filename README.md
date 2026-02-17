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

## Requirements

- Python 3.11+
- For training: NVIDIA GPU with CUDA support
  - Medium conformer (9.4M params): 8GB+ VRAM
  - Large conformer (62M params): 24GB+ VRAM
- Beat Saber installation (Steam) — only needed for extracting official maps

## Installation

```bash
git clone https://github.com/asfilion/beat-weaver.git
cd beat-weaver

# Core (data pipeline only)
pip install -e .

# With ML model dependencies (required for training and generation)
pip install -e ".[ml]"

# Development (adds pytest)
pip install -e ".[ml,dev]"
```

## Quick Start: Training a Model

This is the end-to-end workflow from a fresh clone to a trained model.

### Step 1: Download training data

Download community maps from [BeatSaver](https://beatsaver.com/). This downloads maps with a rating score >= 0.75 and >= 5 upvotes. The download is resumable — you can stop and restart without losing progress.

```bash
# Download ~55K community maps (this takes several hours)
beat-weaver download --min-score 0.75 --output data/raw/beatsaver
```

### Step 2: Extract official maps (optional)

If you have Beat Saber installed via Steam, you can also extract the 214 official/DLC maps. These are higher quality and weighted at 20% of each training batch.

```bash
# Windows (default Steam path)
beat-weaver extract-official --output data/raw/official

# Custom install path
beat-weaver extract-official --beat-saber "/path/to/Beat Saber" --output data/raw/official
```

### Step 3: Process raw maps into Parquet

Parse all downloaded/extracted maps into a normalized Parquet format for training.

```bash
beat-weaver process --input data/raw --output data/processed
```

### Step 4: Build the audio manifest

Create a JSON mapping from song hash to audio file path. This tells the training pipeline where to find each song's audio.

```bash
beat-weaver build-manifest --input data/raw --output data/audio_manifest.json
```

### Step 5: Train the model

Choose a config based on your hardware:

| Config | Params | VRAM | File |
|--------|--------|------|------|
| Small | 1M | 4GB | `configs/small.json` |
| Medium | 6.5M | 6GB | `configs/medium.json` |
| Medium Conformer | 9.4M | 8GB | `configs/medium_conformer.json` |
| **Large Conformer** | **62M** | **24GB+** | **`configs/large_conformer.json`** |

```bash
# Train with the large conformer config (recommended if you have 24GB+ VRAM)
beat-weaver train \
  --config configs/large_conformer.json \
  --audio-manifest data/audio_manifest.json \
  --data data/processed \
  --output output/training

# Or with the medium conformer for 8GB GPUs
beat-weaver train \
  --config configs/medium_conformer.json \
  --audio-manifest data/audio_manifest.json \
  --data data/processed \
  --output output/training
```

On the first run, mel spectrograms are pre-computed and cached to `data/processed/mel_cache/` (~30GB for 23K songs, takes ~25 minutes). Subsequent runs reuse the cache.

Training logs to TensorBoard:

```bash
tensorboard --logdir output/training/tensorboard
```

### Step 6: Resume training (if interrupted)

Always resume from the `best/` checkpoint (never from numbered epoch checkpoints, which may be overwritten during training).

```bash
beat-weaver train \
  --config configs/large_conformer.json \
  --audio-manifest data/audio_manifest.json \
  --data data/processed \
  --output output/training \
  --resume output/training/checkpoints/best
```

### Step 7: Generate a map

```bash
# BPM is auto-detected from the audio
beat-weaver generate \
  --checkpoint output/training/checkpoints/best \
  --audio song.ogg \
  --difficulty Expert \
  --output my_map/

# With explicit BPM and seed for reproducibility
beat-weaver generate \
  --checkpoint output/training/checkpoints/best \
  --audio song.ogg \
  --difficulty ExpertPlus \
  --bpm 128 \
  --seed 42 \
  --output my_map/
```

The output folder can be copied directly to `Beat Saber_Data/CustomLevels/` to play in-game.

### Step 8: Evaluate (optional)

```bash
beat-weaver evaluate \
  --checkpoint output/training/checkpoints/best \
  --audio-manifest data/audio_manifest.json \
  --data data/processed
```

## All-in-One Pipeline

If you want to download, extract, process, and build the manifest in a single command:

```bash
beat-weaver run --beat-saber "/path/to/Beat Saber" --output data/processed
```

Note: this runs with conservative defaults (`--max-maps 100`). For full training data, use the individual steps above.

## Architecture

An encoder-decoder model that takes a log-mel spectrogram as input and generates a sequence of beat-quantized tokens representing note placements.

```
Audio (mel spectrogram + onset) -> [Conformer Encoder] -> [Token Decoder] -> Token Sequence -> v2 Beat Saber Map
```

- **Tokenizer:** 291-token vocabulary encoding difficulty, bar structure, beat positions, and compound note placements (position + direction per hand)
- **Encoder:** Linear projection + RoPE + Conformer blocks (FFN/2 + self-attention + depthwise conv + FFN/2 + LayerNorm). Falls back to standard Transformer with `use_conformer=false`.
- **Decoder:** Token embedding + RoPE + Transformer decoder with cross-attention to encoder
- **Audio features:** Log-mel spectrogram (80 bins) with onset strength channel
- **Training:** AdamW + cosine LR, mixed-precision (fp16), SpecAugment, color balance loss, dataset filtering by difficulty/characteristic/BPM, weighted sampling (official maps oversampled)
- **Inference:** Autoregressive generation with grammar constraints ensuring valid map structure. Windowed generation with overlap stitching for songs of any length.

See [RESEARCH.md](RESEARCH.md) for research details and [plans/](plans/) for implementation plans.

## Project Status

- **Data pipeline** — complete (parsers for v2/v3/v4 maps, BeatSaver downloader, Unity extractor, Parquet storage)
- **ML model** — complete (tokenizer, audio preprocessing, Conformer/Transformer encoder, training loop, inference, exporter, evaluation)
- **Baseline training** — complete (small model: 16 epochs, 23K songs, 60.6% token accuracy, generates playable maps)
- **Model improvements** — complete (dataset filtering, SpecAugment, onset features, RoPE, color balance loss, Conformer encoder)
- **Conformer training** — complete (9.4M params, best val_loss=2.23, 59.4% accuracy at epoch 26, Expert+ only)

## Tests

```bash
# Run all tests (178 total; ML tests auto-skip without ML deps)
python -m pytest tests/ -v
```

## License

[MIT](LICENSE)
