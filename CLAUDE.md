# Beat Saber Track Generator

An AI-powered mod for Beat Saber that generates custom note maps from audio files.

## Project Overview

Given an audio file as input, the system generates block positions and orientations (for both left and right sabers) at a regular sample rate (~20 Hz). The goal is a machine learning model that produces playable, enjoyable Beat Saber tracks automatically.

## Key Components

### 1. Beat Saber Map Format Research
- Reverse-engineer how the game stores official/purchased song maps
- Understand the data format for block positions, orientations, timing, and difficulty levels
- This data becomes the foundation for training data

### 2. Custom Track Loading
- Figure out how to create custom tracks that load into Beat Saber
- Support the standard custom map format so generated tracks are playable in-game

### 3. ML Model for Track Generation
- **Input:** Audio file (waveform, spectrogram features, etc.)
- **Output:** Sequence of block positions and orientations per timestep
- **Model type:** TBD
- **Design options:**
  - Random seed for repeatable tracks
  - Randomized generation for a unique experience every play

### 4. Training Data Pipeline
- **Primary sources:** Official store maps + community-created custom maps
- **Challenge:** Likely insufficient training data from existing maps alone
- **Solution:** Use the feedback capture system (below) to generate additional labeled data

### 5. Feedback Capture System (Later Phase)
- In-game mechanism to collect player feedback on generated tracks
- Enables continuous improvement of the model through real gameplay data
- Additional source of training data beyond what's publicly available

## Beat Saber Map Format Quick Reference

Maps are **JSON files** in folders. See [LEARNINGS.md](LEARNINGS.md) for full format details. Implementation plans are in [plans/](plans/).

- **Schemas:** v2 (most custom maps), v3, v4 (latest official). All JSON-based, `.dat` extension.
- **Entry point:** `Info.dat` — song metadata, references difficulty files
- **Grid:** 4 columns (x: 0-3) x 3 rows (y: 0-2)
- **Colors:** 0 = Red/Left, 1 = Blue/Right
- **Cut directions:** 0=Up, 1=Down, 2=Left, 3=Right, 4=UpLeft, 5=UpRight, 6=DownLeft, 7=DownRight, 8=Any
- **Timing:** All in beats (float). Convert: `seconds = beat * 60.0 / BPM`
- **Custom maps location:** `<Beat Saber>/Beat Saber_Data/CustomLevels/`
- **Official maps:** Extracted from Unity bundles via `UnityPy` (328 levels, 65 bundles)
- **Training data sources:** Custom maps from [BeatSaver](https://beatsaver.com/) (v2 JSON) + 65 official levels (v4 gzip JSON)
- **Local install:** `C:\Program Files (x86)\Steam\steamapps\common\Beat Saber`

## Data Pipeline (`beat_weaver` package)

**CLI:** `beat-weaver <command>` (install: `pip install -e .`)

| Command | Description |
|---------|-------------|
| `beat-weaver download` | Download custom maps from BeatSaver API |
| `beat-weaver extract-official` | Extract official maps from Unity bundles |
| `beat-weaver process` | Normalize raw maps to Parquet |
| `beat-weaver run` | Full pipeline (all sources) |

**Key modules:**
- `beat_weaver.parsers.beatmap_parser.parse_map_folder(path)` — parse any map folder
- `beat_weaver.sources.local_custom` — iterate CustomLevels/
- `beat_weaver.sources.beatsaver` — BeatSaver API client + downloader
- `beat_weaver.sources.unity_extractor` — official map extraction
- `beat_weaver.storage.writer` — Parquet output (notes/bombs/obstacles)

**Output format:** `data/processed/notes.parquet` with columns: song_hash, source, difficulty, characteristic, bpm, beat, time_seconds, x, y, color, cut_direction, angle_offset

**Tests:** `python -m pytest tests/ -v` (27 tests)

## ML Model Design Decisions

See [LEARNINGS.md](LEARNINGS.md) for full research details.

- **Architecture:** Encoder-decoder transformer (~20-60M params)
- **Audio input:** Log-mel spectrogram (80 bins, sr=22050, hop=512, ~43 fps), beat-aligned framing
- **Output:** Beat-quantized event tokens with compound notes (~305 token vocabulary)
- **Quantization:** 1/16th note beat subdivisions
- **Token format:** `[DIFF] [BAR] [POS] [LEFT_TOKEN] [RIGHT_TOKEN] ...`
- **Sequence length:** ~2,500-4,000 tokens for 3-min Expert map
- **Difficulty:** Prepended decoder token (`[EASY]` through `[EXPERT_PLUS]`)
- **Training loss:** Cross-entropy + focal loss for timing + density regression auxiliary
- **Map export:** v2 format, model output maps 1:1 to v2 JSON fields
- **Key evaluation:** Onset alignment F1, parity violation rate, NPS accuracy, beat alignment

## Open Questions

- **Windowed evaluation:** Score individual segments rather than whole tracks for finer training signal
- **Feedback capture system:** In-game mechanism to collect player feedback (later phase)
- **RL fine-tuning:** After supervised pretraining, fine-tune with player feedback reward model
