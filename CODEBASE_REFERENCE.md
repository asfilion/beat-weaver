# Codebase Reference

Quick reference for the beat_weaver package structure, generated for AI context.

## Directory Tree

```
beat_weaver/
├── __init__.py                          # "Beat Weaver - AI-powered Beat Saber track generator."
├── cli.py                               # CLI: download, extract-official, build-manifest, process, run, train, generate, evaluate
├── model/                               # ML model package (requires [ml] optional deps)
│   ├── __init__.py
│   ├── audio.py                         # Mel spectrogram extraction + beat-aligned framing + BPM detection + audio manifest
│   ├── config.py                        # ModelConfig dataclass with all hyperparameters
│   ├── dataset.py                       # PyTorch Dataset: Parquet + audio → (mel, tokens, mask)
│   ├── evaluate.py                      # Quality metrics: onset F1, NPS accuracy, parity, diversity
│   ├── exporter.py                      # Token sequence → playable v2 Beat Saber map folder
│   ├── inference.py                     # Autoregressive generation with grammar-constrained decoding
│   ├── tokenizer.py                     # Token vocabulary (291), encode/decode beatmap ↔ tokens
│   ├── training.py                      # Training loop: mixed-precision, checkpointing, early stopping
│   └── transformer.py                   # AudioEncoder + TokenDecoder + BeatWeaverModel (~40M params)
├── parsers/
│   ├── __init__.py
│   ├── beatmap_parser.py               # Top-level: parse_map_folder(path) → list[NormalizedBeatmap]
│   ├── dat_reader.py                   # Read .dat files (JSON or gzip)
│   └── info_parser.py                  # Parse Info.dat metadata
├── schemas/
│   ├── __init__.py
│   ├── detection.py                    # Version detection logic
│   ├── normalized.py                   # Unified data schema (dataclasses) ← KEY FILE
│   ├── v2.py                           # v2-specific parsers
│   ├── v3.py                           # v3-specific parsers
│   └── v4.py                           # v4-specific parsers
├── pipeline/
│   ├── __init__.py
│   ├── batch.py                        # Full pipeline orchestration (PipelineConfig, run_pipeline)
│   ├── cache.py                        # Processing cache
│   └── processor.py                    # Individual map processing
├── sources/
│   ├── __init__.py
│   ├── beatsaver.py                    # BeatSaver API client + parallel downloader (score≥0.9, upvotes≥5)
│   ├── local_custom.py                 # Local CustomLevels iteration
│   └── unity_extractor.py             # Official maps + audio from Unity bundles (214 levels: 65 base + 149 DLC)
└── storage/
    ├── __init__.py
    └── writer.py                       # Parquet output (notes/bombs/obstacles) + JSON metadata

tests/
├── __init__.py
├── test_audio.py                       # Audio preprocessing tests (12 tests, skipped without [ml])
├── test_evaluate.py                    # Evaluation metrics tests (19 tests)
├── test_exporter.py                    # Map export tests (4 tests, skipped without [ml])
├── test_inference.py                   # Grammar mask + generation tests (12 tests, skipped without [ml])
├── test_model.py                       # Transformer forward/backward tests (9 tests, skipped without [ml])
├── test_parsers.py                     # Info/beatmap parser tests (11 tests)
├── test_schemas.py                     # Schema & version parsing tests (16 tests)
├── test_tokenizer.py                   # Tokenizer encode/decode tests (26 tests)
├── test_weighted_sampler.py            # Source weighting tests (9 tests, skipped without [ml])
└── test_writer.py                     # Parquet writer tests: row groups, file splitting, reader (13 tests)

configs/
└── small.json                         # Small model config (1M params, batch_size=32, 2 layers, dim=128)
```

## Core Dataclasses (`beat_weaver/schemas/normalized.py`)

```python
@dataclass
class Note:
    beat: float
    time_seconds: float       # beat * 60.0 / bpm
    x: int                    # column 0-3
    y: int                    # row 0-2
    color: int                # 0=Red/Left, 1=Blue/Right
    cut_direction: int        # 0=Up,1=Down,2=Left,3=Right,4=UL,5=UR,6=DL,7=DR,8=Any
    angle_offset: int = 0

@dataclass
class Bomb:
    beat: float
    time_seconds: float
    x: int
    y: int

@dataclass
class Obstacle:
    beat: float
    time_seconds: float
    duration_beats: float
    x: int
    y: int
    width: int
    height: int

@dataclass
class DifficultyInfo:
    characteristic: str       # "Standard", "OneSaber", "NoArrows", "360Degree", "90Degree"
    difficulty: str           # "Easy", "Normal", "Hard", "Expert", "ExpertPlus"
    difficulty_rank: int      # 1, 3, 5, 7, 9
    note_jump_speed: float
    note_jump_offset: float
    note_count: int = 0
    bomb_count: int = 0
    obstacle_count: int = 0
    nps: float | None = None

@dataclass
class SongMetadata:
    source: str               # "beatsaver", "local_custom", "official"
    source_id: str
    hash: str = ""
    song_name: str = ""
    song_sub_name: str = ""
    song_author: str = ""
    mapper_name: str = ""
    bpm: float = 0.0
    song_duration_seconds: float | None = None
    upvotes: int | None = None
    downvotes: int | None = None
    score: float | None = None

@dataclass
class NormalizedBeatmap:
    metadata: SongMetadata
    difficulty_info: DifficultyInfo
    notes: list[Note] = field(default_factory=list)
    bombs: list[Bomb] = field(default_factory=list)
    obstacles: list[Obstacle] = field(default_factory=list)
```

## CLI Entry Points (`beat_weaver/cli.py`)

- `beat-weaver download` — Download custom maps from BeatSaver API (parallel, resumable, score≥0.9, upvotes≥5)
- `beat-weaver extract-official` — Extract official maps + audio from Unity bundles (base + DLC)
- `beat-weaver build-manifest` — Build audio manifest (hash -> audio path) from raw map folders
- `beat-weaver process` — Normalize raw maps to Parquet (auto-detects source from path)
- `beat-weaver run` — Full pipeline (all sources)
- `beat-weaver train` — Train the ML model
- `beat-weaver generate` — Generate a Beat Saber map from audio (BPM auto-detected if not provided)
- `beat-weaver evaluate` — Evaluate model on test data

Entry point: `beat-weaver = "beat_weaver.cli:main"` (pyproject.toml)

## pyproject.toml

```toml
[project]
name = "beat-weaver"
version = "0.1.0"
description = "AI-powered Beat Saber track generator"
requires-python = ">=3.11"
dependencies = [
    "requests>=2.31",
    "UnityPy>=1.10",
    "pyarrow>=15.0",
    "tqdm>=4.66",
]

[project.optional-dependencies]
ml = [
    "torch>=2.2",
    "torchaudio>=2.2",
    "librosa>=0.10",
    "soundfile>=0.12",
    "tensorboard>=2.16",
]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
]

[project.scripts]
beat-weaver = "beat_weaver.cli:main"

[tool.setuptools.packages.find]
include = ["beat_weaver*"]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"
```

## Output Format (Parquet)

- `data/processed/notes_NNNN.parquet` — numbered files, one row group per song_hash, split at 1GB. Columns: song_hash, source, difficulty, characteristic, bpm, beat, time_seconds, x, y, color, cut_direction, angle_offset
- `data/processed/bombs_NNNN.parquet` — bombs with position data (same numbering scheme)
- `data/processed/obstacles_NNNN.parquet` — walls with duration/dimensions
- `data/processed/metadata.json` — Song-level metadata (list of dicts)
- `data/processed/mel_cache/` — Pre-computed mel spectrograms (`{song_hash}_{bpm}.npy`)
- `data/audio_manifest.json` — Maps song_hash → audio file path

Reader `read_notes_parquet(path)` handles: directory with `notes_*.parquet`, legacy `notes.parquet`, or direct file path.
