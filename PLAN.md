# Plan: Beat Saber Training Data Pipeline

## Context

We need to transform existing Beat Saber tracks into usable ML training data. There are three data sources with different access methods: local custom levels (plain JSON, trivial), BeatSaver community maps (bulk download via API, ~20k+ maps), and official/DLC maps (locked in Unity asset bundles, need extraction). This plan covers building a Python pipeline to access, parse, normalize, and store map data from all three sources.

## Pre-Step: Update LEARNINGS.md

Append new findings from this research round to `LEARNINGS.md`:
- Local installation structure (what's actually in the Beat Saber directory)
- Official maps stored in `StreamingAssets/aa/StandaloneWindows64/` as `.bundle` files and `BeatmapLevelsData/` binary files
- Built-in levels ("Beat Saber" and "Magic") already in CustomLevels/ as v2 JSON
- Important: Magic has v2 Info.dat but v3 beatmap files (cross-version case)
- BeatSaver API details (endpoints, filtering, rate limits)
- Existing datasets (DeepSaber: 765 songs, BeatMapSynthesizer: ~8k maps)
- UnityPy as the tool for Unity asset extraction
- Existing automapper projects and their data approaches

## Step 1: Project Scaffolding

Create the Python package structure and dependencies.

**Files to create:**
- `pyproject.toml` — dependencies: `requests`, `UnityPy`, `pyarrow`, `tqdm`
- `beat_weaver/__init__.py`
- `beat_weaver/schemas/__init__.py`
- `beat_weaver/parsers/__init__.py`
- `beat_weaver/sources/__init__.py`
- `beat_weaver/pipeline/__init__.py`
- `beat_weaver/storage/__init__.py`
- `tests/__init__.py`
- `.gitignore` — add `data/`, `*.egg-info`, `__pycache__`, `.venv`
- `data/` directory structure: `raw/beatsaver/`, `raw/official/`, `raw/custom_local/`, `processed/`, `cache/`

```
beat_weaver/
    __init__.py
    schemas/
        __init__.py
        normalized.py          # Dataclasses: Note, Bomb, Obstacle, SongMetadata, NormalizedBeatmap
        detection.py           # Version detection from JSON content
        v2.py                  # v2 note/obstacle parsing
        v3.py                  # v3 note/obstacle parsing
        v4.py                  # v4 note/obstacle parsing (index-based)
    parsers/
        __init__.py
        dat_reader.py          # Read .dat files with auto gzip detection
        info_parser.py         # Info.dat parsing (v2 and v4 styles)
        beatmap_parser.py      # Orchestrator: folder → NormalizedBeatmap list
    sources/
        __init__.py
        local_custom.py        # Iterate CustomLevels/ folders
        beatsaver.py           # BeatSaver API client + zip downloader
        unity_extractor.py     # UnityPy-based official map extraction
    pipeline/
        __init__.py
        processor.py           # Single map → normalized data
        batch.py               # Orchestrate all sources
        cache.py               # Track downloaded/processed maps
    storage/
        __init__.py
        writer.py              # Write Parquet + metadata JSON
    cli.py                     # argparse CLI entry points
```

## Step 2: Normalized Data Format (`schemas/normalized.py`)

Define dataclasses that all parsers output:

- `Note`: beat, time_seconds, x (0-3), y (0-2), color (0-1), cut_direction (0-8), angle_offset
- `Bomb`: beat, time_seconds, x, y
- `Obstacle`: beat, time_seconds, duration_beats, x, y, width, height
- `SongMetadata`: source, source_id, hash, song_name, song_author, mapper_name, bpm, duration, quality scores
- `DifficultyInfo`: characteristic, difficulty, rank, note_jump_speed, note_jump_offset, note/bomb/obstacle counts
- `NormalizedBeatmap`: metadata + difficulty_info + notes + bombs + obstacles

## Step 3: Schema Detection + Version Parsers (`schemas/`)

**`detection.py`** — Detect version from JSON content:
- `_version` field → v2
- `colorNotesData` key → v4
- `colorNotes` key → v3
- `_notes` key → v2 (fallback)

**`v2.py`** — Parse `_notes` array (type 0/1 = notes, type 3 = bombs), `_obstacles` (type 0 = full wall → height 5, type 1 = crouch → height 3)

**`v3.py`** — Parse `colorNotes`, `bombNotes`, `obstacles` with abbreviated field names

**`v4.py`** — Dereference index arrays: `colorNotes[i].i` → `colorNotesData[idx]`, same for bombs/obstacles

## Step 4: Parsers (`parsers/`)

**`dat_reader.py`** — Read `.dat` file, auto-detect gzip (magic bytes `1f 8b`), return parsed JSON dict

**`info_parser.py`** — Parse Info.dat (both `_difficultyBeatmapSets` v2-style and `difficultyBeatmaps` v4-style), extract BPM and list of (DifficultyInfo, filename) pairs

**`beatmap_parser.py`** — Top-level `parse_map_folder(path)`:
1. Find Info.dat (case-insensitive)
2. Parse Info.dat for metadata + difficulty file list
3. For each difficulty file: read, detect version, dispatch to v2/v3/v4 parser
4. Return list of NormalizedBeatmap
5. Log and skip on errors (don't fail the whole batch for one bad map)

**Key edge case:** Info.dat version can differ from beatmap file version (e.g., Magic: v2 Info.dat + v3 beatmaps). Detect independently per file.

## Step 5: Local Custom Levels Reader (`sources/local_custom.py`)

Iterate `Beat Saber_Data/CustomLevels/` subfolders, call `parse_map_folder()` on each. Currently yields 2 built-in songs. Also supports any user-installed custom maps.

Path: `C:\Program Files (x86)\Steam\steamapps\common\Beat Saber\Beat Saber_Data\CustomLevels\`

## Step 6: BeatSaver Downloader (`sources/beatsaver.py`)

**API client:**
- Base URL: `https://api.beatsaver.com`
- Search: `GET /search/text/{page}` with `sortOrder=Rating`, min score filter
- Rate limit: 1 req/sec
- Filter: score >= 0.7, automapper=false

**Downloader:**
- Download zip from `versions[0].downloadURL`
- Extract to `data/raw/beatsaver/{hash}/`
- Save BeatSaver metadata as `_beatsaver_meta.json` alongside (quality scores for later filtering)
- Skip if already downloaded (check cache)

**Cache:** `data/cache/beatsaver_index.json` tracks `{id: {hash, downloaded_at, score}}`

## Step 7: Official Map Extractor (`sources/unity_extractor.py`)

Uses `UnityPy` to extract from bundle files in `StreamingAssets/aa/StandaloneWindows64/`.

**Approach — discovery first:**
1. Enumerate all objects in `*_pack_assets_all_*.bundle` files (38 packs)
2. Log object types and names to understand internal structure
3. Extract `TextAsset` objects that look like map data (Info.dat, *.dat patterns)
4. Handle gzip decompression for v4 content
5. Also try `MonoBehaviour` objects if TextAssets don't contain maps
6. Write extracted files to `data/raw/official/{pack_name}/{track_name}/`

**Risk:** Internal bundle structure is unknown until we actually run UnityPy. First pass should be exploratory. May also need to look at the `BeatmapLevelsData/` directory files (60 unnamed binary files).

## Step 8: Storage (`storage/writer.py`)

**Output format: Parquet** (columnar, compressed, ML-friendly)

Three tables:
- `notes.parquet` — song_hash, source, difficulty, characteristic, bpm, beat, time_seconds, x, y, color, cut_direction, angle_offset
- `bombs.parquet` — same minus color/cut_direction/angle_offset
- `obstacles.parquet` — song_hash, source, difficulty, characteristic, bpm, beat, time_seconds, duration_beats, x, y, width, height
- `metadata.json` — song-level metadata (name, author, bpm, source, quality scores, difficulty list)

Use `pyarrow` for Parquet writes. Stream incrementally with `ParquetWriter` to avoid holding all data in memory.

## Step 9: Pipeline Orchestration + CLI (`pipeline/`, `cli.py`)

**Batch processor** with config for which sources to include, min quality score, max maps.

**CLI commands:**
- `beat-weaver download` — download from BeatSaver
- `beat-weaver extract-official` — extract from Unity bundles
- `beat-weaver process` — normalize raw maps to Parquet
- `beat-weaver run` — full pipeline (download + extract + process)

**Processing cache:** `data/cache/processed_hashes.json` — SHA-256 of .dat file contents, skip already-processed maps.

## Step 10: Tests

- `tests/fixtures/` — small hand-crafted map folders (v2, v3, v4, mixed-version)
- `test_schemas.py` — version detection, each parser with minimal JSON
- `test_parsers.py` — Info.dat parsing, gzip handling, cross-version (v2 Info + v3 beatmap)
- `test_beatsaver.py` — mock API responses, pagination, download flow
- `test_pipeline.py` — end-to-end: fixture folder → Parquet → verify schema and values

## Implementation Order

1. Scaffolding (Step 1)
2. Normalized dataclasses (Step 2)
3. Schema detection + v2/v3/v4 parsers (Step 3)
4. dat reader + Info.dat parser + beatmap orchestrator (Step 4)
5. Local custom reader (Step 5) — **first end-to-end test against real data**
6. Storage writer (Step 8) — can now parse local maps and write Parquet
7. Tests with fixtures (Step 10)
8. BeatSaver client + downloader (Step 6)
9. Unity extractor — discovery pass first (Step 7)
10. Pipeline orchestration + CLI (Step 9)

## Verification

After each major step:
- **Step 5 done:** Parse the 2 built-in levels, print note counts and sample notes. Verify the "Magic" cross-version case works.
- **Step 8 done:** Write Parquet from local levels, read it back, verify column types and values.
- **Step 6 done:** Download 5 maps from BeatSaver, parse them, verify diverse v2 maps work.
- **Step 7 done:** Run discovery on one `.bundle` file, inspect output to understand internal structure before writing full extraction.
- **Step 9 done:** Run full pipeline on local + small BeatSaver sample, verify complete Parquet output.
