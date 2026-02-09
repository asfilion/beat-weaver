# Plan 003: Performance Refactor — Caching, Parallelism, Vectorization

## Context

Training is bottlenecked by `__getitem__` reloading WAV audio, recomputing mel spectrograms, and re-tokenizing on every access — repeating identical work across epochs. The data pipeline (`process`, `extract-official`) processes map folders sequentially. The Parquet loader iterates 900K+ rows in a Python loop. This refactor targets all these bottlenecks while keeping all 118 tests passing.

## Changes (ordered by impact)

### 1. Cache mel spectrograms to disk

**File:** `beat_weaver/model/dataset.py`

**Problem:** `__getitem__` loads WAV + computes mel + beat-aligns every call. For 1094 samples × 50 epochs = 54,700 redundant computations (~1-2s each).

**Fix:** Add mel spectrogram caching in `__getitem__`:
- Cache dir: `processed_dir / "mel_cache"`, created in `__init__`
- Cache key: `f"{song_hash}_{bpm}.npy"` (song_hash uniquely identifies audio; bpm determines beat alignment)
- On miss: compute mel → `np.save()`. On hit: `np.load()` (~1ms)
- 214 unique songs × ~3MB each ≈ 640MB disk — easily fits
- First epoch: same speed. All subsequent epochs: ~50x faster per sample

### 2. Pre-tokenize in `__init__`, not `__getitem__`

**File:** `beat_weaver/model/dataset.py`

**Problem:** `encode_beatmap()` is called in every `__getitem__`, reconstructing Note objects and running tokenization. Tokens are deterministic — same notes always produce same tokens.

**Fix:** After building `self.samples` in `__init__`:
- Iterate all samples, call `encode_beatmap()` once per sample
- Store pre-computed `token_ids` (list[int]) and `mask` (list[bool]) in each sample dict
- `__getitem__` just converts to tensors — no tokenization needed
- Drop the Note-reconstruction code from `__getitem__`

### 3. Vectorize Parquet loading with pandas groupby

**File:** `beat_weaver/model/dataset.py` (lines 81-112)

**Problem:** `to_pydict()` followed by `for i in range(n_rows)` Python loop over 900K rows building dicts one at a time.

**Fix:** Use pandas groupby:
```python
df = self.notes_table.to_pandas()
grouped = df.groupby(["song_hash", "difficulty", "characteristic"])
for (song_hash, difficulty, characteristic), group in grouped:
    note_dicts = group[["beat","time_seconds","x","y","color","cut_direction","angle_offset","bpm"]].to_dict("records")
```
- C-optimized groupby replaces 900K Python iterations
- `to_dict("records")` is vectorized

### 4. Vectorize beat alignment interpolation

**File:** `beat_weaver/model/audio.py` (lines 101-105)

**Problem:** Python `for i in range(n_mels)` loop calling `np.interp()` 80 times.

**Fix:** Use `scipy.interpolate.interp1d` for a single vectorized call:
```python
from scipy.interpolate import interp1d
f = interp1d(x_coords, mel, axis=1, kind="linear", fill_value="extrapolate")
aligned = f(frame_indices).astype(np.float32)
```
- Note: scipy is already a transitive dependency of librosa, no new dep needed

### 5. Parallelize `cmd_process` map folder processing

**File:** `beat_weaver/cli.py` (lines 66-83)

**Problem:** Sequential `for folder` loop processes each map folder one at a time. With thousands of BeatSaver maps, this becomes very slow.

**Fix:** Use `concurrent.futures.ProcessPoolExecutor`:
- Collect all folder paths first via `rglob("Info.dat")`
- Submit `process_map_folder()` calls in parallel
- Collect results with `as_completed()`
- ~4-8x speedup with default workers

### 6. Parallelize Unity bundle extraction

**File:** `beat_weaver/sources/unity_extractor.py` (lines 486-500)

**Problem:** Sequential `for bundle_path in level_bundles` loop. Each `_extract_level_bundle()` is independent.

**Fix:** Use `concurrent.futures.ProcessPoolExecutor`:
- Pre-build metadata lookup dict (currently done inline in loop)
- Submit all bundle extractions in parallel
- ~4-8x speedup for 214 bundles

### 7. Enable `num_workers > 0` in DataLoader (depends on #1)

**File:** `beat_weaver/model/training.py` (lines 229-245)

**Problem:** `num_workers=0` with comment "audio loading not picklable". Once mel spectrograms are cached (#1), `__getitem__` just loads .npy files — fully picklable.

**Fix:** Set `num_workers=2` for CUDA, 0 for CPU. Add `persistent_workers=True` when workers > 0.
- Conservative for Windows (multiprocessing has overhead)
- GPU stays fed while workers prefetch

### 8. Columnar Parquet row building

**File:** `beat_weaver/storage/writer.py` (lines 83-175)

**Problem:** Builds `list[dict]` of 900K+ dicts (each with 12 keys), then converts to Arrow table.

**Fix:** Build parallel lists (one per column) instead of list of dicts:
```python
hashes, beats, xs, ys, ... = [], [], [], [], ...
for bm in beatmaps:
    for note in bm.notes:
        hashes.append(...)
        beats.append(...)
notes_table = pa.table({"song_hash": hashes, "beat": beats, ...}, schema=NOTES_SCHEMA)
```
- Avoids 900K × 12 dict key-value pair allocations
- ~1.5-2x faster, less memory

## Files Modified

| File | Changes |
|------|---------|
| `beat_weaver/model/dataset.py` | Mel caching (#1), pre-tokenize (#2), pandas groupby (#3) |
| `beat_weaver/model/audio.py` | Vectorized interpolation (#4) |
| `beat_weaver/cli.py` | Parallel process command (#5) |
| `beat_weaver/sources/unity_extractor.py` | Parallel extraction (#6) |
| `beat_weaver/model/training.py` | num_workers > 0 (#7) |
| `beat_weaver/storage/writer.py` | Columnar building (#8) |

## Post-Plan Addition: Parquet Row Group Partitioning

Added after initial plan completion:
- **Writer:** One row group per `song_hash`, numbered files (`notes_0000.parquet`), split at 1 GB
- **Reader:** `read_notes_parquet()` handles multi-file and legacy single-file layouts
- **Dataset:** Uses `read_notes_parquet()` instead of direct `pq.read_table()`
- **Tests:** 13 new tests in `tests/test_writer.py`
- **Small config:** `configs/small.json` (1M params, 15s/epoch vs 456s/epoch)

## Status: COMPLETE

All 8 changes implemented and verified. Additional work done during baseline training run:

### Additional Fixes (discovered during full-dataset training)
- **v3 parser compact format:** `.get()` defaults for v3.3.0 maps that omit default-value keys
- **int8 overflow:** `_clamp8()` for mapping extension coordinates (x=1000, y=3000)
- **Difficulty aliases:** Case-insensitive matching + `Expert+` alias in tokenizer
- **Out-of-grid filtering:** Notes outside standard 4x3 grid filtered during pre-tokenization
- **Mel pre-caching:** `warm_mel_cache()` with ProcessPoolExecutor (~500 songs/min, 14x faster)
- **Memory optimization:** Free raw note dicts after tokenization, explicit DataFrame cleanup
- **Grammar constraint:** Strictly increasing POS in bars prevents duplicate same-beat notes
- **Audio truncation:** `cmd_generate` truncates mel to max_audio_len before inference

### Full-Dataset Training Results
- 23K songs, 42K training samples, 1M param model, batch_size=32
- Best val_loss=2.055, 60.6% accuracy after 13 epochs (~2 hours)
- Generates playable in-game maps

## Verification

1. `python -m pytest tests/ -x -q` — all 131 tests pass (118 original + 13 new)
2. Push to remote, verify CI passes (Python 3.11 + 3.12)
