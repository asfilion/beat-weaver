# Plan 004: Model Improvements — Scaling, Data Quality, Architecture

## Context

Baseline training plateaued at val_loss=2.055 / 60.6% token accuracy with a 1M param model on 42K samples. Research identified several high-impact improvements: the model is undersized, 93% of Expert+ maps are truncated at max_seq_len=1024, quality weighting is broken, and the architecture lacks modern improvements like RoPE and onset features.

Hardware: RTX 4070 (8GB VRAM), 32GB RAM. All changes must fit within these constraints.

## Changes (ordered by implementation dependency)

### 1. Add dataset filtering options to ModelConfig

**File:** `beat_weaver/model/config.py`

**Problem:** No way to filter training data by difficulty, characteristic, or BPM at the config level.

**Fix:** Add filtering fields to `ModelConfig`:
```python
# Data filtering
min_difficulty: str = "Easy"           # Minimum difficulty to include
characteristics: list[str] | None = None  # None = all; ["Standard"] = Standard only
min_bpm: float = 0.0
max_bpm: float = 9999.0
```

### 2. Apply dataset filters in BeatSaberDataset.__init__

**File:** `beat_weaver/model/dataset.py`

**Problem:** All difficulties and characteristics are included, adding noise from Easy/Normal maps and non-Standard modes.

**Fix:** In the `for (song_hash, difficulty, characteristic), group in grouped:` loop, add filtering:
- Skip if `characteristic` not in `config.characteristics` (when set)
- Skip if difficulty rank below `config.min_difficulty` (use a lookup: Easy=1, Normal=3, Hard=5, Expert=7, ExpertPlus=9)
- Skip if BPM outside `[config.min_bpm, config.max_bpm]`
- Log how many samples were filtered and why

### 3. Add SpecAugment to dataset

**File:** `beat_weaver/model/dataset.py`

**Problem:** No data augmentation. Model may overfit to specific spectral patterns.

**Fix:** Add SpecAugment in `__getitem__` (training only, not validation):
```python
def _spec_augment(self, mel: np.ndarray) -> np.ndarray:
    """Apply SpecAugment: random time and frequency masking."""
    n_mels, n_frames = mel.shape
    # Frequency masking: 1-2 bands of width up to 10
    for _ in range(2):
        f = np.random.randint(0, min(10, n_mels))
        f0 = np.random.randint(0, n_mels - f)
        mel[f0:f0 + f, :] = 0.0
    # Time masking: 1-2 bands of width up to 20
    for _ in range(2):
        t = np.random.randint(0, min(20, n_frames))
        t0 = np.random.randint(0, n_frames - t)
        mel[:, t0:t0 + t] = 0.0
    return mel
```
- Add `split` parameter to `__getitem__` context (store `self.split` in `__init__`)
- Only apply during `split == "train"`
- Operates on the numpy array before converting to tensor

### 4. Add onset strength feature channel

**File:** `beat_weaver/model/audio.py`, `beat_weaver/model/dataset.py`

**Problem:** Mel spectrograms encode spectral energy but not onset timing explicitly. Onset strength directly encodes "something rhythmically interesting happened here."

**Note:** `compute_onset_envelope()` already exists in `audio.py` (returns shape `(1, T)`).

**Fix in dataset.py:**
- In `_compute_one_mel()`: also compute onset envelope, beat-align it, and stack with mel → save as `(n_mels+1, T)` array
- In `__getitem__()`: mel loaded from cache already has onset channel
- Update `warm_mel_cache()` to use the new computation

**Fix in audio.py:**
- Add `compute_mel_with_onset()` that returns `(n_mels+1, T)` by stacking mel + onset

**Fix in config.py:**
- Add `use_onset_features: bool = True`

**Fix in transformer.py:**
- Change `AudioEncoder.input_proj` from `Linear(n_mels, encoder_dim)` to `Linear(n_mels + (1 if config.use_onset_features else 0), encoder_dim)`

**Cache compatibility:** Old mel cache files are `(80, T)`. New ones are `(81, T)`. Detect by checking `mel.shape[0]` — if 80, pad with zeros for the onset channel. Or: clear mel cache when onset features are enabled (add a cache version marker).

### 5. Add color balance auxiliary loss

**File:** `beat_weaver/model/training.py`

**Problem:** Model generates 70% red / 30% blue notes. The training data likely has this imbalance, and the loss function doesn't penalize it.

**Fix:** After computing main cross-entropy loss, add an auxiliary loss:
```python
def _color_balance_loss(logits: torch.Tensor, config: ModelConfig) -> torch.Tensor:
    """Penalize deviation from 50/50 LEFT/RIGHT token probability."""
    # Sum predicted probabilities over LEFT and RIGHT token ranges
    probs = torch.softmax(logits, dim=-1)  # (batch, seq, vocab)
    left_prob = probs[:, :, LEFT_BASE:LEFT_BASE+LEFT_COUNT].sum(dim=-1)   # (batch, seq)
    right_prob = probs[:, :, RIGHT_BASE:RIGHT_BASE+RIGHT_COUNT].sum(dim=-1)
    total = left_prob + right_prob + 1e-8
    left_ratio = left_prob / total
    # Penalize deviation from 0.5
    balance_loss = ((left_ratio - 0.5) ** 2).mean()
    return balance_loss
```
- Weight by `config.color_balance_weight` (new field, default 0.1)
- Only compute on positions where LEFT or RIGHT tokens are predicted (mask by target token type)
- Add to config: `color_balance_weight: float = 0.1`

### 6. Fix BeatSaver score persistence

**Files:** `beat_weaver/sources/beatsaver.py`, `beat_weaver/cli.py`

**Problem:** BeatSaver API returns `score` and `upvotes` for each map, but these are only used for filtering during download. They're never written to the map folders, so `parse_map_folder()` can't access them. The writer's `metadata.json` correctly writes `meta.score`, but it's always `None` because the parser doesn't have the info.

**Fix:** Save a sidecar `beatsaver_meta.json` during download:
- In `download_maps()`, after downloading each map, write `{hash}/beatsaver_meta.json` containing `{"score": ..., "upvotes": ..., "downvotes": ...}`
- In `process_map_folder()` or `cmd_process`, check for `beatsaver_meta.json` and inject score/upvotes into `SongMetadata`
- Alternative simpler approach: in `cmd_process`, scan for a bulk `beatsaver_scores.json` file that maps hash → score. Generate this file during the download phase.

**Chosen approach:** Write a `beatsaver_scores.json` in the download directory during `download_maps()`. During `cmd_process`, load this file and inject scores into each `NormalizedBeatmap.metadata.score` before writing.

### 7. Create medium model config

**File:** `configs/medium.json`

**Problem:** Small config (1M params, 2L/128d) has saturated. Need a config sized for 8GB VRAM.

**Config:**
```json
{
  "vocab_size": 291,
  "max_seq_len": 4096,
  "sample_rate": 22050,
  "n_mels": 80,
  "n_fft": 2048,
  "hop_length": 512,
  "max_audio_len": 4096,
  "encoder_layers": 4,
  "encoder_dim": 256,
  "encoder_heads": 8,
  "encoder_ff_dim": 1024,
  "decoder_layers": 3,
  "decoder_dim": 256,
  "decoder_heads": 8,
  "decoder_ff_dim": 1024,
  "batch_size": 4,
  "learning_rate": 3e-4,
  "warmup_steps": 1000,
  "max_epochs": 50,
  "dropout": 0.15,
  "label_smoothing": 0.1,
  "gradient_clip_norm": 1.0,
  "gradient_accumulation_steps": 4,
  "early_stopping_patience": 10,
  "official_ratio": 0.2,
  "density_loss_weight": 0.1,
  "color_balance_weight": 0.1,
  "use_onset_features": true,
  "min_difficulty": "Expert",
  "characteristics": ["Standard"],
  "min_bpm": 50.0,
  "max_bpm": 300.0
}
```

Effective batch size: 4 × 4 = 16. Estimated ~8M params. Should fit in 8GB VRAM.

### 8. Replace sinusoidal PE with RoPE

**File:** `beat_weaver/model/transformer.py`

**Problem:** Sinusoidal absolute positional encoding has poor length generalization and doesn't model relative distances well. RoPE is the modern standard.

**Fix:** Add `RotaryPositionalEncoding` class:
```python
class RotaryPositionalEncoding(nn.Module):
    """Rotary Position Embedding (RoPE) - Su et al. 2021."""

    def __init__(self, dim: int, max_len: int = 16384):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cos, sin) tensors of shape (seq_len, dim/2)."""
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        return freqs.cos(), freqs.sin()
```

**Integration approach:** RoPE is applied inside the attention mechanism, not as an additive embedding. PyTorch's `nn.TransformerEncoderLayer` doesn't natively support RoPE. Two options:

**Option A (simpler):** Use PyTorch's `nn.MultiheadAttention` with custom `forward` that applies rotary embeddings to Q and K before the attention computation. This requires subclassing `TransformerEncoderLayer` and `TransformerDecoderLayer`.

**Option B (cleanest):** Build custom encoder/decoder layers that use RoPE-aware attention. More code but cleaner architecture.

**Chosen: Option A** — Subclass the PyTorch layers, override the self-attention forward to apply RoPE to Q/K. Cross-attention in the decoder does NOT use RoPE (audio and token positions are in different spaces).

**Key implementation details:**
- Apply RoPE only to self-attention Q/K (not cross-attention)
- `apply_rotary_emb(x, cos, sin)` rotates pairs of dimensions
- Remove `SinusoidalPositionalEncoding` from encoder and decoder when using RoPE
- Keep sinusoidal PE as fallback (config flag: `use_rope: bool = True`)
- The embedding still needs positional information — RoPE replaces the additive PE, not the embedding itself

### 9. Invalidate mel cache when features change

**File:** `beat_weaver/model/dataset.py`

**Problem:** Adding onset features changes the cached mel shape from `(80, T)` to `(81, T)`. Old cache files are incompatible.

**Fix:** Add a cache version file `mel_cache/VERSION` that stores a hash of feature-relevant config (n_mels, n_fft, hop_length, sample_rate, use_onset_features). If VERSION doesn't match, log a warning and recompute. The `warm_mel_cache()` function handles the recomputation.

## New Tests

### `tests/test_model.py` additions

1. **test_rope_output_shape**: RoPE encoder produces correct output shape
2. **test_rope_relative_distance**: Verify RoPE attention weights decay with distance
3. **test_rope_vs_sinusoidal_backward**: Both PE variants produce gradients
4. **test_onset_feature_input_dim**: AudioEncoder accepts n_mels+1 input when onset features enabled
5. **test_medium_config_forward**: Forward pass works with medium config dimensions

### `tests/test_audio.py` additions

6. **test_compute_mel_with_onset_shape**: Output has n_mels+1 channels
7. **test_onset_channel_nonnegative**: Onset strength is non-negative
8. **test_onset_beat_aligned_shape**: Beat-aligned onset+mel has correct shape

### `tests/test_dataset_filtering.py` (new file)

9. **test_filter_by_difficulty**: Only Expert/ExpertPlus samples when min_difficulty="Expert"
10. **test_filter_by_characteristic**: Only Standard samples when characteristics=["Standard"]
11. **test_filter_by_bpm**: Samples outside [50, 300] excluded
12. **test_spec_augment_training_only**: SpecAugment applied in train split, not val
13. **test_spec_augment_shape_preserved**: Output shape unchanged after augmentation

### `tests/test_training.py` (new file)

14. **test_color_balance_loss_balanced**: Loss near zero for 50/50 predictions
15. **test_color_balance_loss_imbalanced**: Loss > 0 for 100% left predictions
16. **test_color_balance_loss_gradient**: Gradients flow through auxiliary loss

### `tests/test_beatsaver_scores.py` (new file)

17. **test_scores_json_written**: Download phase writes beatsaver_scores.json
18. **test_scores_injected_into_metadata**: Scores appear in metadata.json after processing

## Files Modified

| File | Changes |
|------|---------|
| `beat_weaver/model/config.py` | Add filtering fields, onset flag, color balance weight, RoPE flag |
| `beat_weaver/model/dataset.py` | Dataset filtering (#2), SpecAugment (#3), onset features (#4), cache versioning (#9) |
| `beat_weaver/model/audio.py` | `compute_mel_with_onset()` (#4) |
| `beat_weaver/model/transformer.py` | RoPE class + RoPE-aware encoder/decoder layers (#8) |
| `beat_weaver/model/training.py` | Color balance auxiliary loss (#5) |
| `beat_weaver/sources/beatsaver.py` | Write beatsaver_scores.json during download (#6) |
| `beat_weaver/cli.py` | Load scores and inject during process (#6) |
| `beat_weaver/pipeline/processor.py` | Accept optional scores dict (#6) |
| `configs/medium.json` | New medium config (#7) |
| `tests/test_model.py` | RoPE tests, onset input dim test, medium config test (#1-5 above) |
| `tests/test_audio.py` | Onset+mel tests (#6-8 above) |
| `tests/test_dataset_filtering.py` | New: filtering + SpecAugment tests (#9-13 above) |
| `tests/test_training.py` | New: color balance loss tests (#14-16 above) |
| `tests/test_beatsaver_scores.py` | New: score persistence tests (#17-18 above) |

## Implementation Order

Dependencies between changes dictate this order:

1. **Config fields** (#1) — all other changes depend on these
2. **Dataset filtering** (#2) — no code deps, just config
3. **SpecAugment** (#3) — no code deps
4. **Onset features** (#4) — requires config field + audio.py + transformer.py input_proj change
5. **Color balance loss** (#5) — requires config field + training.py
6. **BeatSaver scores** (#6) — independent pipeline fix
7. **Medium config** (#7) — depends on all config fields existing
8. **RoPE** (#8) — most complex, independent of others, do last
9. **Cache versioning** (#9) — do alongside onset features

## Verification

1. `python -m pytest tests/ -x -q` after each change — all tests pass
2. Quick sanity check: load medium config, construct model, verify param count (~8M)
3. Run a short training (~3 epochs) on full dataset with medium config to verify no crashes
4. Push to remote, verify CI passes (Python 3.11 + 3.12)
