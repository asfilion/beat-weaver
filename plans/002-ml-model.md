# Plan: ML Model Implementation — Audio-to-Map Transformer

## Context

The data pipeline is complete (PR #1, merged). We have parsers for v2/v3/v4 Beat Saber maps, BeatSaver downloader, Unity extractor, and Parquet storage. Research is complete (PR #2, merged) with decisions documented in RESEARCH.md and CLAUDE.md.

**Goal:** Build an encoder-decoder transformer that takes audio + difficulty as input and generates a sequence of Beat Saber color note tokens. The model outputs beat-quantized compound tokens (~290 vocab) that map 1:1 to v2 Beat Saber JSON.

**Scope:** Color notes only — no bombs, walls, arcs, or chains.

## New Package Structure

```
beat_weaver/model/
    __init__.py
    config.py           # All hyperparameters in one dataclass
    tokenizer.py        # Token vocabulary, encode/decode NormalizedBeatmap ↔ token sequences
    audio.py            # Log-mel spectrogram extraction + beat-aligned framing
    dataset.py          # PyTorch Dataset: load Parquet + audio, produce (spectrogram, token_ids) pairs
    transformer.py      # AudioEncoder + TokenDecoder + BeatWeaver model
    training.py         # Training loop with mixed-precision, checkpointing, logging
    inference.py        # Autoregressive generation with grammar mask
    exporter.py         # Token sequence → v2 Beat Saber map folder
    evaluate.py         # Post-generation quality metrics
```

## Dependencies

Add to `pyproject.toml` under `[project.optional-dependencies]`:
```toml
ml = [
    "torch>=2.2",
    "torchaudio>=2.2",
    "librosa>=0.10",
    "soundfile>=0.12",
    "tensorboard>=2.16",
]
```

Keep core `beat_weaver` free of ML deps — `model/` is optional.

## Phase 1: Tokenizer (`model/tokenizer.py`)

No PyTorch dependency. Pure Python + existing dataclasses.

### Token Vocabulary (291 tokens)

| ID Range | Category | Count | Details |
|----------|----------|-------|---------|
| 0 | PAD | 1 | Padding |
| 1 | START | 1 | Sequence start |
| 2 | END | 1 | Sequence end |
| 3-7 | DIFF_* | 5 | Easy(3), Normal(4), Hard(5), Expert(6), ExpertPlus(7) |
| 8 | BAR | 1 | Bar boundary |
| 9-72 | POS_* | 64 | 1/16th note positions in a 4-beat bar (0-63) |
| 73 | LEFT_EMPTY | 1 | No left-hand note at this position |
| 74-181 | LEFT_x_y_d | 108 | 4 columns × 3 rows × 9 directions |
| 182 | RIGHT_EMPTY | 1 | No right-hand note at this position |
| 183-290 | RIGHT_x_y_d | 108 | 4 columns × 3 rows × 9 directions |

Compound note encoding: `base + x * 27 + y * 9 + direction`

### Key Functions

- `encode_beatmap(beatmap: NormalizedBeatmap) -> list[int]` — NormalizedBeatmap → token IDs
  1. Sort notes by beat
  2. Quantize beats to 1/16th note grid (round to nearest 1/16th within a bar)
  3. Group notes by quantized position
  4. Emit: `START DIFF_x BAR POS_p LEFT_tok RIGHT_tok ... BAR ... END`
  5. Handle multiple notes at same position (take one per hand, warn on duplicates)
- `decode_tokens(token_ids: list[int], bpm: float) -> list[Note]` — token IDs → Note list
  1. Parse BAR/POS tokens to recover beat timing
  2. Decode compound LEFT/RIGHT tokens to (x, y, color, direction)
  3. Convert beats to time_seconds using bpm
- `vocab_size` property → 291

### Existing Code to Reuse

- `beat_weaver.schemas.normalized.Note` — output dataclass from decode
- `beat_weaver.schemas.normalized.NormalizedBeatmap` — input to encode

## Phase 2: Audio Preprocessing (`model/audio.py`)

### Parameters (from research)

| Parameter | Value |
|-----------|-------|
| Sample rate | 22,050 Hz |
| n_mels | 80 |
| n_fft | 2,048 |
| hop_length | 512 |
| Window | Hann |
| Scale | Log-magnitude (dB, ref=max) |

### Key Functions

- `load_audio(path: Path) -> tuple[np.ndarray, int]` — load and resample to 22050 Hz
- `compute_mel_spectrogram(audio: np.ndarray, sr: int) -> np.ndarray` — returns `(80, T)` float32
- `beat_align_spectrogram(mel: np.ndarray, sr: int, hop_length: int, bpm: float) -> np.ndarray` — resample spectrogram frames to align with beat grid (1/16th note resolution). Each output frame = one beat subdivision.
- `compute_onset_envelope(audio: np.ndarray, sr: int) -> np.ndarray` — returns `(1, T)` onset strength

### Audio Manifest

A JSON file mapping song hashes to audio file paths:
```json
{
  "abc123": "/path/to/song.ogg",
  "def456": "/path/to/song.egg"
}
```

- `build_audio_manifest(raw_dirs: list[Path]) -> dict[str, str]` — scan raw map folders, find audio files, map by hash
- `save/load_manifest(path: Path)` — JSON persistence

## Phase 3: Configuration (`model/config.py`)

Single `ModelConfig` dataclass with all hyperparameters:

```python
@dataclass
class ModelConfig:
    # Tokenizer
    vocab_size: int = 291
    max_seq_len: int = 8192

    # Audio
    sample_rate: int = 22050
    n_mels: int = 80
    n_fft: int = 2048
    hop_length: int = 512

    # Encoder
    encoder_layers: int = 6
    encoder_dim: int = 512
    encoder_heads: int = 8
    encoder_ff_dim: int = 2048

    # Decoder
    decoder_layers: int = 6
    decoder_dim: int = 512
    decoder_heads: int = 8
    decoder_ff_dim: int = 2048

    # Training
    batch_size: int = 8
    learning_rate: float = 1e-4
    warmup_steps: int = 4000
    max_epochs: int = 100
    dropout: float = 0.1
    label_smoothing: float = 0.1

    # Auxiliary losses
    density_loss_weight: float = 0.1
```

Include `save/load` methods for JSON serialization (for checkpoints).

## Phase 4: Dataset (`model/dataset.py`)

PyTorch `Dataset` that produces `(mel_spectrogram, token_ids, token_mask)` tuples.

### Data Flow

1. Load `notes.parquet` + `metadata.json` from processed dir
2. Group notes by `(song_hash, difficulty, characteristic)`
3. For each group, reconstruct `NormalizedBeatmap` (notes only)
4. Tokenize with `encode_beatmap()` → token IDs
5. Load audio via manifest, compute mel spectrogram
6. Beat-align spectrogram to match token grid
7. Pad/truncate to `max_seq_len`

### Key Classes

- `BeatSaberDataset(Dataset)` — main dataset
  - `__init__(processed_dir, audio_manifest_path, config, split)`
  - `__getitem__(idx)` → `(mel: Tensor, tokens: LongTensor, mask: BoolTensor)`
  - Lazy audio loading (spectrograms computed on-the-fly or cached)
- `collate_fn(batch)` — pad sequences to batch max length

### Train/Val/Test Split

Split by song hash (not by difficulty) to prevent data leakage. 80/10/10 split.

## Phase 5: Transformer Model (`model/transformer.py`)

### Architecture

```
                         ┌──────────────────────────────────┐
                         │        Audio Encoder              │
  mel_spec (80, T_audio) │  Linear(80 → d_model)            │
  ─────────────────────► │  + Sinusoidal Pos Encoding        │ ──► encoder_out (T_audio, d_model)
                         │  TransformerEncoderLayer × N      │
                         └──────────────────────────────────┘
                                        │ cross-attention
                         ┌──────────────┼───────────────────┐
                         │        Token Decoder              │
  token_ids (T_tokens)   │  Embedding(vocab, d_model)        │
  ─────────────────────► │  + Sinusoidal Pos Encoding        │ ──► logits (T_tokens, vocab_size)
                         │  TransformerDecoderLayer × N      │
                         │  Linear(d_model → vocab_size)     │
                         └──────────────────────────────────┘
```

### Key Classes

- `SinusoidalPositionalEncoding(nn.Module)` — standard sinusoidal PE
- `AudioEncoder(nn.Module)` — linear projection + pos encoding + `nn.TransformerEncoder`
- `TokenDecoder(nn.Module)` — embedding + pos encoding + `nn.TransformerDecoder` + output head
- `BeatWeaverModel(nn.Module)` — combines encoder + decoder
  - `forward(mel, tokens) -> logits` — teacher-forced training
  - Uses PyTorch's built-in `nn.TransformerEncoder` / `nn.TransformerDecoder`

### Model Size Estimate

At d_model=512, 6+6 layers: ~40M parameters. Fits in 8GB VRAM with batch_size=8.

## Phase 6: Training (`model/training.py`)

### Training Loop

- `Trainer` class wrapping training state
- `train(config, train_dataset, val_dataset, output_dir)` — main entry point
- **Loss:** Cross-entropy with label smoothing (0.1) + optional NPS density auxiliary loss
- **Optimizer:** AdamW with cosine LR schedule + linear warmup
- **Mixed precision:** `torch.amp.autocast` for faster training
- **Gradient clipping:** max_norm=1.0
- **Checkpointing:** Save model + optimizer + config every N epochs to `output_dir/checkpoints/`
- **Logging:** TensorBoard for loss curves, learning rate, gradient norms
- **Early stopping:** Based on validation loss (patience = 10 epochs)
- **Metrics logged:** train_loss, val_loss, val_token_accuracy, val_onset_f1

### Checkpoint Format

```
checkpoints/
    config.json
    epoch_001/
        model.pt
        optimizer.pt
        training_state.json   # epoch, step, best_val_loss
    best/
        model.pt
        config.json
```

## Phase 7: Inference (`model/inference.py`)

### Autoregressive Generation

- `generate(model, mel_spectrogram, difficulty, config) -> list[int]`
- Start with `[START, DIFF_x]`, generate token-by-token
- **Temperature sampling** with configurable temperature (default 1.0)
- **Top-k / top-p** filtering for diversity control
- **Grammar mask:** At each step, mask invalid next tokens:
  - After START → only DIFF_* tokens
  - After DIFF_* → only BAR
  - After BAR → only POS_* or BAR or END
  - After POS_* → only LEFT_* tokens
  - After LEFT_* → only RIGHT_* tokens
  - After RIGHT_* → only POS_* or BAR or END
- Stop at END token or max_seq_len

### Seed Support

Accept optional `seed: int` for reproducible generation via `torch.manual_seed`.

## Phase 8: Map Exporter (`model/exporter.py`)

### Token Sequence → Playable Map Folder

- `export_map(token_ids, bpm, song_name, audio_path, output_dir, difficulty, config)`
  1. `decode_tokens()` → list of `Note` objects
  2. Build v2 `Info.dat` JSON (song name, BPM, difficulty, NJS from lookup table)
  3. Build v2 difficulty `.dat` file (`_notes` array with `_time`, `_lineIndex`, `_lineLayer`, `_type`, `_cutDirection`)
  4. Copy audio file as `song.ogg` (convert with ffmpeg if needed)
  5. Write all files to output folder

### NJS Lookup Table

| Difficulty | NJS |
|------------|-----|
| Easy | 10 |
| Normal | 10 |
| Hard | 12 |
| Expert | 16 |
| ExpertPlus | 18 |

### Existing Code to Reuse

- `beat_weaver.schemas.normalized.Note` — decoded note dataclass
- v2 JSON field mapping documented in RESEARCH.md (beat→_time, x→_lineIndex, y→_lineLayer, color→_type, cut_direction→_cutDirection)

## Phase 9: Evaluation (`model/evaluate.py`)

### Post-Generation Metrics

- `evaluate_map(generated_notes, reference_notes, bpm) -> dict[str, float]`
  - **Onset F1:** Match generated note times to reference within ±40ms tolerance
  - **NPS accuracy:** `1 - |nps_gen - nps_ref| / nps_ref`
  - **Beat alignment:** Mean distance from each note to nearest 1/16th grid position
  - **Parity violations:** Track swing state per hand, count forehand/backhand violations
  - **Pattern diversity:** Fraction of unique 4-note subsequences

- `evaluate_standalone(notes, bpm) -> dict[str, float]` — metrics that don't need a reference
  - Parity violations, beat alignment, pattern diversity, NPS

## Phase 10: CLI Extensions (`cli.py`)

Add three new subcommands:

### `beat-weaver train`
```
--data PATH          processed data directory (default: data/processed)
--audio-manifest PATH  audio manifest JSON
--output PATH        output directory for checkpoints
--config PATH        optional JSON config override
--epochs INT         max epochs (default: 100)
--batch-size INT     (default: 8)
--resume PATH        resume from checkpoint
```

### `beat-weaver generate`
```
--checkpoint PATH    model checkpoint directory (required)
--audio PATH         input audio file (required)
--difficulty STR     Easy/Normal/Hard/Expert/ExpertPlus (default: Expert)
--output PATH        output map folder
--bpm FLOAT          song BPM (required, or auto-detect later)
--temperature FLOAT  sampling temperature (default: 1.0)
--seed INT           random seed for reproducibility
```

### `beat-weaver evaluate`
```
--checkpoint PATH    model checkpoint
--data PATH          test data directory
--audio-manifest PATH
--output PATH        evaluation results JSON
```

## Implementation Order

1. **Phase 1: Tokenizer** — no ML deps, testable immediately with existing fixtures
2. **Phase 2: Audio** — librosa only, testable with any audio file
3. **Phase 3: Config** — pure dataclass, trivial
4. **Phase 4: Dataset** — first PyTorch dependency, test with small Parquet + audio
5. **Phase 5: Model** — architecture definition, test with random tensors
6. **Phase 6: Training** — training loop, test on tiny dataset (overfit check)
7. **Phase 7: Inference** — generation with grammar mask, test with trained checkpoint
8. **Phase 8: Exporter** — produce playable maps, test by loading in Beat Saber
9. **Phase 9: Evaluation** — metrics suite
10. **Phase 10: CLI** — wire everything together

## Tests

New test files in `tests/`:

- `test_tokenizer.py` — encode/decode round-trip, vocab size, edge cases (empty bars, simultaneous notes), quantization accuracy
- `test_audio.py` — spectrogram shape, beat alignment frame count, manifest building
- `test_model.py` — forward pass shapes, gradient flow, causal mask correctness
- `test_inference.py` — grammar mask validity, deterministic seed output
- `test_exporter.py` — output folder structure, v2 JSON schema compliance
- `test_evaluate.py` — metric calculations on known inputs

## Verification

After each phase:
- **Phase 1:** Round-trip encode→decode on test fixtures. Verify token count matches expected.
- **Phase 4:** Load real Parquet data, verify dataset length and tensor shapes.
- **Phase 5:** Forward pass on random data, verify output shape `(batch, seq_len, 291)`.
- **Phase 6:** Overfit on 1-3 training examples. Loss should approach 0.
- **Phase 7:** Generate tokens, verify grammar mask produces valid sequences.
- **Phase 8:** Load generated map in Beat Saber (via CustomWIPLevels).
