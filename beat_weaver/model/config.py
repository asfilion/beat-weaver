"""Model configuration — all hyperparameters in one dataclass."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class ModelConfig:
    """All hyperparameters for the BeatWeaver model."""

    # Tokenizer
    vocab_size: int = 291
    max_seq_len: int = 2048

    # Audio
    sample_rate: int = 22050
    n_mels: int = 80
    n_fft: int = 2048
    hop_length: int = 512
    max_audio_len: int = 8192  # Max frames for audio encoder (truncated in dataset)

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
    batch_size: int = 2
    learning_rate: float = 1e-4
    warmup_steps: int = 4000
    max_epochs: int = 100
    dropout: float = 0.1
    label_smoothing: float = 0.1
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 4  # Effective batch = batch_size * this
    early_stopping_patience: int = 10

    # Data weighting
    official_ratio: float = 0.2  # Target fraction of each batch from official maps

    # Data filtering
    min_difficulty: str = "Easy"  # Minimum difficulty to include
    characteristics: list[str] | None = None  # None = all; ["Standard"] = Standard only
    min_bpm: float = 0.0
    max_bpm: float = 9999.0

    # Audio features
    use_onset_features: bool = False  # Concatenate onset strength as extra mel channel

    # Positional encoding
    use_rope: bool = True  # Use RoPE instead of sinusoidal PE

    # Conformer encoder
    use_conformer: bool = True  # Conformer blocks instead of Transformer encoder
    conformer_kernel_size: int = 31  # Depthwise conv kernel size

    # Auxiliary losses
    density_loss_weight: float = 0.1
    color_balance_weight: float = 0.0  # Weight for color balance auxiliary loss

    def save(self, path: Path) -> None:
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ModelConfig:
        """Load config from JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # Only pass known fields to handle forward/backward compat
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
