"""Encoder-decoder transformer for audio-to-map generation.

Architecture:
    Audio Encoder: Linear(n_mels → d_model) + SinusoidalPE + TransformerEncoder
    Token Decoder: Embedding(vocab → d_model) + SinusoidalPE + TransformerDecoder + Linear(d_model → vocab)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from beat_weaver.model.config import ModelConfig


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 16384, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Shape: (1, max_len, d_model) for batch-first
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding. x shape: (batch, seq_len, d_model)."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class AudioEncoder(nn.Module):
    """Encode mel spectrogram into contextualized audio representations."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_proj = nn.Linear(config.n_mels, config.encoder_dim)
        self.pos_enc = SinusoidalPositionalEncoding(
            config.encoder_dim, dropout=config.dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.encoder_dim,
            nhead=config.encoder_heads,
            dim_feedforward=config.encoder_ff_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.encoder_layers,
        )

    def forward(
        self, mel: torch.Tensor, mel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode mel spectrogram.

        Args:
            mel: (batch, n_mels, T_audio)
            mel_mask: (batch, T_audio) — True for valid positions

        Returns:
            (batch, T_audio, encoder_dim)
        """
        # Transpose to (batch, T_audio, n_mels)
        x = mel.transpose(1, 2)
        x = self.input_proj(x)
        x = self.pos_enc(x)

        # TransformerEncoder expects src_key_padding_mask where True = ignore
        padding_mask = ~mel_mask if mel_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return x


class TokenDecoder(nn.Module):
    """Decode token sequence with cross-attention to audio encoder output."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.decoder_dim)
        self.pos_enc = SinusoidalPositionalEncoding(
            config.decoder_dim, dropout=config.dropout,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.decoder_dim,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_ff_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.decoder_layers,
        )
        self.output_proj = nn.Linear(config.decoder_dim, config.vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode tokens with cross-attention to encoder output.

        Args:
            tokens: (batch, T_tokens) — token IDs
            memory: (batch, T_audio, encoder_dim) — encoder output
            token_mask: (batch, T_tokens) — True for valid token positions
            memory_mask: (batch, T_audio) — True for valid audio positions

        Returns:
            (batch, T_tokens, vocab_size) — logits
        """
        # Causal mask for autoregressive decoding
        seq_len = tokens.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=tokens.device,
        )

        x = self.embedding(tokens)
        x = self.pos_enc(x)

        # Padding masks (True = ignore in PyTorch convention)
        tgt_key_padding_mask = ~token_mask if token_mask is not None else None
        memory_key_padding_mask = ~memory_mask if memory_mask is not None else None

        x = self.decoder(
            x, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=True,
        )
        return self.output_proj(x)


class BeatWeaverModel(nn.Module):
    """Full encoder-decoder model for Beat Saber map generation."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = AudioEncoder(config)
        self.decoder = TokenDecoder(config)

    def forward(
        self,
        mel: torch.Tensor,
        tokens: torch.Tensor,
        mel_mask: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forced forward pass.

        Args:
            mel: (batch, n_mels, T_audio)
            tokens: (batch, T_tokens) — input token IDs (shifted right)
            mel_mask: (batch, T_audio) — True for valid audio positions
            token_mask: (batch, T_tokens) — True for valid token positions

        Returns:
            (batch, T_tokens, vocab_size) — logits for next token prediction
        """
        memory = self.encoder(mel, mel_mask)
        logits = self.decoder(tokens, memory, token_mask, mel_mask)
        return logits

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
