"""Encoder-decoder transformer for audio-to-map generation.

Architecture:
    Audio Encoder: Linear(n_mels → d_model) + PE + TransformerEncoder
    Token Decoder: Embedding(vocab → d_model) + PE + TransformerDecoder + Linear(d_model → vocab)

Supports both sinusoidal positional encoding and RoPE (config.use_rope).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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


# ── RoPE ──────────────────────────────────────────────────────────────────────


class RotaryPositionalEncoding(nn.Module):
    """Rotary Position Embedding (RoPE) — Su et al. 2021.

    Precomputes cos/sin frequency tables. Applied to Q/K in attention.
    """

    def __init__(self, dim: int, max_len: int = 16384):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cos, sin) each of shape (1, 1, seq_len, dim/2)."""
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        return freqs.cos().unsqueeze(0).unsqueeze(0), freqs.sin().unsqueeze(0).unsqueeze(0)


def _apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embedding to x of shape (batch, heads, seq_len, head_dim).

    cos, sin: (1, 1, seq_len, head_dim/2)
    """
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class RoPEMultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE applied to Q and K."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Forward pass. RoPE applied when rope_cos/rope_sin provided."""
        B, S, _ = query.shape
        _, T, _ = key.shape

        q = self.q_proj(query).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        if rope_cos is not None and rope_sin is not None:
            q = _apply_rotary_emb(q, rope_cos[:, :, :S], rope_sin[:, :, :S])
            k = _apply_rotary_emb(k, rope_cos[:, :, :T], rope_sin[:, :, :T])

        # Build attention mask for SDPA
        if key_padding_mask is not None and attn_mask is None:
            # key_padding_mask: (B, T), True = ignore
            # SDPA needs float mask: -inf for masked positions
            kpm = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            attn_mask = torch.zeros(B, 1, S, T, device=query.device, dtype=query.dtype)
            attn_mask.masked_fill_(kpm, float("-inf"))
        elif key_padding_mask is not None and attn_mask is not None:
            # Combine causal mask with padding mask
            kpm = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            # attn_mask is (S, T) causal — broadcast to (B, 1, S, T)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, S, T).clone()
            attn_mask.masked_fill_(kpm, float("-inf"))
            is_causal = False  # mask already encodes causality

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out_proj(out)


# ── Conformer ─────────────────────────────────────────────────────────────────


class ConformerFeedForward(nn.Module):
    """Half-step FFN for Conformer Macaron structure.

    LayerNorm -> Linear -> SiLU -> Dropout -> Linear -> Dropout.
    The 0.5 scaling is applied externally in ConformerBlock.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class ConformerConvModule(nn.Module):
    """Conformer convolution module.

    LayerNorm -> Pointwise(d, 2d) -> GLU -> DepthwiseConv1d -> BatchNorm -> SiLU
    -> Pointwise(d, d) -> Dropout.
    """

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: (batch, seq_len, d_model), padding_mask: (batch, seq_len) True=ignore."""
        x = self.norm(x)
        # (B, T, D) -> (B, D, T) for Conv1d
        x = x.transpose(1, 2)

        # Zero out padded positions before convolution
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(1), 0.0)

        x = self.pointwise1(x)  # (B, 2D, T)
        x = F.glu(x, dim=1)  # (B, D, T)
        x = self.depthwise(x)  # (B, D, T)

        # Zero out padded positions after depthwise conv
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(1), 0.0)

        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise2(x)  # (B, D, T)
        x = self.dropout(x)

        # Final mask to ensure padded positions are zero
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(1), 0.0)

        # Back to (B, T, D)
        return x.transpose(1, 2)


class ConformerBlock(nn.Module):
    """Conformer block: FFN/2 + Self-Attention + ConvModule + FFN/2 + LayerNorm.

    Forward signature matches RoPEEncoderLayer for drop-in use in AudioEncoder.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self.ffn1 = ConformerFeedForward(d_model, dim_feedforward, dropout)
        self.self_attn = RoPEMultiHeadAttention(d_model, nhead, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv_module = ConformerConvModule(d_model, kernel_size, dropout)
        self.ffn2 = ConformerFeedForward(d_model, dim_feedforward, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        src: torch.Tensor,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # First half-step FFN
        x = src + 0.5 * self.ffn1(src)

        # Self-attention with RoPE
        attn_in = self.attn_norm(x)
        attn_out = self.self_attn(
            attn_in, attn_in, attn_in,
            rope_cos=rope_cos, rope_sin=rope_sin,
            key_padding_mask=src_key_padding_mask,
        )
        x = x + self.attn_dropout(attn_out)

        # Convolution module
        x = x + self.conv_module(x, padding_mask=src_key_padding_mask)

        # Second half-step FFN
        x = x + 0.5 * self.ffn2(x)

        # Final layer norm
        return self.final_norm(x)


class RoPEEncoderLayer(nn.Module):
    """Transformer encoder layer with RoPE self-attention."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = RoPEMultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        src: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm self-attention with RoPE
        x = self.norm1(src)
        x = self.self_attn(
            x, x, x, rope_cos=rope_cos, rope_sin=rope_sin,
            key_padding_mask=src_key_padding_mask,
        )
        src = src + self.dropout1(x)
        # Pre-norm FFN
        x = self.norm2(src)
        x = self.linear2(self.dropout2(self.activation(self.linear1(x))))
        src = src + self.dropout3(x)
        return src


class RoPEDecoderLayer(nn.Module):
    """Transformer decoder layer with RoPE on self-attention only.

    Cross-attention uses standard attention (no RoPE) since audio and token
    positions are in different spaces.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = RoPEMultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = RoPEMultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        tgt_is_causal: bool = False,
    ) -> torch.Tensor:
        # Pre-norm self-attention with RoPE + causal mask
        x = self.norm1(tgt)
        x = self.self_attn(
            x, x, x, rope_cos=rope_cos, rope_sin=rope_sin,
            attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask,
            is_causal=tgt_is_causal and tgt_mask is None,
        )
        tgt = tgt + self.dropout1(x)
        # Pre-norm cross-attention (no RoPE)
        x = self.norm2(tgt)
        x = self.cross_attn(
            x, memory, memory,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(x)
        # Pre-norm FFN
        x = self.norm3(tgt)
        x = self.linear2(self.dropout3(self.activation(self.linear1(x))))
        tgt = tgt + self.dropout4(x)
        return tgt


# ── Main modules ──────────────────────────────────────────────────────────────


class AudioEncoder(nn.Module):
    """Encode mel spectrogram into contextualized audio representations."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.use_rope = config.use_rope
        self.use_conformer = config.use_conformer
        input_dim = config.n_mels + (1 if config.use_onset_features else 0)
        self.input_proj = nn.Linear(input_dim, config.encoder_dim)

        if config.use_conformer:
            # Conformer encoder (supports both RoPE and sinusoidal PE)
            if config.use_rope:
                self.rope = RotaryPositionalEncoding(
                    config.encoder_dim // config.encoder_heads,
                    max_len=config.max_audio_len,
                )
            else:
                self.pos_enc = SinusoidalPositionalEncoding(
                    config.encoder_dim, max_len=config.max_audio_len,
                    dropout=config.dropout,
                )
            self.dropout = nn.Dropout(config.dropout)
            self.layers = nn.ModuleList([
                ConformerBlock(
                    config.encoder_dim, config.encoder_heads,
                    config.encoder_ff_dim, config.conformer_kernel_size,
                    config.dropout,
                )
                for _ in range(config.encoder_layers)
            ])
        elif config.use_rope:
            self.rope = RotaryPositionalEncoding(
                config.encoder_dim // config.encoder_heads,
                max_len=config.max_audio_len,
            )
            self.dropout = nn.Dropout(config.dropout)
            self.layers = nn.ModuleList([
                RoPEEncoderLayer(
                    config.encoder_dim, config.encoder_heads,
                    config.encoder_ff_dim, config.dropout,
                )
                for _ in range(config.encoder_layers)
            ])
        else:
            self.pos_enc = SinusoidalPositionalEncoding(
                config.encoder_dim, max_len=config.max_audio_len,
                dropout=config.dropout,
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

        padding_mask = ~mel_mask if mel_mask is not None else None

        if self.use_conformer:
            if self.use_rope:
                x = self.dropout(x)
                cos, sin = self.rope(x.size(1), x.device)
            else:
                x = self.pos_enc(x)
                cos, sin = None, None
            for layer in self.layers:
                x = layer(x, cos, sin, src_key_padding_mask=padding_mask)
        elif self.use_rope:
            x = self.dropout(x)
            cos, sin = self.rope(x.size(1), x.device)
            for layer in self.layers:
                x = layer(x, cos, sin, src_key_padding_mask=padding_mask)
        else:
            x = self.pos_enc(x)
            x = self.encoder(x, src_key_padding_mask=padding_mask)

        return x


class TokenDecoder(nn.Module):
    """Decode token sequence with cross-attention to audio encoder output."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.use_rope = config.use_rope
        self.embedding = nn.Embedding(config.vocab_size, config.decoder_dim)
        self.output_proj = nn.Linear(config.decoder_dim, config.vocab_size)

        if config.use_rope:
            self.rope = RotaryPositionalEncoding(
                config.decoder_dim // config.decoder_heads,
                max_len=config.max_seq_len,
            )
            self.dropout = nn.Dropout(config.dropout)
            self.layers = nn.ModuleList([
                RoPEDecoderLayer(
                    config.decoder_dim, config.decoder_heads,
                    config.decoder_ff_dim, config.dropout,
                )
                for _ in range(config.decoder_layers)
            ])
        else:
            self.pos_enc = SinusoidalPositionalEncoding(
                config.decoder_dim, max_len=config.max_seq_len,
                dropout=config.dropout,
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
        seq_len = tokens.size(1)
        x = self.embedding(tokens)

        tgt_key_padding_mask = ~token_mask if token_mask is not None else None
        memory_key_padding_mask = ~memory_mask if memory_mask is not None else None

        if self.use_rope:
            x = self.dropout(x)
            cos, sin = self.rope(seq_len, tokens.device)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=tokens.device,
            )
            for layer in self.layers:
                x = layer(
                    x, memory, cos, sin,
                    tgt_mask=causal_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
        else:
            x = self.pos_enc(x)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=tokens.device,
            )
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
