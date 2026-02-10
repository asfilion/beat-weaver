"""Autoregressive generation with grammar-constrained decoding."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from beat_weaver.model.config import ModelConfig
from beat_weaver.model.tokenizer import (
    BAR,
    DIFF_EASY,
    DIFF_EXPERT_PLUS,
    END,
    LEFT_BASE,
    LEFT_COUNT,
    LEFT_EMPTY,
    POS_BASE,
    POS_COUNT,
    RIGHT_BASE,
    RIGHT_COUNT,
    RIGHT_EMPTY,
    START,
    VOCAB_SIZE,
    difficulty_to_token,
)
from beat_weaver.model.transformer import BeatWeaverModel


def _build_grammar_mask(last_token: int, last_pos_in_bar: int = -1) -> torch.Tensor:
    """Build a boolean mask over the vocabulary for valid next tokens.

    Returns a tensor of shape (VOCAB_SIZE,) where True = allowed.

    Args:
        last_token: The most recently generated token.
        last_pos_in_bar: The last POS offset used in the current bar (-1 if none).
            Used to enforce strictly increasing positions within a bar,
            preventing multiple notes at the same beat.

    Grammar rules:
        START      → DIFF_*
        DIFF_*     → BAR
        BAR        → POS_* | BAR | END
        POS_*      → LEFT_* | LEFT_EMPTY
        LEFT_*     → RIGHT_* | RIGHT_EMPTY
        RIGHT_*    → POS_* (strictly >) | BAR | END
    """
    mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool)

    if last_token == START:
        # After START → only difficulty tokens
        mask[DIFF_EASY: DIFF_EXPERT_PLUS + 1] = True

    elif DIFF_EASY <= last_token <= DIFF_EXPERT_PLUS:
        # After DIFF → only BAR
        mask[BAR] = True

    elif last_token == BAR:
        # After BAR → POS, BAR, or END
        mask[POS_BASE: POS_BASE + POS_COUNT] = True
        mask[BAR] = True
        mask[END] = True

    elif POS_BASE <= last_token < POS_BASE + POS_COUNT:
        # After POS → LEFT note or LEFT_EMPTY
        mask[LEFT_EMPTY] = True
        mask[LEFT_BASE: LEFT_BASE + LEFT_COUNT] = True

    elif last_token == LEFT_EMPTY or (LEFT_BASE <= last_token < LEFT_BASE + LEFT_COUNT):
        # After LEFT → RIGHT note or RIGHT_EMPTY
        mask[RIGHT_EMPTY] = True
        mask[RIGHT_BASE: RIGHT_BASE + RIGHT_COUNT] = True

    elif last_token == RIGHT_EMPTY or (RIGHT_BASE <= last_token < RIGHT_BASE + RIGHT_COUNT):
        # After RIGHT → POS (strictly increasing), BAR, or END
        # Only allow POS tokens with offset > last_pos_in_bar
        min_next = last_pos_in_bar + 1
        if min_next < POS_COUNT:
            mask[POS_BASE + min_next: POS_BASE + POS_COUNT] = True
        mask[BAR] = True
        mask[END] = True

    else:
        # Unknown state — allow everything except PAD/START
        mask[2:] = True

    return mask


def _sample_with_filter(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    """Sample a token from logits with temperature, top-k, and top-p filtering."""
    if temperature <= 0:
        return logits.argmax().item()

    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_val = values[-1]
        logits = torch.where(logits < min_val, torch.full_like(logits, float("-inf")), logits)

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative prob above threshold
        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_mask] = float("-inf")
        # Scatter back
        logits = torch.zeros_like(logits).scatter(0, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


@torch.no_grad()
def generate(
    model: BeatWeaverModel,
    mel_spectrogram: torch.Tensor,
    difficulty: str,
    config: ModelConfig,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int | None = None,
    mel_mask: torch.Tensor | None = None,
) -> list[int]:
    """Generate a token sequence autoregressively.

    Args:
        model: Trained BeatWeaverModel.
        mel_spectrogram: (n_mels, T_audio) — single spectrogram (no batch dim).
        difficulty: Difficulty name (e.g., "Expert").
        config: Model configuration.
        temperature: Sampling temperature (0 = greedy).
        top_k: Top-k filtering (0 = disabled).
        top_p: Top-p / nucleus filtering (1.0 = disabled).
        seed: Random seed for reproducibility.
        mel_mask: (T_audio,) — True for valid positions.

    Returns:
        List of token IDs including START and END.
    """
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    device = next(model.parameters()).device

    # Prepare mel: add batch dimension
    mel = mel_spectrogram.unsqueeze(0).to(device)  # (1, n_mels, T_audio)
    if mel_mask is not None:
        mel_mask = mel_mask.unsqueeze(0).to(device)  # (1, T_audio)

    # Encode audio once
    memory = model.encoder(mel, mel_mask)

    # Start with [START, DIFF_x]
    diff_token = difficulty_to_token(difficulty)
    tokens = [START, diff_token]
    last_pos_in_bar = -1  # Track last POS offset in current bar

    for _ in range(config.max_seq_len - 2):
        # Prepare decoder input
        token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        token_mask = torch.ones(1, len(tokens), dtype=torch.bool, device=device)

        logits = model.decoder(token_tensor, memory, token_mask, mel_mask)
        # logits: (1, seq_len, vocab_size) — take last position
        next_logits = logits[0, -1]  # (vocab_size,)

        # Apply grammar mask (with position tracking for one-note-per-color-per-beat)
        grammar_mask = _build_grammar_mask(tokens[-1], last_pos_in_bar).to(device)
        next_logits[~grammar_mask] = float("-inf")

        # Sample
        next_token = _sample_with_filter(next_logits, temperature, top_k, top_p)
        tokens.append(next_token)

        # Update position tracking
        if next_token == BAR:
            last_pos_in_bar = -1  # Reset on new bar
        elif POS_BASE <= next_token < POS_BASE + POS_COUNT:
            last_pos_in_bar = next_token - POS_BASE

        if next_token == END:
            break

    return tokens
