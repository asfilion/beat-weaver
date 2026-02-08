"""Tests for inference and grammar-constrained generation."""

import pytest
import torch

from beat_weaver.model.config import ModelConfig
from beat_weaver.model.inference import _build_grammar_mask, generate
from beat_weaver.model.tokenizer import (
    BAR,
    DIFF_EASY,
    DIFF_EXPERT,
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
)
from beat_weaver.model.transformer import BeatWeaverModel


class TestGrammarMask:
    def test_after_start(self):
        mask = _build_grammar_mask(START)
        # Only difficulty tokens allowed
        assert mask[DIFF_EASY:DIFF_EXPERT_PLUS + 1].all()
        assert not mask[START]
        assert not mask[BAR]
        assert not mask[END]

    def test_after_difficulty(self):
        mask = _build_grammar_mask(DIFF_EXPERT)
        # Only BAR allowed
        assert mask[BAR]
        assert mask.sum() == 1

    def test_after_bar(self):
        mask = _build_grammar_mask(BAR)
        # POS, BAR, or END
        assert mask[POS_BASE:POS_BASE + POS_COUNT].all()
        assert mask[BAR]
        assert mask[END]
        assert not mask[START]
        assert not mask[LEFT_EMPTY]

    def test_after_pos(self):
        mask = _build_grammar_mask(POS_BASE + 10)
        # LEFT tokens
        assert mask[LEFT_EMPTY]
        assert mask[LEFT_BASE:LEFT_BASE + LEFT_COUNT].all()
        assert not mask[BAR]
        assert not mask[RIGHT_EMPTY]

    def test_after_left(self):
        mask = _build_grammar_mask(LEFT_BASE + 5)
        # RIGHT tokens
        assert mask[RIGHT_EMPTY]
        assert mask[RIGHT_BASE:RIGHT_BASE + RIGHT_COUNT].all()
        assert not mask[BAR]
        assert not mask[LEFT_EMPTY]

    def test_after_left_empty(self):
        mask = _build_grammar_mask(LEFT_EMPTY)
        # RIGHT tokens
        assert mask[RIGHT_EMPTY]
        assert mask[RIGHT_BASE:RIGHT_BASE + RIGHT_COUNT].all()

    def test_after_right(self):
        mask = _build_grammar_mask(RIGHT_BASE + 5)
        # POS, BAR, or END
        assert mask[POS_BASE:POS_BASE + POS_COUNT].all()
        assert mask[BAR]
        assert mask[END]

    def test_after_right_empty(self):
        mask = _build_grammar_mask(RIGHT_EMPTY)
        assert mask[POS_BASE:POS_BASE + POS_COUNT].all()
        assert mask[BAR]
        assert mask[END]


class TestGenerate:
    @pytest.fixture
    def small_model(self):
        config = ModelConfig(
            vocab_size=291,
            max_seq_len=64,
            n_mels=80,
            encoder_layers=1,
            encoder_dim=32,
            encoder_heads=4,
            encoder_ff_dim=64,
            decoder_layers=1,
            decoder_dim=32,
            decoder_heads=4,
            decoder_ff_dim=64,
            dropout=0.0,
        )
        model = BeatWeaverModel(config)
        return model, config

    def test_starts_and_ends_correctly(self, small_model):
        model, config = small_model
        mel = torch.randn(80, 20)
        tokens = generate(model, mel, "Expert", config, temperature=1.0, seed=42)
        assert tokens[0] == START
        assert tokens[1] == DIFF_EXPERT
        # Should end with END or hit max_seq_len
        assert tokens[-1] == END or len(tokens) == config.max_seq_len

    def test_grammar_valid_sequence(self, small_model):
        """Every generated token should follow grammar rules."""
        model, config = small_model
        mel = torch.randn(80, 20)
        tokens = generate(model, mel, "Expert", config, temperature=1.0, seed=42)

        for i in range(1, len(tokens)):
            prev = tokens[i - 1]
            curr = tokens[i]
            mask = _build_grammar_mask(prev)
            assert mask[curr], (
                f"Token {curr} not valid after {prev} at position {i}. "
                f"Sequence so far: {tokens[:i+1]}"
            )

    def test_deterministic_with_seed(self, small_model):
        model, config = small_model
        mel = torch.randn(80, 20)
        t1 = generate(model, mel, "Expert", config, temperature=0.5, seed=123)
        t2 = generate(model, mel, "Expert", config, temperature=0.5, seed=123)
        assert t1 == t2

    def test_greedy_decoding(self, small_model):
        model, config = small_model
        mel = torch.randn(80, 20)
        tokens = generate(model, mel, "Expert", config, temperature=0)
        assert tokens[0] == START
        assert tokens[1] == DIFF_EXPERT
