"""Tests for the transformer model architecture."""

import pytest
import torch

from beat_weaver.model.config import ModelConfig
from beat_weaver.model.transformer import (
    AudioEncoder,
    BeatWeaverModel,
    SinusoidalPositionalEncoding,
    TokenDecoder,
)


@pytest.fixture
def small_config():
    """A small config for fast testing."""
    return ModelConfig(
        vocab_size=291,
        max_seq_len=128,
        n_mels=80,
        encoder_layers=2,
        encoder_dim=64,
        encoder_heads=4,
        encoder_ff_dim=128,
        decoder_layers=2,
        decoder_dim=64,
        decoder_heads=4,
        decoder_ff_dim=128,
        dropout=0.0,
    )


class TestPositionalEncoding:
    def test_output_shape(self):
        pe = SinusoidalPositionalEncoding(d_model=64, max_len=512, dropout=0.0)
        x = torch.zeros(2, 100, 64)
        out = pe(x)
        assert out.shape == (2, 100, 64)

    def test_different_positions_differ(self):
        pe = SinusoidalPositionalEncoding(d_model=64, max_len=512, dropout=0.0)
        x = torch.zeros(1, 10, 64)
        out = pe(x)
        # Different positions should have different encodings
        assert not torch.allclose(out[0, 0], out[0, 1])


class TestAudioEncoder:
    def test_output_shape(self, small_config):
        encoder = AudioEncoder(small_config)
        mel = torch.randn(2, 80, 50)  # batch=2, n_mels=80, T=50
        mel_mask = torch.ones(2, 50, dtype=torch.bool)
        out = encoder(mel, mel_mask)
        assert out.shape == (2, 50, small_config.encoder_dim)

    def test_no_mask(self, small_config):
        encoder = AudioEncoder(small_config)
        mel = torch.randn(2, 80, 50)
        out = encoder(mel)
        assert out.shape == (2, 50, small_config.encoder_dim)


class TestTokenDecoder:
    def test_output_shape(self, small_config):
        decoder = TokenDecoder(small_config)
        tokens = torch.randint(0, 291, (2, 20))
        memory = torch.randn(2, 50, small_config.encoder_dim)
        token_mask = torch.ones(2, 20, dtype=torch.bool)
        memory_mask = torch.ones(2, 50, dtype=torch.bool)
        out = decoder(tokens, memory, token_mask, memory_mask)
        assert out.shape == (2, 20, small_config.vocab_size)

    def test_causal_masking(self, small_config):
        """Verify output at position i doesn't depend on future tokens."""
        decoder = TokenDecoder(small_config)
        decoder.eval()

        memory = torch.randn(1, 10, small_config.encoder_dim)
        tokens = torch.randint(0, 291, (1, 5))

        out_full = decoder(tokens, memory)

        # Change future token
        tokens_mod = tokens.clone()
        tokens_mod[0, 4] = (tokens[0, 4].item() + 1) % 291

        out_mod = decoder(tokens_mod, memory)

        # Position 3 output should be identical (doesn't see position 4)
        assert torch.allclose(out_full[0, 3], out_mod[0, 3], atol=1e-5)


class TestBeatWeaverModel:
    def test_forward_shape(self, small_config):
        model = BeatWeaverModel(small_config)
        mel = torch.randn(2, 80, 50)
        tokens = torch.randint(0, 291, (2, 20))
        mel_mask = torch.ones(2, 50, dtype=torch.bool)
        token_mask = torch.ones(2, 20, dtype=torch.bool)

        logits = model(mel, tokens, mel_mask, token_mask)
        assert logits.shape == (2, 20, 291)

    def test_gradient_flow(self, small_config):
        """Verify gradients flow from loss back through both encoder and decoder."""
        model = BeatWeaverModel(small_config)
        mel = torch.randn(2, 80, 50)
        tokens = torch.randint(0, 291, (2, 20))
        target = torch.randint(0, 291, (2, 20))

        logits = model(mel, tokens)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 291), target.reshape(-1),
        )
        loss.backward()

        # Check encoder has gradients
        for name, param in model.encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for encoder.{name}"
                break

        # Check decoder has gradients
        for name, param in model.decoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for decoder.{name}"
                break

    def test_count_parameters(self, small_config):
        model = BeatWeaverModel(small_config)
        count = model.count_parameters()
        assert count > 0
        assert isinstance(count, int)
