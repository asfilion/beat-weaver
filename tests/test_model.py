"""Tests for the transformer model architecture."""

import pytest

torch = pytest.importorskip("torch")

from beat_weaver.model.config import ModelConfig
from beat_weaver.model.transformer import (
    AudioEncoder,
    BeatWeaverModel,
    RotaryPositionalEncoding,
    SinusoidalPositionalEncoding,
    TokenDecoder,
    _apply_rotary_emb,
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


class TestRoPE:
    @pytest.fixture
    def rope_config(self):
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
            use_rope=True,
        )

    def test_rope_output_shape(self, rope_config):
        """RoPE encoder produces correct output shape."""
        encoder = AudioEncoder(rope_config)
        mel = torch.randn(2, 80, 50)
        mel_mask = torch.ones(2, 50, dtype=torch.bool)
        out = encoder(mel, mel_mask)
        assert out.shape == (2, 50, rope_config.encoder_dim)

    def test_rope_decoder_output_shape(self, rope_config):
        """RoPE decoder produces correct output shape."""
        decoder = TokenDecoder(rope_config)
        tokens = torch.randint(0, 291, (2, 20))
        memory = torch.randn(2, 50, rope_config.encoder_dim)
        token_mask = torch.ones(2, 20, dtype=torch.bool)
        memory_mask = torch.ones(2, 50, dtype=torch.bool)
        out = decoder(tokens, memory, token_mask, memory_mask)
        assert out.shape == (2, 20, rope_config.vocab_size)

    def test_rope_full_model_backward(self, rope_config):
        """Both RoPE encoder and decoder produce gradients."""
        model = BeatWeaverModel(rope_config)
        mel = torch.randn(2, 80, 50)
        tokens = torch.randint(0, 291, (2, 20))
        target = torch.randint(0, 291, (2, 20))

        logits = model(mel, tokens)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 291), target.reshape(-1),
        )
        loss.backward()

        # Check encoder has gradients
        has_grad = False
        for param in model.encoder.parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        assert has_grad, "No gradients in RoPE encoder"

        # Check decoder has gradients
        has_grad = False
        for param in model.decoder.parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        assert has_grad, "No gradients in RoPE decoder"

    def test_rotary_encoding_shape(self):
        """RotaryPositionalEncoding returns correct shapes."""
        rope = RotaryPositionalEncoding(dim=16, max_len=512)
        cos, sin = rope(100, torch.device("cpu"))
        assert cos.shape == (1, 1, 100, 8)  # dim/2 = 8
        assert sin.shape == (1, 1, 100, 8)

    def test_apply_rotary_preserves_shape(self):
        """_apply_rotary_emb preserves input shape."""
        x = torch.randn(2, 4, 10, 16)  # batch, heads, seq, head_dim
        cos = torch.ones(1, 1, 10, 8)
        sin = torch.zeros(1, 1, 10, 8)
        out = _apply_rotary_emb(x, cos, sin)
        assert out.shape == x.shape


class TestOnsetFeatureInput:
    def test_onset_input_dim(self):
        """AudioEncoder accepts n_mels+1 input when onset features enabled."""
        config = ModelConfig(
            n_mels=80,
            encoder_layers=1,
            encoder_dim=64,
            encoder_heads=4,
            encoder_ff_dim=128,
            decoder_layers=1,
            decoder_dim=64,
            decoder_heads=4,
            decoder_ff_dim=128,
            dropout=0.0,
            use_onset_features=True,
        )
        encoder = AudioEncoder(config)
        mel = torch.randn(2, 81, 50)  # 80 mel + 1 onset
        mel_mask = torch.ones(2, 50, dtype=torch.bool)
        out = encoder(mel, mel_mask)
        assert out.shape == (2, 50, config.encoder_dim)


class TestMediumConfig:
    def test_medium_config_forward(self):
        """Forward pass works with medium config dimensions."""
        config = ModelConfig.load("configs/medium.json")
        model = BeatWeaverModel(config)
        # Use small sequences for speed
        mel = torch.randn(1, 81, 100)  # onset features = 81 channels
        tokens = torch.randint(0, 291, (1, 32))
        mel_mask = torch.ones(1, 100, dtype=torch.bool)
        token_mask = torch.ones(1, 32, dtype=torch.bool)

        logits = model(mel, tokens, mel_mask, token_mask)
        assert logits.shape == (1, 32, 291)
        assert model.count_parameters() > 1_000_000  # Should be ~8M
