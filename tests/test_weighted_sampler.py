"""Tests for dataset source weighting and WeightedRandomSampler construction."""

from unittest.mock import MagicMock

from beat_weaver.model.dataset import build_weighted_sampler


def _make_dataset(samples):
    """Create a mock BeatSaberDataset with given sample dicts."""
    ds = MagicMock()
    ds.samples = samples
    ds.__len__ = lambda self: len(samples)
    return ds


def test_returns_none_for_single_source():
    """No sampler needed when all samples are from one source."""
    ds = _make_dataset([
        {"source": "beatsaver", "score": 0.85},
        {"source": "beatsaver", "score": 0.90},
    ])
    assert build_weighted_sampler(ds, official_ratio=0.2) is None


def test_returns_none_for_all_official():
    ds = _make_dataset([
        {"source": "official", "score": None},
        {"source": "official", "score": None},
    ])
    assert build_weighted_sampler(ds, official_ratio=0.2) is None


def test_returns_sampler_for_mixed_sources():
    ds = _make_dataset([
        {"source": "official", "score": None},
        {"source": "beatsaver", "score": 0.80},
    ])
    sampler = build_weighted_sampler(ds, official_ratio=0.2)
    assert sampler is not None


def test_weight_ratio_matches_official_ratio():
    """Verify that the sum of official weights / total weights = official_ratio."""
    samples = [
        {"source": "official", "score": None},
        {"source": "official", "score": None},
        {"source": "beatsaver", "score": 0.80},
        {"source": "beatsaver", "score": 0.90},
        {"source": "beatsaver", "score": 0.85},
    ]
    ds = _make_dataset(samples)

    for ratio in [0.1, 0.2, 0.3, 0.5]:
        sampler = build_weighted_sampler(ds, official_ratio=ratio)
        weights = list(sampler.weights)

        official_weight_sum = sum(weights[i] for i, s in enumerate(samples) if s["source"] == "official")
        total_weight_sum = sum(weights)

        actual_ratio = official_weight_sum / total_weight_sum
        assert abs(actual_ratio - ratio) < 1e-9, f"Expected {ratio}, got {actual_ratio}"


def test_custom_weights_proportional_to_score():
    """Higher-scored custom maps should get higher weights."""
    samples = [
        {"source": "official", "score": None},
        {"source": "beatsaver", "score": 0.75},
        {"source": "beatsaver", "score": 0.95},
    ]
    ds = _make_dataset(samples)
    sampler = build_weighted_sampler(ds, official_ratio=0.2)
    weights = list(sampler.weights)

    # Custom weights should equal their scores
    assert weights[1] == 0.75
    assert weights[2] == 0.95


def test_missing_score_defaults_to_one():
    """Samples without a score should get weight 1.0."""
    samples = [
        {"source": "official", "score": None},
        {"source": "beatsaver", "score": None},
        {"source": "local_custom"},  # no "score" key at all
    ]
    ds = _make_dataset(samples)
    sampler = build_weighted_sampler(ds, official_ratio=0.2)
    weights = list(sampler.weights)

    # Both custom samples should have weight 1.0
    assert weights[1] == 1.0
    assert weights[2] == 1.0


def test_num_samples_equals_dataset_length():
    samples = [
        {"source": "official", "score": None},
        {"source": "beatsaver", "score": 0.80},
        {"source": "beatsaver", "score": 0.85},
    ]
    ds = _make_dataset(samples)
    sampler = build_weighted_sampler(ds, official_ratio=0.2)
    assert sampler.num_samples == len(samples)


def test_replacement_is_true():
    """Weighted sampling must use replacement to oversample the minority source."""
    samples = [
        {"source": "official", "score": None},
        {"source": "beatsaver", "score": 0.80},
    ]
    ds = _make_dataset(samples)
    sampler = build_weighted_sampler(ds, official_ratio=0.2)
    assert sampler.replacement is True


def test_extreme_ratios():
    """Very small and very large official_ratio should still produce valid weights."""
    samples = [
        {"source": "official", "score": None},
        {"source": "beatsaver", "score": 0.80},
        {"source": "beatsaver", "score": 0.90},
    ]
    ds = _make_dataset(samples)

    # Very small ratio — official maps rarely sampled
    sampler = build_weighted_sampler(ds, official_ratio=0.01)
    weights = list(sampler.weights)
    assert all(w > 0 for w in weights)

    # Large ratio — official maps dominate
    sampler = build_weighted_sampler(ds, official_ratio=0.9)
    weights = list(sampler.weights)
    assert all(w > 0 for w in weights)
    # Official weight should be much larger
    assert weights[0] > weights[1]
    assert weights[0] > weights[2]
