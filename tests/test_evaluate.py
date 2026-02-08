"""Tests for evaluation metrics."""

import pytest

from beat_weaver.model.evaluate import (
    _beat_alignment,
    _notes_per_second,
    _nps_accuracy,
    _onset_f1,
    _parity_violations,
    _pattern_diversity,
    evaluate_map,
    evaluate_standalone,
)
from beat_weaver.schemas.normalized import Note


def _note(beat: float, color: int = 0, direction: int = 1, x: int = 0, y: int = 0, bpm: float = 120.0) -> Note:
    return Note(
        beat=beat, time_seconds=beat * 60.0 / bpm,
        x=x, y=y, color=color, cut_direction=direction,
    )


class TestOnsetF1:
    def test_perfect_match(self):
        notes = [_note(0.0), _note(1.0), _note(2.0)]
        f1 = _onset_f1(notes, notes)
        assert f1 == pytest.approx(1.0)

    def test_no_overlap(self):
        gen = [_note(0.0), _note(1.0)]
        ref = [_note(10.0), _note(11.0)]
        f1 = _onset_f1(gen, ref)
        assert f1 == 0.0

    def test_both_empty(self):
        assert _onset_f1([], []) == 1.0

    def test_one_empty(self):
        assert _onset_f1([_note(0.0)], []) == 0.0
        assert _onset_f1([], [_note(0.0)]) == 0.0

    def test_close_match(self):
        gen = [_note(0.0), _note(1.0)]
        ref = [_note(0.01), _note(1.02)]  # within 40ms tolerance
        f1 = _onset_f1(gen, ref)
        assert f1 == pytest.approx(1.0)


class TestNPSAccuracy:
    def test_identical(self):
        notes = [_note(0.0), _note(1.0), _note(2.0)]
        acc = _nps_accuracy(notes, notes)
        assert acc == pytest.approx(1.0)

    def test_double_density(self):
        gen = [_note(i * 0.5) for i in range(6)]
        ref = [_note(i * 1.0) for i in range(3)]
        acc = _nps_accuracy(gen, ref)
        assert acc < 1.0
        assert acc >= 0.0


class TestBeatAlignment:
    def test_perfect_alignment(self):
        """Notes on exact 1/16th grid."""
        notes = [_note(0.0), _note(0.25), _note(0.5), _note(1.0)]
        assert _beat_alignment(notes) == pytest.approx(0.0)

    def test_off_grid(self):
        notes = [_note(0.1)]  # not on 1/16th grid (0.0625 increments)
        alignment = _beat_alignment(notes)
        assert alignment > 0.0

    def test_empty(self):
        assert _beat_alignment([]) == 0.0


class TestParityViolations:
    def test_no_violations(self):
        """Alternating up/down is correct parity."""
        notes = [
            _note(0.0, direction=1),  # Down
            _note(1.0, direction=0),  # Up
            _note(2.0, direction=1),  # Down
        ]
        rate = _parity_violations(notes)
        assert rate == 0.0

    def test_all_same_direction(self):
        """All down = violations after the first."""
        notes = [
            _note(0.0, direction=1),  # Down
            _note(1.0, direction=1),  # Down (violation)
            _note(2.0, direction=1),  # Down (violation)
        ]
        rate = _parity_violations(notes)
        assert rate == pytest.approx(2 / 3)

    def test_any_direction_resets(self):
        """Direction 8 (Any) never causes violations."""
        notes = [
            _note(0.0, direction=1),  # Down
            _note(1.0, direction=8),  # Any — resets
            _note(2.0, direction=1),  # Down — no violation (after Any)
        ]
        rate = _parity_violations(notes)
        assert rate == 0.0

    def test_empty(self):
        assert _parity_violations([]) == 0.0


class TestPatternDiversity:
    def test_all_unique(self):
        notes = [
            _note(i, x=i % 4, y=i % 3, direction=i % 9)
            for i in range(10)
        ]
        diversity = _pattern_diversity(notes)
        assert diversity == 1.0

    def test_all_same(self):
        notes = [_note(float(i), x=0, y=0, direction=1) for i in range(10)]
        diversity = _pattern_diversity(notes)
        # Only 1 unique pattern out of 7 windows
        assert diversity < 0.5

    def test_short_sequence(self):
        notes = [_note(0.0)]
        assert _pattern_diversity(notes) == 1.0


class TestEvaluateMap:
    def test_returns_all_metrics(self):
        gen = [_note(0.0), _note(1.0)]
        ref = [_note(0.0), _note(1.0)]
        result = evaluate_map(gen, ref, bpm=120.0)
        assert "onset_f1" in result
        assert "nps_accuracy" in result
        assert "beat_alignment" in result
        assert "parity_violation_rate" in result
        assert "pattern_diversity" in result
        assert "nps" in result


class TestEvaluateStandalone:
    def test_returns_metrics(self):
        notes = [_note(0.0), _note(1.0), _note(2.0)]
        result = evaluate_standalone(notes, bpm=120.0)
        assert "beat_alignment" in result
        assert "parity_violation_rate" in result
        assert "pattern_diversity" in result
        assert "nps" in result
