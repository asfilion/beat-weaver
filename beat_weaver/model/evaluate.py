"""Post-generation quality metrics for Beat Saber maps."""

from __future__ import annotations

from collections import Counter

from beat_weaver.schemas.normalized import Note

# Cut direction vectors for parity checking
# 0=Up, 1=Down, 2=Left, 3=Right, 4=UpLeft, 5=UpRight, 6=DownLeft, 7=DownRight, 8=Any
_DIR_VECTORS = {
    0: (0, 1),    # Up
    1: (0, -1),   # Down
    2: (-1, 0),   # Left
    3: (1, 0),    # Right
    4: (-1, 1),   # UpLeft
    5: (1, 1),    # UpRight
    6: (-1, -1),  # DownLeft
    7: (1, -1),   # DownRight
    8: (0, 0),    # Any (no specific direction)
}


def _onset_f1(
    generated: list[Note],
    reference: list[Note],
    tolerance_seconds: float = 0.04,
) -> float:
    """Match generated note times to reference within tolerance.

    Returns F1 score (harmonic mean of precision and recall).
    """
    if not generated and not reference:
        return 1.0
    if not generated or not reference:
        return 0.0

    gen_times = sorted(n.time_seconds for n in generated)
    ref_times = sorted(n.time_seconds for n in reference)

    # Greedy matching
    matched_gen = set()
    matched_ref = set()
    ref_idx = 0

    for gi, gt in enumerate(gen_times):
        # Find closest unmatched ref
        best_ri = None
        best_dist = float("inf")
        for ri in range(max(0, ref_idx - 5), min(len(ref_times), ref_idx + 20)):
            if ri in matched_ref:
                continue
            dist = abs(gt - ref_times[ri])
            if dist <= tolerance_seconds and dist < best_dist:
                best_dist = dist
                best_ri = ri
        if best_ri is not None:
            matched_gen.add(gi)
            matched_ref.add(best_ri)
            ref_idx = best_ri

    tp = len(matched_gen)
    precision = tp / len(gen_times) if gen_times else 0.0
    recall = tp / len(ref_times) if ref_times else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _nps_accuracy(generated: list[Note], reference: list[Note]) -> float:
    """Compare notes-per-second between generated and reference.

    Returns 1 - |nps_gen - nps_ref| / nps_ref, clamped to [0, 1].
    """
    def _nps(notes: list[Note]) -> float:
        if len(notes) < 2:
            return 0.0
        times = [n.time_seconds for n in notes]
        duration = max(times) - min(times)
        return len(notes) / duration if duration > 0 else 0.0

    nps_gen = _nps(generated)
    nps_ref = _nps(reference)
    if nps_ref == 0:
        return 1.0 if nps_gen == 0 else 0.0
    return max(0.0, 1.0 - abs(nps_gen - nps_ref) / nps_ref)


def _beat_alignment(notes: list[Note], subdivisions_per_beat: int = 16) -> float:
    """Mean distance from each note beat to nearest 1/16th grid position.

    Lower is better. Returns 0.0 for perfectly aligned notes.
    """
    if not notes:
        return 0.0
    grid = 1.0 / subdivisions_per_beat
    distances = []
    for note in notes:
        nearest = round(note.beat / grid) * grid
        distances.append(abs(note.beat - nearest))
    return sum(distances) / len(distances)


def _parity_violations(notes: list[Note]) -> float:
    """Count parity violations per hand.

    Tracks swing state (forehand/backhand) and counts direction reversals
    that violate natural swing patterns.

    Returns fraction of notes that are parity violations.
    """
    if not notes:
        return 0.0

    # Separate by hand
    left_notes = sorted([n for n in notes if n.color == 0], key=lambda n: n.beat)
    right_notes = sorted([n for n in notes if n.color == 1], key=lambda n: n.beat)

    violations = 0
    total = 0

    for hand_notes in [left_notes, right_notes]:
        last_dir = None
        for note in hand_notes:
            if note.cut_direction == 8:  # Any — never a violation
                last_dir = None
                continue
            total += 1
            if last_dir is not None:
                dy_last = _DIR_VECTORS[last_dir][1]
                dy_curr = _DIR_VECTORS[note.cut_direction][1]
                # Parity violation: two consecutive same-vertical-direction swings
                # (e.g., down-down or up-up without alternating)
                if dy_last != 0 and dy_curr != 0 and dy_last == dy_curr:
                    violations += 1
            last_dir = note.cut_direction

    return violations / max(1, total)


def _pattern_diversity(notes: list[Note], window: int = 4) -> float:
    """Fraction of unique n-note subsequences.

    Higher is more diverse. Returns ratio of unique to total windows.
    """
    if len(notes) < window:
        return 1.0

    sorted_notes = sorted(notes, key=lambda n: n.beat)
    patterns: list[tuple] = []
    for i in range(len(sorted_notes) - window + 1):
        pattern = tuple(
            (n.x, n.y, n.color, n.cut_direction)
            for n in sorted_notes[i: i + window]
        )
        patterns.append(pattern)

    if not patterns:
        return 1.0
    return len(set(patterns)) / len(patterns)


def _notes_per_second(notes: list[Note]) -> float:
    """Compute notes per second."""
    if len(notes) < 2:
        return 0.0
    times = [n.time_seconds for n in notes]
    duration = max(times) - min(times)
    return len(notes) / duration if duration > 0 else 0.0


def evaluate_map(
    generated_notes: list[Note],
    reference_notes: list[Note],
    bpm: float,
) -> dict[str, float]:
    """Evaluate generated map against a reference.

    Returns dict with metrics:
        onset_f1, nps_accuracy, beat_alignment, parity_violation_rate,
        pattern_diversity, nps
    """
    return {
        "onset_f1": _onset_f1(generated_notes, reference_notes),
        "nps_accuracy": _nps_accuracy(generated_notes, reference_notes),
        "beat_alignment": _beat_alignment(generated_notes),
        "parity_violation_rate": _parity_violations(generated_notes),
        "pattern_diversity": _pattern_diversity(generated_notes),
        "nps": _notes_per_second(generated_notes),
    }


def evaluate_standalone(notes: list[Note], bpm: float) -> dict[str, float]:
    """Evaluate a map without a reference (standalone quality metrics).

    Returns dict with metrics:
        beat_alignment, parity_violation_rate, pattern_diversity, nps
    """
    return {
        "beat_alignment": _beat_alignment(notes),
        "parity_violation_rate": _parity_violations(notes),
        "pattern_diversity": _pattern_diversity(notes),
        "nps": _notes_per_second(notes),
    }
