"""Parse Beat Saber v3 format beatmaps into normalized data structures.

V3 maps use short single-letter keys (b, x, y, c, d, a) and separate arrays
for color notes, bomb notes, and obstacles.
"""

from beat_weaver.schemas.normalized import Bomb, Note, Obstacle


def parse_v3_notes(beatmap: dict, bpm: float) -> tuple[list[Note], list[Bomb]]:
    """Parse notes and bombs from a v3 beatmap.

    Args:
        beatmap: Parsed JSON dict of a v3 difficulty file.
        bpm: Beats per minute for time conversion.

    Returns:
        Tuple of (notes sorted by beat, bombs sorted by beat).
    """
    notes: list[Note] = []
    bombs: list[Bomb] = []

    for raw in beatmap.get("colorNotes", []):
        beat = raw["b"]
        time_seconds = beat * 60.0 / bpm
        notes.append(Note(
            beat=beat,
            time_seconds=time_seconds,
            x=raw.get("x", 0),
            y=raw.get("y", 0),
            color=raw.get("c", 0),
            cut_direction=raw.get("d", 8),
            angle_offset=raw.get("a", 0),
        ))

    for raw in beatmap.get("bombNotes", []):
        beat = raw["b"]
        time_seconds = beat * 60.0 / bpm
        bombs.append(Bomb(
            beat=beat,
            time_seconds=time_seconds,
            x=raw.get("x", 0),
            y=raw.get("y", 0),
        ))

    notes.sort(key=lambda n: n.beat)
    bombs.sort(key=lambda b: b.beat)
    return notes, bombs


def parse_v3_obstacles(beatmap: dict, bpm: float) -> list[Obstacle]:
    """Parse obstacles from a v3 beatmap.

    Args:
        beatmap: Parsed JSON dict of a v3 difficulty file.
        bpm: Beats per minute for time conversion.

    Returns:
        List of obstacles sorted by beat.
    """
    obstacles: list[Obstacle] = []

    for raw in beatmap.get("obstacles", []):
        beat = raw["b"]
        time_seconds = beat * 60.0 / bpm
        obstacles.append(Obstacle(
            beat=beat,
            time_seconds=time_seconds,
            duration_beats=raw.get("d", 1.0),
            x=raw.get("x", 0),
            y=raw.get("y", 0),
            width=raw.get("w", 1),
            height=raw.get("h", 5),
        ))

    obstacles.sort(key=lambda o: o.beat)
    return obstacles
