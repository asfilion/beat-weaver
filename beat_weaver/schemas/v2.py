"""Parse Beat Saber v2 format beatmaps into normalized data structures.

V2 maps use underscore-prefixed keys (_notes, _obstacles, _events) and store
notes and bombs together in a single _notes array differentiated by _type.
"""

from beat_weaver.schemas.normalized import Bomb, Note, Obstacle


def parse_v2_notes(beatmap: dict, bpm: float) -> tuple[list[Note], list[Bomb]]:
    """Parse notes and bombs from a v2 beatmap.

    Args:
        beatmap: Parsed JSON dict of a v2 difficulty file.
        bpm: Beats per minute for time conversion.

    Returns:
        Tuple of (notes sorted by beat, bombs sorted by beat).
    """
    notes: list[Note] = []
    bombs: list[Bomb] = []

    for raw in beatmap.get("_notes", []):
        beat = raw["_time"]
        time_seconds = beat * 60.0 / bpm
        x = raw["_lineIndex"]
        y = raw["_lineLayer"]
        note_type = raw["_type"]

        if note_type == 3:
            bombs.append(Bomb(
                beat=beat,
                time_seconds=time_seconds,
                x=x,
                y=y,
            ))
        elif note_type in (0, 1):
            notes.append(Note(
                beat=beat,
                time_seconds=time_seconds,
                x=x,
                y=y,
                color=note_type,
                cut_direction=raw["_cutDirection"],
                angle_offset=0,
            ))

    notes.sort(key=lambda n: n.beat)
    bombs.sort(key=lambda b: b.beat)
    return notes, bombs


def parse_v2_obstacles(beatmap: dict, bpm: float) -> list[Obstacle]:
    """Parse obstacles from a v2 beatmap.

    Args:
        beatmap: Parsed JSON dict of a v2 difficulty file.
        bpm: Beats per minute for time conversion.

    Returns:
        List of obstacles sorted by beat.
    """
    obstacles: list[Obstacle] = []

    for raw in beatmap.get("_obstacles", []):
        beat = raw["_time"]
        time_seconds = beat * 60.0 / bpm
        obstacle_type = raw["_type"]

        if obstacle_type == 1:
            y = 2
            height = 3
        else:
            y = 0
            height = 5

        obstacles.append(Obstacle(
            beat=beat,
            time_seconds=time_seconds,
            duration_beats=raw["_duration"],
            x=raw["_lineIndex"],
            y=y,
            width=raw["_width"],
            height=height,
        ))

    obstacles.sort(key=lambda o: o.beat)
    return obstacles
