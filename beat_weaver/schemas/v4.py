"""Parse Beat Saber v4 format beatmaps into normalized data structures.

V4 maps use an index-based dereferencing pattern: note/bomb/obstacle arrays
contain beat timing and an index into separate data arrays that hold the
position, color, and direction information.
"""

from beat_weaver.schemas.normalized import Bomb, Note, Obstacle


def parse_v4_notes(beatmap: dict, bpm: float) -> tuple[list[Note], list[Bomb]]:
    """Parse notes and bombs from a v4 beatmap.

    Args:
        beatmap: Parsed JSON dict of a v4 difficulty file.
        bpm: Beats per minute for time conversion.

    Returns:
        Tuple of (notes sorted by beat, bombs sorted by beat).
    """
    notes: list[Note] = []
    bombs: list[Bomb] = []

    color_notes = beatmap.get("colorNotes", [])
    color_notes_data = beatmap.get("colorNotesData", [])

    for raw in color_notes:
        beat = raw["b"]
        idx = raw["i"]
        if idx < 0 or idx >= len(color_notes_data):
            continue
        data = color_notes_data[idx]
        time_seconds = beat * 60.0 / bpm
        notes.append(Note(
            beat=beat,
            time_seconds=time_seconds,
            x=data["x"],
            y=data["y"],
            color=data["c"],
            cut_direction=data["d"],
            angle_offset=data.get("a", 0),
        ))

    bomb_notes = beatmap.get("bombNotes", [])
    bomb_notes_data = beatmap.get("bombNotesData", [])

    for raw in bomb_notes:
        beat = raw["b"]
        idx = raw["i"]
        if idx < 0 or idx >= len(bomb_notes_data):
            continue
        data = bomb_notes_data[idx]
        time_seconds = beat * 60.0 / bpm
        bombs.append(Bomb(
            beat=beat,
            time_seconds=time_seconds,
            x=data["x"],
            y=data["y"],
        ))

    notes.sort(key=lambda n: n.beat)
    bombs.sort(key=lambda b: b.beat)
    return notes, bombs


def parse_v4_obstacles(beatmap: dict, bpm: float) -> list[Obstacle]:
    """Parse obstacles from a v4 beatmap.

    Args:
        beatmap: Parsed JSON dict of a v4 difficulty file.
        bpm: Beats per minute for time conversion.

    Returns:
        List of obstacles sorted by beat.
    """
    obstacles: list[Obstacle] = []

    obstacle_entries = beatmap.get("obstacles", [])
    obstacles_data = beatmap.get("obstaclesData", [])

    for raw in obstacle_entries:
        beat = raw["b"]
        idx = raw["i"]
        if idx < 0 or idx >= len(obstacles_data):
            continue
        data = obstacles_data[idx]
        time_seconds = beat * 60.0 / bpm
        obstacles.append(Obstacle(
            beat=beat,
            time_seconds=time_seconds,
            duration_beats=data["d"],
            x=data["x"],
            y=data.get("y", 0),
            width=data["w"],
            height=data.get("h", 5),
        ))

    obstacles.sort(key=lambda o: o.beat)
    return obstacles
