"""Top-level orchestrator: parse an entire map folder into normalized beatmaps."""

import logging
from pathlib import Path

from beat_weaver.parsers.dat_reader import read_dat_file
from beat_weaver.parsers.info_parser import parse_info
from beat_weaver.schemas.detection import detect_beatmap_version
from beat_weaver.schemas.normalized import NormalizedBeatmap
from beat_weaver.schemas.v2 import parse_v2_notes, parse_v2_obstacles
from beat_weaver.schemas.v3 import parse_v3_notes, parse_v3_obstacles
from beat_weaver.schemas.v4 import parse_v4_notes, parse_v4_obstacles

logger = logging.getLogger(__name__)

_NOTE_PARSERS = {
    "2": parse_v2_notes,
    "3": parse_v3_notes,
    "4": parse_v4_notes,
}

_OBSTACLE_PARSERS = {
    "2": parse_v2_obstacles,
    "3": parse_v3_obstacles,
    "4": parse_v4_obstacles,
}


def _find_info_dat(folder: Path) -> Path:
    """Find Info.dat in a folder, case-insensitive."""
    for name in ("Info.dat", "info.dat", "INFO.dat"):
        path = folder / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No Info.dat found in {folder}")


def parse_map_folder(
    folder: Path,
    source: str = "unknown",
    source_id: str = "",
) -> list[NormalizedBeatmap]:
    """Parse an entire map folder into a list of NormalizedBeatmap (one per difficulty).

    Logs and skips individual difficulties that fail to parse.
    """
    info_path = _find_info_dat(folder)
    info_data = read_dat_file(info_path)
    metadata, difficulties = parse_info(info_data, source=source, source_id=source_id)

    results = []
    for diff_info, filename in difficulties:
        dat_path = folder / filename
        if not dat_path.exists():
            logger.warning("Missing difficulty file: %s", dat_path)
            continue
        try:
            beatmap_data = read_dat_file(dat_path)
            version = detect_beatmap_version(beatmap_data)

            parse_notes = _NOTE_PARSERS.get(version)
            parse_obstacles = _OBSTACLE_PARSERS.get(version)
            if parse_notes is None or parse_obstacles is None:
                logger.warning("Unsupported beatmap version %s in %s", version, dat_path)
                continue

            notes, bombs = parse_notes(beatmap_data, metadata.bpm)
            obstacles = parse_obstacles(beatmap_data, metadata.bpm)

            diff_info.note_count = len(notes)
            diff_info.bomb_count = len(bombs)
            diff_info.obstacle_count = len(obstacles)
            if metadata.bpm > 0 and notes:
                duration = notes[-1].time_seconds - notes[0].time_seconds
                if duration > 0:
                    diff_info.nps = len(notes) / duration

            results.append(NormalizedBeatmap(
                metadata=metadata,
                difficulty_info=diff_info,
                notes=notes,
                bombs=bombs,
                obstacles=obstacles,
            ))
        except Exception:
            logger.exception("Failed to parse %s", dat_path)
            continue

    return results
