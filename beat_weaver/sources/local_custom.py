"""Read custom levels from a local Beat Saber installation."""

import logging
from pathlib import Path
from typing import Iterator

from beat_weaver.parsers.beatmap_parser import parse_map_folder
from beat_weaver.schemas.normalized import NormalizedBeatmap

logger = logging.getLogger(__name__)

DEFAULT_BEAT_SABER_PATH = Path(
    r"C:\Program Files (x86)\Steam\steamapps\common\Beat Saber"
)
CUSTOM_LEVELS_SUBPATH = Path("Beat Saber_Data") / "CustomLevels"


def iter_local_custom_maps(
    beat_saber_path: Path = DEFAULT_BEAT_SABER_PATH,
) -> Iterator[NormalizedBeatmap]:
    """Iterate all custom level folders and yield NormalizedBeatmap objects.

    Skips folders that fail to parse.
    """
    custom_dir = beat_saber_path / CUSTOM_LEVELS_SUBPATH
    if not custom_dir.exists():
        logger.warning("CustomLevels directory not found: %s", custom_dir)
        return

    for folder in sorted(custom_dir.iterdir()):
        if not folder.is_dir():
            continue
        try:
            yield from parse_map_folder(
                folder, source="local_custom", source_id=folder.name
            )
        except Exception:
            logger.exception("Failed to parse local custom level: %s", folder.name)
            continue
