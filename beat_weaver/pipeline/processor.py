"""Process individual map folders into normalized data."""

import hashlib
import logging
from pathlib import Path

from beat_weaver.parsers.beatmap_parser import parse_map_folder
from beat_weaver.schemas.normalized import NormalizedBeatmap

logger = logging.getLogger(__name__)


def compute_map_hash(folder: Path) -> str:
    """Compute a SHA-256 hash of all .dat files in a map folder."""
    hasher = hashlib.sha256()
    for dat_file in sorted(folder.glob("*.dat")):
        hasher.update(dat_file.read_bytes())
    # Also include Info.dat / info.dat
    for name in ("Info.dat", "info.dat"):
        info = folder / name
        if info.exists():
            hasher.update(info.read_bytes())
            break
    return hasher.hexdigest()


def process_map_folder(
    folder: Path, source: str, source_id: str
) -> list[NormalizedBeatmap]:
    """Process one map folder, returning normalized beatmaps or empty list on failure."""
    try:
        content_hash = compute_map_hash(folder)
        beatmaps = parse_map_folder(folder, source=source, source_id=source_id)
        for bm in beatmaps:
            bm.metadata.hash = content_hash
        return beatmaps
    except Exception:
        logger.exception("Failed to process %s", folder)
        return []
