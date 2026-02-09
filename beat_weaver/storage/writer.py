"""Write normalized Beat Saber beatmap data to Parquet files and JSON metadata."""

import json
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from beat_weaver.schemas.normalized import NormalizedBeatmap

logger = logging.getLogger(__name__)

# Maximum Parquet file size in bytes before starting a new file.
MAX_FILE_BYTES: int = 1_000_000_000  # 1 GB

# --- Arrow schemas -----------------------------------------------------------

NOTES_SCHEMA = pa.schema(
    [
        pa.field("song_hash", pa.string()),
        pa.field("source", pa.string()),
        pa.field("difficulty", pa.string()),
        pa.field("characteristic", pa.string()),
        pa.field("bpm", pa.float32()),
        pa.field("beat", pa.float32()),
        pa.field("time_seconds", pa.float32()),
        pa.field("x", pa.int8()),
        pa.field("y", pa.int8()),
        pa.field("color", pa.int8()),
        pa.field("cut_direction", pa.int8()),
        pa.field("angle_offset", pa.int16()),
    ]
)

BOMBS_SCHEMA = pa.schema(
    [
        pa.field("song_hash", pa.string()),
        pa.field("source", pa.string()),
        pa.field("difficulty", pa.string()),
        pa.field("characteristic", pa.string()),
        pa.field("bpm", pa.float32()),
        pa.field("beat", pa.float32()),
        pa.field("time_seconds", pa.float32()),
        pa.field("x", pa.int8()),
        pa.field("y", pa.int8()),
    ]
)

OBSTACLES_SCHEMA = pa.schema(
    [
        pa.field("song_hash", pa.string()),
        pa.field("source", pa.string()),
        pa.field("difficulty", pa.string()),
        pa.field("characteristic", pa.string()),
        pa.field("bpm", pa.float32()),
        pa.field("beat", pa.float32()),
        pa.field("time_seconds", pa.float32()),
        pa.field("duration_beats", pa.float32()),
        pa.field("x", pa.int8()),
        pa.field("y", pa.int8()),
        pa.field("width", pa.int8()),
        pa.field("height", pa.int8()),
    ]
)


# --- Public API --------------------------------------------------------------


def _write_tables_chunked(
    tables_by_hash: dict[str, pa.Table],
    output_dir: Path,
    prefix: str,
    schema: pa.Schema,
    max_file_bytes: int = MAX_FILE_BYTES,
) -> list[Path]:
    """Write Arrow tables as one-row-group-per-song files, splitting at *max_file_bytes*.

    Each song_hash gets its own row group inside the Parquet file.  When the
    current file would exceed *max_file_bytes*, a new file is started.

    Returns the list of written file paths.
    """
    written: list[Path] = []
    file_idx = 0
    writer: pq.ParquetWriter | None = None
    current_path: Path | None = None

    def _open_writer() -> tuple[pq.ParquetWriter, Path]:
        nonlocal file_idx
        p = output_dir / f"{prefix}_{file_idx:04d}.parquet"
        w = pq.ParquetWriter(p, schema, compression="snappy")
        file_idx += 1
        return w, p

    for _hash, table in sorted(tables_by_hash.items()):
        if table.num_rows == 0:
            continue

        # Start a new file if none open yet
        if writer is None:
            writer, current_path = _open_writer()

        writer.write_table(table)

        # Check if current file exceeds max size
        assert current_path is not None
        current_size = current_path.stat().st_size
        if current_size >= max_file_bytes:
            writer.close()
            written.append(current_path)
            logger.debug("Closed %s (%d bytes)", current_path.name, current_size)
            writer, current_path = _open_writer()

    # Close the final file
    if writer is not None:
        writer.close()
        assert current_path is not None
        written.append(current_path)

    return written


def write_parquet(
    beatmaps: list[NormalizedBeatmap],
    output_dir: Path,
    max_file_bytes: int = MAX_FILE_BYTES,
) -> None:
    """Write a list of normalized beatmaps to Parquet files and JSON metadata.

    Each song_hash gets its own row group so that readers can push down
    predicates and skip irrelevant data.  When a Parquet file exceeds
    *max_file_bytes* (default 1 GB), a new numbered file is started.

    Produces inside *output_dir*:
      - notes_NNNN.parquet  (one or more)
      - bombs_NNNN.parquet  (one or more)
      - obstacles_NNNN.parquet  (one or more)
      - metadata.json

    Parameters
    ----------
    beatmaps:
        Normalized beatmap objects (one per difficulty).
    output_dir:
        Directory to write output files into. Created if it doesn't exist.
    max_file_bytes:
        Maximum size in bytes per Parquet file before splitting.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Accumulate columnar data grouped by song_hash
    notes_by_hash: dict[str, dict[str, list]] = {}
    bombs_by_hash: dict[str, dict[str, list]] = {}
    obstacles_by_hash: dict[str, dict[str, list]] = {}
    metadata_by_hash: dict[str, dict] = {}

    for bm in beatmaps:
        meta = bm.metadata
        diff = bm.difficulty_info

        song_hash = meta.hash
        source = meta.source
        difficulty = diff.difficulty
        characteristic = diff.characteristic
        bpm = meta.bpm

        # --- Notes ---
        if song_hash not in notes_by_hash:
            notes_by_hash[song_hash] = {k: [] for k in NOTES_SCHEMA.names}
        cols = notes_by_hash[song_hash]
        for note in bm.notes:
            cols["song_hash"].append(song_hash)
            cols["source"].append(source)
            cols["difficulty"].append(difficulty)
            cols["characteristic"].append(characteristic)
            cols["bpm"].append(bpm)
            cols["beat"].append(note.beat)
            cols["time_seconds"].append(note.time_seconds)
            cols["x"].append(note.x)
            cols["y"].append(note.y)
            cols["color"].append(note.color)
            cols["cut_direction"].append(note.cut_direction)
            cols["angle_offset"].append(note.angle_offset)

        # --- Bombs ---
        if song_hash not in bombs_by_hash:
            bombs_by_hash[song_hash] = {k: [] for k in BOMBS_SCHEMA.names}
        bcols = bombs_by_hash[song_hash]
        for bomb in bm.bombs:
            bcols["song_hash"].append(song_hash)
            bcols["source"].append(source)
            bcols["difficulty"].append(difficulty)
            bcols["characteristic"].append(characteristic)
            bcols["bpm"].append(bpm)
            bcols["beat"].append(bomb.beat)
            bcols["time_seconds"].append(bomb.time_seconds)
            bcols["x"].append(bomb.x)
            bcols["y"].append(bomb.y)

        # --- Obstacles ---
        if song_hash not in obstacles_by_hash:
            obstacles_by_hash[song_hash] = {k: [] for k in OBSTACLES_SCHEMA.names}
        ocols = obstacles_by_hash[song_hash]
        for obs in bm.obstacles:
            ocols["song_hash"].append(song_hash)
            ocols["source"].append(source)
            ocols["difficulty"].append(difficulty)
            ocols["characteristic"].append(characteristic)
            ocols["bpm"].append(bpm)
            ocols["beat"].append(obs.beat)
            ocols["time_seconds"].append(obs.time_seconds)
            ocols["duration_beats"].append(obs.duration_beats)
            ocols["x"].append(obs.x)
            ocols["y"].append(obs.y)
            ocols["width"].append(obs.width)
            ocols["height"].append(obs.height)

        # --- Metadata (deduplicated by hash) ---
        if song_hash not in metadata_by_hash:
            metadata_by_hash[song_hash] = {
                "hash": song_hash,
                "source": source,
                "source_id": meta.source_id,
                "song_name": meta.song_name,
                "song_author": meta.song_author,
                "mapper_name": meta.mapper_name,
                "bpm": bpm,
                "score": meta.score,
                "difficulties": [],
            }

        metadata_by_hash[song_hash]["difficulties"].append(
            {
                "characteristic": characteristic,
                "difficulty": difficulty,
                "note_count": diff.note_count,
                "nps": diff.nps,
            }
        )

    # Clamp int8 columns to [-128, 127] to handle mapping-extension maps
    def _clamp8(values: list[int]) -> list[int]:
        return [max(-128, min(127, v)) for v in values]

    for cols in notes_by_hash.values():
        for k in ("x", "y", "color", "cut_direction"):
            cols[k] = _clamp8(cols[k])
    for bcols in bombs_by_hash.values():
        for k in ("x", "y"):
            bcols[k] = _clamp8(bcols[k])
    for ocols in obstacles_by_hash.values():
        for k in ("x", "y", "width", "height"):
            ocols[k] = _clamp8(ocols[k])

    # Convert accumulated columns to Arrow tables (one per song_hash)
    notes_tables = {
        h: pa.table(cols, schema=NOTES_SCHEMA)
        for h, cols in notes_by_hash.items()
    }
    bombs_tables = {
        h: pa.table(cols, schema=BOMBS_SCHEMA)
        for h, cols in bombs_by_hash.items()
    }
    obstacles_tables = {
        h: pa.table(cols, schema=OBSTACLES_SCHEMA)
        for h, cols in obstacles_by_hash.items()
    }

    # Write chunked Parquet files (one row group per song, split at max_file_bytes)
    notes_files = _write_tables_chunked(
        notes_tables, output_dir, "notes", NOTES_SCHEMA, max_file_bytes,
    )
    bombs_files = _write_tables_chunked(
        bombs_tables, output_dir, "bombs", BOMBS_SCHEMA, max_file_bytes,
    )
    obstacles_files = _write_tables_chunked(
        obstacles_tables, output_dir, "obstacles", OBSTACLES_SCHEMA, max_file_bytes,
    )

    logger.info(
        "Wrote %d notes files, %d bombs files, %d obstacles files to %s",
        len(notes_files), len(bombs_files), len(obstacles_files), output_dir,
    )

    # --- Write metadata JSON ---
    metadata_list = list(metadata_by_hash.values())
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2)


def read_notes_parquet(path: Path) -> pa.Table:
    """Read notes Parquet file(s) and return a single Arrow table.

    Accepts either a single ``.parquet`` file or a directory containing
    ``notes_*.parquet`` files.
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("notes_*.parquet"))
        if not files:
            # Backward compat: try single-file layout
            single = path / "notes.parquet"
            if single.exists():
                return pq.read_table(single)
            raise FileNotFoundError(f"No notes Parquet files in {path}")
        tables = [pq.read_table(f) for f in files]
        return pa.concat_tables(tables)
    return pq.read_table(path)
