"""Write normalized Beat Saber beatmap data to Parquet files and JSON metadata."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from beat_weaver.schemas.normalized import NormalizedBeatmap

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


def write_parquet(beatmaps: list[NormalizedBeatmap], output_dir: Path) -> None:
    """Write a list of normalized beatmaps to Parquet files and JSON metadata.

    Produces four files inside *output_dir*:
      - notes.parquet
      - bombs.parquet
      - obstacles.parquet
      - metadata.json

    Parameters
    ----------
    beatmaps:
        Normalized beatmap objects (one per difficulty).
    output_dir:
        Directory to write output files into. Created if it doesn't exist.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    note_rows: list[dict] = []
    bomb_rows: list[dict] = []
    obstacle_rows: list[dict] = []
    # Deduplicate metadata by song hash.
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
        for note in bm.notes:
            note_rows.append(
                {
                    "song_hash": song_hash,
                    "source": source,
                    "difficulty": difficulty,
                    "characteristic": characteristic,
                    "bpm": bpm,
                    "beat": note.beat,
                    "time_seconds": note.time_seconds,
                    "x": note.x,
                    "y": note.y,
                    "color": note.color,
                    "cut_direction": note.cut_direction,
                    "angle_offset": note.angle_offset,
                }
            )

        # --- Bombs ---
        for bomb in bm.bombs:
            bomb_rows.append(
                {
                    "song_hash": song_hash,
                    "source": source,
                    "difficulty": difficulty,
                    "characteristic": characteristic,
                    "bpm": bpm,
                    "beat": bomb.beat,
                    "time_seconds": bomb.time_seconds,
                    "x": bomb.x,
                    "y": bomb.y,
                }
            )

        # --- Obstacles ---
        for obs in bm.obstacles:
            obstacle_rows.append(
                {
                    "song_hash": song_hash,
                    "source": source,
                    "difficulty": difficulty,
                    "characteristic": characteristic,
                    "bpm": bpm,
                    "beat": obs.beat,
                    "time_seconds": obs.time_seconds,
                    "duration_beats": obs.duration_beats,
                    "x": obs.x,
                    "y": obs.y,
                    "width": obs.width,
                    "height": obs.height,
                }
            )

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

    # --- Write Parquet files ---
    notes_table = pa.Table.from_pylist(note_rows, schema=NOTES_SCHEMA)
    pq.write_table(notes_table, output_dir / "notes.parquet", compression="snappy")

    bombs_table = pa.Table.from_pylist(bomb_rows, schema=BOMBS_SCHEMA)
    pq.write_table(bombs_table, output_dir / "bombs.parquet", compression="snappy")

    obstacles_table = pa.Table.from_pylist(obstacle_rows, schema=OBSTACLES_SCHEMA)
    pq.write_table(
        obstacles_table, output_dir / "obstacles.parquet", compression="snappy"
    )

    # --- Write metadata JSON ---
    metadata_list = list(metadata_by_hash.values())
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2)


def read_notes_parquet(path: Path) -> pa.Table:
    """Read a notes Parquet file and return it as an Arrow table."""
    return pq.read_table(path)
