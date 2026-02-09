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

    # Columnar lists — avoids 900K+ dict allocations per column
    n_hashes: list[str] = []
    n_sources: list[str] = []
    n_difficulties: list[str] = []
    n_characteristics: list[str] = []
    n_bpms: list[float] = []
    n_beats: list[float] = []
    n_times: list[float] = []
    n_xs: list[int] = []
    n_ys: list[int] = []
    n_colors: list[int] = []
    n_cut_dirs: list[int] = []
    n_angles: list[int] = []

    b_hashes: list[str] = []
    b_sources: list[str] = []
    b_difficulties: list[str] = []
    b_characteristics: list[str] = []
    b_bpms: list[float] = []
    b_beats: list[float] = []
    b_times: list[float] = []
    b_xs: list[int] = []
    b_ys: list[int] = []

    o_hashes: list[str] = []
    o_sources: list[str] = []
    o_difficulties: list[str] = []
    o_characteristics: list[str] = []
    o_bpms: list[float] = []
    o_beats: list[float] = []
    o_times: list[float] = []
    o_durations: list[float] = []
    o_xs: list[int] = []
    o_ys: list[int] = []
    o_widths: list[int] = []
    o_heights: list[int] = []

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
            n_hashes.append(song_hash)
            n_sources.append(source)
            n_difficulties.append(difficulty)
            n_characteristics.append(characteristic)
            n_bpms.append(bpm)
            n_beats.append(note.beat)
            n_times.append(note.time_seconds)
            n_xs.append(note.x)
            n_ys.append(note.y)
            n_colors.append(note.color)
            n_cut_dirs.append(note.cut_direction)
            n_angles.append(note.angle_offset)

        # --- Bombs ---
        for bomb in bm.bombs:
            b_hashes.append(song_hash)
            b_sources.append(source)
            b_difficulties.append(difficulty)
            b_characteristics.append(characteristic)
            b_bpms.append(bpm)
            b_beats.append(bomb.beat)
            b_times.append(bomb.time_seconds)
            b_xs.append(bomb.x)
            b_ys.append(bomb.y)

        # --- Obstacles ---
        for obs in bm.obstacles:
            o_hashes.append(song_hash)
            o_sources.append(source)
            o_difficulties.append(difficulty)
            o_characteristics.append(characteristic)
            o_bpms.append(bpm)
            o_beats.append(obs.beat)
            o_times.append(obs.time_seconds)
            o_durations.append(obs.duration_beats)
            o_xs.append(obs.x)
            o_ys.append(obs.y)
            o_widths.append(obs.width)
            o_heights.append(obs.height)

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

    # --- Write Parquet files (from columnar lists, no dict overhead) ---
    notes_table = pa.table(
        {
            "song_hash": n_hashes, "source": n_sources,
            "difficulty": n_difficulties, "characteristic": n_characteristics,
            "bpm": n_bpms, "beat": n_beats, "time_seconds": n_times,
            "x": n_xs, "y": n_ys, "color": n_colors,
            "cut_direction": n_cut_dirs, "angle_offset": n_angles,
        },
        schema=NOTES_SCHEMA,
    )
    pq.write_table(notes_table, output_dir / "notes.parquet", compression="snappy")

    bombs_table = pa.table(
        {
            "song_hash": b_hashes, "source": b_sources,
            "difficulty": b_difficulties, "characteristic": b_characteristics,
            "bpm": b_bpms, "beat": b_beats, "time_seconds": b_times,
            "x": b_xs, "y": b_ys,
        },
        schema=BOMBS_SCHEMA,
    )
    pq.write_table(bombs_table, output_dir / "bombs.parquet", compression="snappy")

    obstacles_table = pa.table(
        {
            "song_hash": o_hashes, "source": o_sources,
            "difficulty": o_difficulties, "characteristic": o_characteristics,
            "bpm": o_bpms, "beat": o_beats, "time_seconds": o_times,
            "duration_beats": o_durations, "x": o_xs, "y": o_ys,
            "width": o_widths, "height": o_heights,
        },
        schema=OBSTACLES_SCHEMA,
    )
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
