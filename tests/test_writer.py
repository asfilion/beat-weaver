"""Tests for Parquet writer — row groups, file splitting, and reader."""

import pyarrow.parquet as pq
import pytest

from beat_weaver.schemas.normalized import (
    Bomb,
    DifficultyInfo,
    Note,
    NormalizedBeatmap,
    Obstacle,
    SongMetadata,
)
from beat_weaver.storage.writer import (
    MAX_FILE_BYTES,
    read_notes_parquet,
    write_parquet,
)


def _make_beatmap(
    song_hash: str,
    n_notes: int = 10,
    difficulty: str = "Expert",
    source: str = "test",
) -> NormalizedBeatmap:
    """Create a minimal beatmap with *n_notes* notes."""
    notes = [
        Note(
            beat=float(i),
            time_seconds=float(i) * 0.5,
            x=i % 4,
            y=i % 3,
            color=i % 2,
            cut_direction=i % 9,
            angle_offset=0,
        )
        for i in range(n_notes)
    ]
    bombs = [Bomb(beat=0.5, time_seconds=0.25, x=0, y=0)]
    obstacles = [
        Obstacle(beat=1.0, time_seconds=0.5, duration_beats=2.0, x=0, y=0, width=1, height=3)
    ]
    return NormalizedBeatmap(
        metadata=SongMetadata(
            source=source,
            source_id=f"id_{song_hash}",
            hash=song_hash,
            bpm=120.0,
        ),
        difficulty_info=DifficultyInfo(
            characteristic="Standard",
            difficulty=difficulty,
            difficulty_rank=7,
            note_jump_speed=16.0,
            note_jump_offset=0.0,
            note_count=n_notes,
        ),
        notes=notes,
        bombs=bombs,
        obstacles=obstacles,
    )


class TestWriteParquet:
    """Test the core write_parquet function."""

    def test_produces_numbered_files(self, tmp_path):
        """write_parquet should produce notes_0000.parquet, not notes.parquet."""
        beatmaps = [_make_beatmap("hash_a"), _make_beatmap("hash_b")]
        write_parquet(beatmaps, tmp_path)

        notes_files = sorted(tmp_path.glob("notes_*.parquet"))
        assert len(notes_files) >= 1
        assert notes_files[0].name == "notes_0000.parquet"

        # Old single-file should NOT exist
        assert not (tmp_path / "notes.parquet").exists()

    def test_one_row_group_per_song(self, tmp_path):
        """Each song_hash should get its own row group."""
        beatmaps = [
            _make_beatmap("aaa", n_notes=5),
            _make_beatmap("bbb", n_notes=8),
            _make_beatmap("ccc", n_notes=3),
        ]
        write_parquet(beatmaps, tmp_path)

        pf = pq.ParquetFile(tmp_path / "notes_0000.parquet")
        assert pf.metadata.num_row_groups == 3

        # Verify each row group contains rows for exactly one song_hash
        hashes_per_rg = set()
        for i in range(pf.metadata.num_row_groups):
            rg_table = pf.read_row_group(i)
            unique = set(rg_table.column("song_hash").to_pylist())
            assert len(unique) == 1, f"Row group {i} has multiple hashes: {unique}"
            hashes_per_rg.update(unique)
        assert hashes_per_rg == {"aaa", "bbb", "ccc"}

    def test_total_row_count_preserved(self, tmp_path):
        """Total notes across all files should equal what was written."""
        beatmaps = [
            _make_beatmap("song1", n_notes=20),
            _make_beatmap("song2", n_notes=30),
        ]
        write_parquet(beatmaps, tmp_path)

        table = read_notes_parquet(tmp_path)
        assert table.num_rows == 50

    def test_metadata_json_written(self, tmp_path):
        """metadata.json should still be produced."""
        import json

        beatmaps = [_make_beatmap("x")]
        write_parquet(beatmaps, tmp_path)

        meta = json.loads((tmp_path / "metadata.json").read_text())
        assert len(meta) == 1
        assert meta[0]["hash"] == "x"

    def test_bombs_and_obstacles_written(self, tmp_path):
        """Bombs and obstacles should also use numbered files."""
        beatmaps = [_make_beatmap("h1")]
        write_parquet(beatmaps, tmp_path)

        bombs_files = list(tmp_path.glob("bombs_*.parquet"))
        obstacles_files = list(tmp_path.glob("obstacles_*.parquet"))
        assert len(bombs_files) >= 1
        assert len(obstacles_files) >= 1


class TestFileSplitting:
    """Test that files are split when exceeding max_file_bytes."""

    def test_splits_at_max_size(self, tmp_path):
        """When max_file_bytes is tiny, each song should get its own file."""
        beatmaps = [
            _make_beatmap("alpha", n_notes=100),
            _make_beatmap("beta", n_notes=100),
            _make_beatmap("gamma", n_notes=100),
        ]
        # Use a tiny limit so every song triggers a new file
        write_parquet(beatmaps, tmp_path, max_file_bytes=1)

        notes_files = sorted(tmp_path.glob("notes_*.parquet"))
        # With max_file_bytes=1, each song's row group should trigger a split
        assert len(notes_files) >= 2, f"Expected multiple files, got {len(notes_files)}"

    def test_all_data_survives_split(self, tmp_path):
        """Total rows should be preserved even across multiple files."""
        beatmaps = [
            _make_beatmap(f"song_{i}", n_notes=50) for i in range(5)
        ]
        write_parquet(beatmaps, tmp_path, max_file_bytes=1)

        table = read_notes_parquet(tmp_path)
        assert table.num_rows == 250

    def test_single_file_when_small(self, tmp_path):
        """Small datasets should produce exactly one file."""
        beatmaps = [_make_beatmap("tiny", n_notes=5)]
        write_parquet(beatmaps, tmp_path)

        notes_files = list(tmp_path.glob("notes_*.parquet"))
        assert len(notes_files) == 1


class TestReadNotesParquet:
    """Test the reader handles both old and new layouts."""

    def test_reads_directory_of_files(self, tmp_path):
        """Reader should combine multiple numbered files."""
        beatmaps = [
            _make_beatmap(f"s{i}", n_notes=10) for i in range(3)
        ]
        write_parquet(beatmaps, tmp_path, max_file_bytes=1)

        table = read_notes_parquet(tmp_path)
        assert table.num_rows == 30

    def test_reads_single_legacy_file(self, tmp_path):
        """Reader should fall back to notes.parquet for backward compat."""
        # Manually write a single file in old layout
        import pyarrow as pa
        from beat_weaver.storage.writer import NOTES_SCHEMA

        rows = {name: [] for name in NOTES_SCHEMA.names}
        rows["song_hash"].append("legacy")
        rows["source"].append("test")
        rows["difficulty"].append("Hard")
        rows["characteristic"].append("Standard")
        rows["bpm"].append(120.0)
        rows["beat"].append(1.0)
        rows["time_seconds"].append(0.5)
        rows["x"].append(0)
        rows["y"].append(0)
        rows["color"].append(0)
        rows["cut_direction"].append(1)
        rows["angle_offset"].append(0)
        table = pa.table(rows, schema=NOTES_SCHEMA)
        pq.write_table(table, tmp_path / "notes.parquet")

        result = read_notes_parquet(tmp_path)
        assert result.num_rows == 1
        assert result.column("song_hash").to_pylist() == ["legacy"]

    def test_reads_single_file_path(self, tmp_path):
        """Reader should accept a direct path to a .parquet file."""
        beatmaps = [_make_beatmap("direct", n_notes=5)]
        write_parquet(beatmaps, tmp_path)

        single_file = sorted(tmp_path.glob("notes_*.parquet"))[0]
        table = read_notes_parquet(single_file)
        assert table.num_rows == 5

    def test_raises_on_empty_dir(self, tmp_path):
        """Reader should raise FileNotFoundError on empty directory."""
        with pytest.raises(FileNotFoundError):
            read_notes_parquet(tmp_path)

    def test_row_groups_are_queryable(self, tmp_path):
        """Row groups should allow reading only specific songs."""
        beatmaps = [
            _make_beatmap("keep", n_notes=10),
            _make_beatmap("skip", n_notes=10),
        ]
        write_parquet(beatmaps, tmp_path)

        pf = pq.ParquetFile(tmp_path / "notes_0000.parquet")
        # Read only the first row group and check its hash
        rg0 = pf.read_row_group(0)
        rg0_hash = set(rg0.column("song_hash").to_pylist())
        assert len(rg0_hash) == 1  # Single song per row group
