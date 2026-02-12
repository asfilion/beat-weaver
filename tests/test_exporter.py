"""Tests for the v2 Beat Saber map exporter."""

import json

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

from beat_weaver.model.exporter import export_map, export_notes
from beat_weaver.model.tokenizer import (
    BAR,
    DIFF_EXPERT,
    END,
    LEFT_EMPTY,
    POS_BASE,
    RIGHT_EMPTY,
    START,
    _encode_note_token,
    LEFT_BASE,
    RIGHT_BASE,
)
from beat_weaver.schemas.normalized import Note


@pytest.fixture
def audio_file(tmp_path):
    """Create a dummy audio file."""
    import soundfile as sf

    sr = 22050
    audio = np.zeros(sr * 2, dtype=np.float32)
    path = tmp_path / "song.ogg"
    sf.write(str(path), audio, sr)
    return path


class TestExportMap:
    def test_creates_folder_structure(self, tmp_path, audio_file):
        # Simple token sequence: one left note at beat 0
        tokens = [
            START, DIFF_EXPERT, BAR,
            POS_BASE + 0,
            _encode_note_token(LEFT_BASE, 1, 0, 1),
            RIGHT_EMPTY,
            END,
        ]
        output = tmp_path / "output_map"
        result = export_map(tokens, bpm=120.0, song_name="Test Song",
                           audio_path=audio_file, output_dir=output)

        assert (result / "Info.dat").exists()
        assert (result / "Expert.dat").exists()
        assert (result / "song.ogg").exists()

    def test_info_dat_structure(self, tmp_path, audio_file):
        tokens = [START, DIFF_EXPERT, BAR, POS_BASE, LEFT_EMPTY, RIGHT_EMPTY, END]
        output = tmp_path / "output_map"
        export_map(tokens, bpm=128.0, song_name="My Song",
                  audio_path=audio_file, output_dir=output)

        info = json.loads((output / "Info.dat").read_text())
        assert info["_version"] == "2.0.0"
        assert info["_songName"] == "My Song"
        assert info["_beatsPerMinute"] == 128.0
        assert info["_levelAuthorName"] == "BeatWeaver AI"

        sets = info["_difficultyBeatmapSets"]
        assert len(sets) == 1
        assert sets[0]["_beatmapCharacteristicName"] == "Standard"
        bm = sets[0]["_difficultyBeatmaps"][0]
        assert bm["_difficulty"] == "Expert"
        assert bm["_noteJumpMovementSpeed"] == 16

    def test_difficulty_dat_notes(self, tmp_path, audio_file):
        tokens = [
            START, DIFF_EXPERT, BAR,
            POS_BASE + 0,
            _encode_note_token(LEFT_BASE, 2, 1, 3),
            _encode_note_token(RIGHT_BASE, 1, 0, 1),
            END,
        ]
        output = tmp_path / "output_map"
        export_map(tokens, bpm=120.0, song_name="Test",
                  audio_path=audio_file, output_dir=output)

        dat = json.loads((output / "Expert.dat").read_text())
        assert dat["_version"] == "2.0.0"
        notes = dat["_notes"]
        assert len(notes) == 2

        # Left note (color=0)
        left = [n for n in notes if n["_type"] == 0][0]
        assert left["_lineIndex"] == 2
        assert left["_lineLayer"] == 1
        assert left["_cutDirection"] == 3

        # Right note (color=1)
        right = [n for n in notes if n["_type"] == 1][0]
        assert right["_lineIndex"] == 1
        assert right["_lineLayer"] == 0
        assert right["_cutDirection"] == 1

    def test_empty_map(self, tmp_path, audio_file):
        tokens = [START, DIFF_EXPERT, END]
        output = tmp_path / "output_map"
        export_map(tokens, bpm=120.0, song_name="Empty",
                  audio_path=audio_file, output_dir=output)

        dat = json.loads((output / "Expert.dat").read_text())
        assert dat["_notes"] == []


class TestExportNotes:
    def test_export_notes_creates_valid_map(self, tmp_path, audio_file):
        """export_notes produces valid v2 map files from a note list."""
        notes = [
            Note(beat=0.0, time_seconds=0.0, x=1, y=0, color=0, cut_direction=1),
            Note(beat=0.5, time_seconds=0.25, x=2, y=1, color=1, cut_direction=0),
            Note(beat=4.0, time_seconds=2.0, x=3, y=2, color=0, cut_direction=3),
        ]
        output = tmp_path / "notes_map"
        result = export_notes(notes, bpm=120.0, song_name="Test Notes",
                              audio_path=audio_file, output_dir=output)

        assert (result / "Info.dat").exists()
        assert (result / "Expert.dat").exists()
        assert (result / "song.ogg").exists()

        info = json.loads((result / "Info.dat").read_text())
        assert info["_songName"] == "Test Notes"
        assert info["_beatsPerMinute"] == 120.0

        dat = json.loads((result / "Expert.dat").read_text())
        assert len(dat["_notes"]) == 3
        # Notes should be sorted by time
        times = [n["_time"] for n in dat["_notes"]]
        assert times == sorted(times)
        # Check first note
        assert dat["_notes"][0]["_lineIndex"] == 1
        assert dat["_notes"][0]["_type"] == 0
        assert dat["_notes"][0]["_cutDirection"] == 1

    def test_export_notes_empty(self, tmp_path, audio_file):
        """export_notes handles an empty note list."""
        output = tmp_path / "empty_notes_map"
        result = export_notes([], bpm=120.0, song_name="Empty",
                              audio_path=audio_file, output_dir=output)

        dat = json.loads((result / "Expert.dat").read_text())
        assert dat["_notes"] == []
