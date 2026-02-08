"""Tests for the v2 Beat Saber map exporter."""

import json

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

from beat_weaver.model.exporter import export_map
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
