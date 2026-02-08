"""Tests for schema detection and version-specific parsers."""

import json
from beat_weaver.schemas.detection import detect_info_version, detect_beatmap_version
from beat_weaver.schemas.v2 import parse_v2_notes, parse_v2_obstacles
from beat_weaver.schemas.v3 import parse_v3_notes, parse_v3_obstacles
from beat_weaver.schemas.v4 import parse_v4_notes, parse_v4_obstacles
import pytest


class TestDetection:
    def test_detect_info_v2(self):
        assert detect_info_version({"_version": "2.0.0"}) == "2"

    def test_detect_info_v4(self):
        assert detect_info_version({"version": "4.0.0"}) == "4"

    def test_detect_info_v3(self):
        assert detect_info_version({"version": "3.0.0"}) == "3"

    def test_detect_info_fallback(self):
        assert detect_info_version({}) == "2"

    def test_detect_beatmap_v2(self):
        assert detect_beatmap_version({"_version": "2.1.0", "_notes": []}) == "2"

    def test_detect_beatmap_v3(self):
        assert detect_beatmap_version({"version": "3.0.0", "colorNotes": []}) == "3"

    def test_detect_beatmap_v4(self):
        assert detect_beatmap_version({"version": "4.0.0", "colorNotesData": []}) == "4"

    def test_detect_beatmap_v2_fallback(self):
        assert detect_beatmap_version({"_notes": []}) == "2"

    def test_detect_beatmap_unknown(self):
        with pytest.raises(ValueError):
            detect_beatmap_version({"unknown": True})


class TestV2Parser:
    def test_parse_notes(self):
        beatmap = {
            "_notes": [
                {"_time": 4.0, "_lineIndex": 1, "_lineLayer": 0, "_type": 0, "_cutDirection": 1},
                {"_time": 4.0, "_lineIndex": 2, "_lineLayer": 0, "_type": 1, "_cutDirection": 1},
                {"_time": 8.0, "_lineIndex": 1, "_lineLayer": 1, "_type": 3, "_cutDirection": 0},
            ]
        }
        notes, bombs = parse_v2_notes(beatmap, bpm=120.0)
        assert len(notes) == 2
        assert len(bombs) == 1
        assert notes[0].color == 0
        assert notes[1].color == 1
        assert notes[0].x == 1
        assert notes[0].time_seconds == pytest.approx(4.0 * 60.0 / 120.0)
        assert bombs[0].beat == 8.0

    def test_parse_obstacles(self):
        beatmap = {
            "_obstacles": [
                {"_time": 10.0, "_lineIndex": 0, "_lineLayer": 0, "_type": 0, "_duration": 2.0, "_width": 1},
                {"_time": 15.0, "_lineIndex": 0, "_lineLayer": 0, "_type": 1, "_duration": 1.0, "_width": 4},
            ]
        }
        obstacles = parse_v2_obstacles(beatmap, bpm=120.0)
        assert len(obstacles) == 2
        assert obstacles[0].height == 5  # full wall
        assert obstacles[0].y == 0
        assert obstacles[1].height == 3  # crouch wall
        assert obstacles[1].y == 2

    def test_empty_beatmap(self):
        notes, bombs = parse_v2_notes({}, bpm=120.0)
        assert notes == []
        assert bombs == []


class TestV3Parser:
    def test_parse_notes(self):
        beatmap = {
            "colorNotes": [
                {"b": 4.0, "x": 1, "y": 0, "c": 0, "d": 1, "a": 15},
            ],
            "bombNotes": [
                {"b": 6.0, "x": 2, "y": 1},
            ],
        }
        notes, bombs = parse_v3_notes(beatmap, bpm=140.0)
        assert len(notes) == 1
        assert notes[0].angle_offset == 15
        assert len(bombs) == 1
        assert bombs[0].x == 2

    def test_parse_obstacles(self):
        beatmap = {
            "obstacles": [
                {"b": 10.0, "x": 0, "y": 0, "d": 2.0, "w": 1, "h": 3},
            ],
        }
        obstacles = parse_v3_obstacles(beatmap, bpm=140.0)
        assert len(obstacles) == 1
        assert obstacles[0].height == 3
        assert obstacles[0].width == 1


class TestV4Parser:
    def test_parse_notes_with_deref(self):
        beatmap = {
            "colorNotes": [
                {"b": 2.0, "r": 0, "i": 0},
                {"b": 4.0, "r": 0, "i": 1},
            ],
            "colorNotesData": [
                {"x": 1, "y": 0, "c": 0, "d": 1, "a": 0},
                {"x": 2, "y": 2, "c": 1, "d": 0, "a": 10},
            ],
            "bombNotes": [{"b": 3.0, "r": 0, "i": 0}],
            "bombNotesData": [{"x": 1, "y": 1}],
        }
        notes, bombs = parse_v4_notes(beatmap, bpm=150.0)
        assert len(notes) == 2
        assert notes[0].x == 1
        assert notes[0].color == 0
        assert notes[1].x == 2
        assert notes[1].angle_offset == 10
        assert len(bombs) == 1

    def test_skip_out_of_range_index(self):
        beatmap = {
            "colorNotes": [{"b": 1.0, "r": 0, "i": 99}],
            "colorNotesData": [{"x": 0, "y": 0, "c": 0, "d": 0}],
            "bombNotes": [],
            "bombNotesData": [],
        }
        notes, bombs = parse_v4_notes(beatmap, bpm=120.0)
        assert len(notes) == 0

    def test_shared_data_index(self):
        beatmap = {
            "colorNotes": [
                {"b": 1.0, "r": 0, "i": 0},
                {"b": 2.0, "r": 0, "i": 0},
                {"b": 3.0, "r": 0, "i": 0},
            ],
            "colorNotesData": [
                {"x": 1, "y": 0, "c": 0, "d": 1},
            ],
            "bombNotes": [],
            "bombNotesData": [],
        }
        notes, _ = parse_v4_notes(beatmap, bpm=120.0)
        assert len(notes) == 3
        assert all(n.x == 1 for n in notes)
