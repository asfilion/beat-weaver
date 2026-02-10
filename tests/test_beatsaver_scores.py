"""Tests for BeatSaver score persistence in the pipeline."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from beat_weaver.sources.beatsaver import load_beatsaver_meta


class TestBeatsaverMeta:
    def test_load_meta_exists(self, tmp_path):
        """Load _beatsaver_meta.json from a map directory."""
        meta = {
            "id": "abc123",
            "stats": {"score": 0.95, "upvotes": 100, "downvotes": 5},
        }
        meta_path = tmp_path / "_beatsaver_meta.json"
        meta_path.write_text(json.dumps(meta))

        loaded = load_beatsaver_meta(tmp_path)
        assert loaded is not None
        assert loaded["stats"]["score"] == 0.95
        assert loaded["stats"]["upvotes"] == 100

    def test_load_meta_missing(self, tmp_path):
        """Returns None when no _beatsaver_meta.json exists."""
        result = load_beatsaver_meta(tmp_path)
        assert result is None


class TestScoreInjection:
    def test_scores_injected_into_metadata(self, tmp_path):
        """Scores from _beatsaver_meta.json are injected into processed beatmaps."""
        from beat_weaver.cli import _process_single_folder

        # Create a minimal map folder with Info.dat + _beatsaver_meta.json
        map_folder = tmp_path / "test_map"
        map_folder.mkdir()

        info = {
            "_version": "2.0.0",
            "_songName": "Test Song",
            "_songSubName": "",
            "_songAuthorName": "Author",
            "_levelAuthorName": "Mapper",
            "_beatsPerMinute": 120,
            "_songFilename": "song.ogg",
            "_difficultyBeatmapSets": [{
                "_beatmapCharacteristicName": "Standard",
                "_difficultyBeatmaps": [{
                    "_difficulty": "Expert",
                    "_difficultyRank": 7,
                    "_beatmapFilename": "ExpertStandard.dat",
                    "_noteJumpMovementSpeed": 16,
                    "_noteJumpStartBeatOffset": 0,
                }],
            }],
        }
        (map_folder / "Info.dat").write_text(json.dumps(info))

        # Create a v2 difficulty file with a single note
        diff_data = {
            "_version": "2.0.0",
            "_notes": [
                {"_time": 1.0, "_lineIndex": 1, "_lineLayer": 0,
                 "_type": 0, "_cutDirection": 1},
            ],
            "_events": [],
            "_obstacles": [],
        }
        (map_folder / "ExpertStandard.dat").write_text(json.dumps(diff_data))

        # Create _beatsaver_meta.json with score data
        meta = {
            "id": "abc",
            "stats": {"score": 0.92, "upvotes": 50, "downvotes": 3},
        }
        (map_folder / "_beatsaver_meta.json").write_text(json.dumps(meta))

        # Process the folder
        beatmaps = _process_single_folder(map_folder, source="beatsaver")
        assert len(beatmaps) >= 1
        assert beatmaps[0].metadata.score == 0.92
        assert beatmaps[0].metadata.upvotes == 50
        assert beatmaps[0].metadata.downvotes == 3
