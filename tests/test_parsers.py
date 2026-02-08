"""Tests for Info.dat parsing, dat_reader, and beatmap_parser."""

import gzip
import json
from pathlib import Path

import pytest

from beat_weaver.parsers.beatmap_parser import parse_map_folder
from beat_weaver.parsers.dat_reader import read_dat_file
from beat_weaver.parsers.info_parser import parse_info

FIXTURES = Path(__file__).parent / "fixtures"


class TestDatReader:
    def test_read_json(self, tmp_path):
        dat = tmp_path / "test.dat"
        dat.write_text('{"_version": "2.0.0", "_notes": []}')
        result = read_dat_file(dat)
        assert result["_version"] == "2.0.0"

    def test_read_gzip(self, tmp_path):
        dat = tmp_path / "test.dat"
        content = json.dumps({"version": "4.0.0", "colorNotes": []}).encode()
        dat.write_bytes(gzip.compress(content))
        result = read_dat_file(dat)
        assert result["version"] == "4.0.0"


class TestInfoParser:
    def test_parse_v2_info(self):
        info_data = json.loads((FIXTURES / "v2_map" / "Info.dat").read_text())
        metadata, difficulties = parse_info(info_data, source="test", source_id="v2")
        assert metadata.song_name == "Test Song V2"
        assert metadata.bpm == 120.0
        assert metadata.mapper_name == "Test Mapper"
        assert len(difficulties) == 2
        assert difficulties[0][0].difficulty == "Easy"
        assert difficulties[1][0].difficulty == "Expert"
        assert difficulties[0][1] == "Easy.dat"

    def test_parse_v4_info(self):
        info_data = json.loads((FIXTURES / "v4_map" / "Info.dat").read_text())
        metadata, difficulties = parse_info(info_data, source="test", source_id="v4")
        assert metadata.song_name == "Test Song V4"
        assert metadata.bpm == 150.0
        assert metadata.mapper_name == "TestMapperV4"
        assert len(difficulties) == 1
        assert difficulties[0][0].difficulty == "ExpertPlus"


class TestBeatmapParser:
    def test_parse_v2_map(self):
        results = parse_map_folder(FIXTURES / "v2_map", source="test", source_id="v2")
        assert len(results) == 2  # Easy + Expert

        easy = next(r for r in results if r.difficulty_info.difficulty == "Easy")
        assert easy.difficulty_info.note_count == 4  # 4 notes (1 bomb excluded)
        assert easy.difficulty_info.bomb_count == 1
        assert easy.difficulty_info.obstacle_count == 2
        assert easy.notes[0].beat == 4.0
        assert easy.notes[0].color == 0  # red

        expert = next(r for r in results if r.difficulty_info.difficulty == "Expert")
        assert expert.difficulty_info.note_count == 8

    def test_parse_v3_map(self):
        results = parse_map_folder(FIXTURES / "v3_map", source="test", source_id="v3")
        assert len(results) == 1

        hard = results[0]
        assert hard.difficulty_info.note_count == 4
        assert hard.difficulty_info.bomb_count == 2
        assert hard.difficulty_info.obstacle_count == 1
        # v3 supports angle offset
        note_with_angle = next(n for n in hard.notes if n.angle_offset != 0)
        assert note_with_angle.angle_offset == 15

    def test_parse_v4_map(self):
        results = parse_map_folder(FIXTURES / "v4_map", source="test", source_id="v4")
        assert len(results) == 1

        ep = results[0]
        assert ep.difficulty_info.note_count == 4
        assert ep.difficulty_info.bomb_count == 1
        assert ep.difficulty_info.obstacle_count == 1
        # v4 index deref: note at i=0 and i=0 share same data
        notes_at_idx0 = [n for n in ep.notes if n.x == 1 and n.y == 0 and n.color == 0]
        assert len(notes_at_idx0) == 2  # Two notes reference colorNotesData[0]

    def test_cross_version_v2_info_v3_beatmap(self):
        """Test the real-world case: v2 Info.dat + v3 difficulty files (like Magic)."""
        results = parse_map_folder(FIXTURES / "v3_map", source="test", source_id="v3")
        # v3_map fixture has v2 Info.dat + v3 Hard.dat
        assert len(results) == 1
        assert results[0].metadata.bpm == 140.0
        assert results[0].difficulty_info.note_count == 4

    def test_missing_info_dat(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_map_folder(tmp_path)

    def test_missing_difficulty_file(self, tmp_path):
        info = {
            "_version": "2.0.0",
            "_songName": "Missing",
            "_songAuthorName": "",
            "_levelAuthorName": "",
            "_beatsPerMinute": 120.0,
            "_difficultyBeatmapSets": [{
                "_beatmapCharacteristicName": "Standard",
                "_difficultyBeatmaps": [{
                    "_difficulty": "Easy",
                    "_difficultyRank": 1,
                    "_beatmapFilename": "NonExistent.dat",
                    "_noteJumpMovementSpeed": 10.0,
                    "_noteJumpStartBeatOffset": 0.0,
                }],
            }],
        }
        (tmp_path / "Info.dat").write_text(json.dumps(info))
        results = parse_map_folder(tmp_path)
        assert results == []  # Missing file is skipped, not an error
