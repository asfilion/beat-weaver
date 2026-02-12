"""Export token sequences to playable v2 Beat Saber map folders."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from beat_weaver.model.tokenizer import decode_tokens
from beat_weaver.schemas.normalized import Note

# NJS lookup by difficulty
_NJS_TABLE = {
    "Easy": 10,
    "Normal": 10,
    "Hard": 12,
    "Expert": 16,
    "ExpertPlus": 18,
}

# Difficulty rank for Info.dat
_DIFFICULTY_RANK = {
    "Easy": 1,
    "Normal": 3,
    "Hard": 5,
    "Expert": 7,
    "ExpertPlus": 9,
}


def _build_info_dat(
    song_name: str,
    bpm: float,
    difficulty: str,
    audio_filename: str = "song.ogg",
) -> dict:
    """Build a v2 Info.dat structure."""
    njs = _NJS_TABLE.get(difficulty, 16)
    rank = _DIFFICULTY_RANK.get(difficulty, 7)

    return {
        "_version": "2.0.0",
        "_songName": song_name,
        "_songSubName": "",
        "_songAuthorName": "",
        "_levelAuthorName": "BeatWeaver AI",
        "_beatsPerMinute": bpm,
        "_songTimeOffset": 0,
        "_shuffle": 0,
        "_shufflePeriod": 0.5,
        "_previewStartTime": 12,
        "_previewDuration": 10,
        "_songFilename": audio_filename,
        "_coverImageFilename": "",
        "_environmentName": "DefaultEnvironment",
        "_difficultyBeatmapSets": [
            {
                "_beatmapCharacteristicName": "Standard",
                "_difficultyBeatmaps": [
                    {
                        "_difficulty": difficulty,
                        "_difficultyRank": rank,
                        "_beatmapFilename": f"{difficulty}.dat",
                        "_noteJumpMovementSpeed": njs,
                        "_noteJumpStartBeatOffset": 0,
                    }
                ],
            }
        ],
    }


def _build_difficulty_dat(notes: list, version: str = "2.0.0") -> dict:
    """Build a v2 difficulty .dat structure from decoded notes."""
    v2_notes = []
    for note in notes:
        v2_notes.append({
            "_time": note.beat,
            "_lineIndex": note.x,
            "_lineLayer": note.y,
            "_type": note.color,
            "_cutDirection": note.cut_direction,
        })

    return {
        "_version": version,
        "_notes": sorted(v2_notes, key=lambda n: n["_time"]),
        "_obstacles": [],
        "_events": [],
    }


def export_map(
    token_ids: list[int],
    bpm: float,
    song_name: str,
    audio_path: Path,
    output_dir: Path,
    difficulty: str = "Expert",
) -> Path:
    """Export a token sequence to a playable v2 Beat Saber map folder.

    Args:
        token_ids: Generated token sequence.
        bpm: Song BPM.
        song_name: Display name for the song.
        audio_path: Path to the audio file.
        output_dir: Where to create the map folder.
        difficulty: Difficulty name.

    Returns:
        Path to the created map folder.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Decode tokens to notes
    notes = decode_tokens(token_ids, bpm)

    # Determine audio filename — copy to folder as song.ogg/.egg
    audio_path = Path(audio_path)
    audio_filename = f"song{audio_path.suffix}"
    dest_audio = output_dir / audio_filename
    shutil.copy2(audio_path, dest_audio)

    # Write Info.dat
    info = _build_info_dat(song_name, bpm, difficulty, audio_filename)
    (output_dir / "Info.dat").write_text(json.dumps(info, indent=2))

    # Write difficulty file
    diff_dat = _build_difficulty_dat(notes)
    (output_dir / f"{difficulty}.dat").write_text(json.dumps(diff_dat, indent=2))

    return output_dir


def export_notes(
    notes: list[Note],
    bpm: float,
    song_name: str,
    audio_path: Path,
    output_dir: Path,
    difficulty: str = "Expert",
) -> Path:
    """Export a note list to a playable v2 Beat Saber map folder.

    Unlike export_map() which takes token IDs, this takes pre-decoded notes
    (e.g. from windowed generation where notes have already been merged).

    Args:
        notes: List of Note objects.
        bpm: Song BPM.
        song_name: Display name for the song.
        audio_path: Path to the audio file.
        output_dir: Where to create the map folder.
        difficulty: Difficulty name.

    Returns:
        Path to the created map folder.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_path = Path(audio_path)
    audio_filename = f"song{audio_path.suffix}"
    dest_audio = output_dir / audio_filename
    shutil.copy2(audio_path, dest_audio)

    info = _build_info_dat(song_name, bpm, difficulty, audio_filename)
    (output_dir / "Info.dat").write_text(json.dumps(info, indent=2))

    diff_dat = _build_difficulty_dat(notes)
    (output_dir / f"{difficulty}.dat").write_text(json.dumps(diff_dat, indent=2))

    return output_dir
