"""Parse Info.dat files from both v2 and v4 Beat Saber formats."""

from beat_weaver.schemas.normalized import SongMetadata, DifficultyInfo

DIFFICULTY_RANK_MAP = {
    "Easy": 1,
    "Normal": 3,
    "Hard": 5,
    "Expert": 7,
    "ExpertPlus": 9,
}


def parse_info(
    info_data: dict, source: str = "", source_id: str = ""
) -> tuple[SongMetadata, list[tuple[DifficultyInfo, str]]]:
    """Parse Info.dat content.

    Returns (SongMetadata, list of (DifficultyInfo, beatmap_filename) pairs).
    """
    if "_version" in info_data or "_songName" in info_data:
        return _parse_v2(info_data, source, source_id)
    return _parse_v4(info_data, source, source_id)


def _parse_v2(
    data: dict, source: str, source_id: str
) -> tuple[SongMetadata, list[tuple[DifficultyInfo, str]]]:
    mapper = data.get("_levelAuthorName", "")

    metadata = SongMetadata(
        source=source,
        source_id=source_id,
        hash="",
        song_name=data.get("_songName", ""),
        song_sub_name=data.get("_songSubName", ""),
        song_author=data.get("_songAuthorName", ""),
        mapper_name=mapper,
        bpm=float(data.get("_beatsPerMinute", 0.0)),
    )

    difficulties: list[tuple[DifficultyInfo, str]] = []

    for bset in data.get("_difficultyBeatmapSets", []):
        characteristic = bset.get("_beatmapCharacteristicName", "Standard")

        for bmap in bset.get("_difficultyBeatmaps", []):
            difficulty = bmap.get("_difficulty", "")
            rank = bmap.get("_difficultyRank", DIFFICULTY_RANK_MAP.get(difficulty, 0))
            filename = bmap.get("_beatmapFilename", "")

            info = DifficultyInfo(
                characteristic=characteristic,
                difficulty=difficulty,
                difficulty_rank=rank,
                note_jump_speed=float(bmap.get("_noteJumpMovementSpeed", 0.0)),
                note_jump_offset=float(bmap.get("_noteJumpStartBeatOffset", 0.0)),
                note_count=0,
                bomb_count=0,
                obstacle_count=0,
            )
            difficulties.append((info, filename))

    return metadata, difficulties


def _parse_v4(
    data: dict, source: str, source_id: str
) -> tuple[SongMetadata, list[tuple[DifficultyInfo, str]]]:
    song = data.get("song", {})
    audio = data.get("audio", {})

    metadata = SongMetadata(
        source=source,
        source_id=source_id,
        hash="",
        song_name=song.get("title", ""),
        song_sub_name=song.get("subTitle", ""),
        song_author=song.get("author", ""),
        mapper_name="",
        bpm=float(audio.get("bpm", 0.0)),
    )

    difficulties: list[tuple[DifficultyInfo, str]] = []

    for bmap in data.get("difficultyBeatmaps", []):
        difficulty = bmap.get("difficulty", "")
        rank = DIFFICULTY_RANK_MAP.get(difficulty, 0)
        filename = bmap.get("beatmapDataFilename", "")

        # Extract mapper name from first entry if available
        authors = bmap.get("beatmapAuthors", {})
        mappers = authors.get("mappers", [])
        if mappers and not metadata.mapper_name:
            metadata.mapper_name = mappers[0]

        info = DifficultyInfo(
            characteristic=bmap.get("characteristic", "Standard"),
            difficulty=difficulty,
            difficulty_rank=rank,
            note_jump_speed=float(bmap.get("noteJumpMovementSpeed", 0.0)),
            note_jump_offset=float(bmap.get("noteJumpStartBeatOffset", 0.0)),
            note_count=0,
            bomb_count=0,
            obstacle_count=0,
        )
        difficulties.append((info, filename))

    return metadata, difficulties
