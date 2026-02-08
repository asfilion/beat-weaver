"""Normalized Beat Saber map data format.

Dataclasses that represent a unified schema across all Beat Saber map versions
(v2, v3, v4). Parsers convert version-specific formats into these structures
for downstream ML pipeline consumption.
"""

from dataclasses import dataclass, field


@dataclass
class Note:
    """A single color note (red or blue saber)."""

    beat: float
    time_seconds: float  # computed: beat * 60.0 / bpm
    x: int  # column 0-3
    y: int  # row 0-2
    color: int  # 0=Red/Left, 1=Blue/Right
    cut_direction: int
    # cut_direction values:
    #   0=Up, 1=Down, 2=Left, 3=Right,
    #   4=UpLeft, 5=UpRight, 6=DownLeft, 7=DownRight,
    #   8=Any
    angle_offset: int = 0  # rotation degrees; only non-zero in v3/v4


@dataclass
class Bomb:
    """A bomb note that the player must avoid hitting."""

    beat: float
    time_seconds: float
    x: int  # column 0-3
    y: int  # row 0-2


@dataclass
class Obstacle:
    """A wall/obstacle the player must dodge."""

    beat: float
    time_seconds: float
    duration_beats: float
    x: int
    y: int
    width: int
    height: int  # v2: derived from type (0->5, 1->3); v3/v4: explicit


@dataclass
class DifficultyInfo:
    """Metadata for one difficulty level of a beatmap."""

    characteristic: str  # "Standard", "OneSaber", "NoArrows", "360Degree", "90Degree"
    difficulty: str  # "Easy", "Normal", "Hard", "Expert", "ExpertPlus"
    difficulty_rank: int  # 1, 3, 5, 7, 9
    note_jump_speed: float
    note_jump_offset: float
    note_count: int = 0  # filled after parsing
    bomb_count: int = 0
    obstacle_count: int = 0
    nps: float | None = None  # notes per second


@dataclass
class SongMetadata:
    """Song-level metadata from the map info file and external sources."""

    source: str  # "beatsaver", "local_custom", "official"
    source_id: str  # BeatSaver ID, folder name, or bundle name
    hash: str = ""  # content hash for dedup
    song_name: str = ""
    song_sub_name: str = ""
    song_author: str = ""
    mapper_name: str = ""
    bpm: float = 0.0
    song_duration_seconds: float | None = None
    upvotes: int | None = None  # BeatSaver only
    downvotes: int | None = None
    score: float | None = None  # 0.0-1.0 rating


@dataclass
class NormalizedBeatmap:
    """Complete parsed result for one difficulty of a Beat Saber map."""

    metadata: SongMetadata
    difficulty_info: DifficultyInfo
    notes: list[Note] = field(default_factory=list)  # sorted by beat
    bombs: list[Bomb] = field(default_factory=list)  # sorted by beat
    obstacles: list[Obstacle] = field(default_factory=list)  # sorted by beat
