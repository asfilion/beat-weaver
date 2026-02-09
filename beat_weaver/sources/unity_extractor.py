"""Extract official Beat Saber beatmaps from Unity asset bundles.

Beat Saber stores official maps in two locations:

1. **Pack bundles** (StandaloneWindows64/<pack>_pack_assets_all_*.bundle):
   MonoBehaviour objects containing song metadata -- level ID, song name,
   artist, BPM, duration, note-jump speeds per difficulty, etc.

2. **Level data bundles** (BeatmapLevelsData/<levelID>):
   UnityFS bundles (no .bundle extension) containing:
   - One MonoBehaviour ("BeatmapLevelDataSO") that maps characteristics
     and difficulty integers to TextAsset path-IDs.
   - Gzipped TextAssets: <Name><Diff>.beatmap.gz (v4 beatmap JSON),
     <Name>.audio.gz (BPM/sample info), and <Name>.lightshow.gz.
   - One AudioClip (the song audio, stored as a resource reference).

The extractor reads both locations and writes each level as a standard
map folder that ``parse_map_folder()`` can consume:

    <output_dir>/<levelID>/
        Info.dat          -- synthesised v2-style info
        <Difficulty>.dat  -- gzipped v4 beatmap JSON per difficulty
"""

import gzip
import json
import logging
from pathlib import Path
from typing import Any

import UnityPy

from beat_weaver.parsers.dat_reader import GZIP_MAGIC

logger = logging.getLogger(__name__)

# Difficulty integer -> human-readable name used by the rest of our pipeline.
DIFFICULTY_INT_TO_NAME = {
    0: "Easy",
    1: "Normal",
    2: "Hard",
    3: "Expert",
    4: "ExpertPlus",
}

# Default locations within a Steam installation.
DEFAULT_BEAT_SABER_PATH = Path(
    r"C:\Program Files (x86)\Steam\steamapps\common\Beat Saber"
)
BUNDLES_SUBPATH = (
    Path("Beat Saber_Data")
    / "StreamingAssets"
    / "aa"
    / "StandaloneWindows64"
)
LEVELS_DATA_SUBPATH = (
    Path("Beat Saber_Data") / "StreamingAssets" / "BeatmapLevelsData"
)
DLC_LEVELS_SUBPATH = Path("DLC") / "Levels"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_raw_bytes(script_field: Any) -> bytes:
    """Coerce a UnityPy ``m_Script`` value to plain ``bytes``.

    Depending on the UnityPy version and compression, the field may arrive
    as ``bytes``, ``memoryview``, or a (possibly mangled) ``str``.
    """
    if isinstance(script_field, bytes):
        return script_field
    if isinstance(script_field, memoryview):
        return bytes(script_field)
    if isinstance(script_field, str):
        # UnityPy sometimes decodes binary data as latin-1 / surrogateescape.
        return script_field.encode("utf-8", errors="surrogateescape")
    return bytes(script_field)


def _decompress_if_gzip(data: bytes) -> bytes:
    """Return decompressed data if *data* starts with the gzip magic bytes."""
    if data[:2] == GZIP_MAGIC:
        return gzip.decompress(data)
    return data


# ---------------------------------------------------------------------------
# Pack-bundle metadata reader
# ---------------------------------------------------------------------------

def _read_pack_metadata(bundles_dir: Path) -> dict[str, dict]:
    """Scan all ``*_pack_assets_all_*.bundle`` files and return per-level metadata.

    Returns:
        ``{level_id: {song_name, song_author, bpm, duration, offset, pack,
        difficulties: [{characteristic, difficulty, njs, njo}]}}``
    """
    metadata: dict[str, dict] = {}
    pack_bundles = sorted(bundles_dir.glob("*_pack_assets_all_*.bundle"))

    for pb in pack_bundles:
        pack_name = pb.name.split("_pack_assets_all_")[0]
        try:
            env = UnityPy.load(str(pb))
        except Exception:
            logger.warning("Failed to load pack bundle: %s", pb.name)
            continue

        for obj in env.objects:
            if obj.type.name != "MonoBehaviour":
                continue
            try:
                tree = obj.parse_as_dict()
            except Exception:
                continue

            name = tree.get("m_Name", "")
            level_id = tree.get("_levelID")
            if not level_id:
                continue
            # Skip non-level MonoBehaviours (pack definitions, promo, products)
            if "BeatmapLevel" not in name:
                continue
            if any(k in name for k in ("Pack", "Product", "Leaderboard", "Promo", "Collection")):
                continue

            # Collect per-difficulty info from the preview sets.
            diff_list: list[dict] = []
            for bset in tree.get("_previewDifficultyBeatmapSets", []):
                for entry in bset.get("_previewDifficultyBeatmaps", []):
                    diff_int = entry.get("_difficulty", -1)
                    diff_name = DIFFICULTY_INT_TO_NAME.get(diff_int)
                    if diff_name is None:
                        continue
                    diff_list.append({
                        "difficulty": diff_name,
                        "njs": entry.get("_noteJumpMovementSpeed", 0.0),
                        "njo": entry.get("_noteJumpStartBeatOffset", 0.0),
                    })

            metadata[level_id] = {
                "pack": pack_name,
                "song_name": tree.get("_songName", ""),
                "song_sub_name": tree.get("_songSubName", ""),
                "song_author": tree.get("_songAuthorName", ""),
                "level_author": tree.get("_levelAuthorName", ""),
                "bpm": float(tree.get("_beatsPerMinute", 0.0)),
                "duration": float(tree.get("_songDuration", 0.0)),
                "offset": float(tree.get("_songTimeOffset", 0.0)),
                "difficulties": diff_list,
            }

    return metadata


# ---------------------------------------------------------------------------
# Level-data bundle reader
# ---------------------------------------------------------------------------

def _extract_level_bundle(
    bundle_path: Path,
    pack_meta: dict | None,
    output_dir: Path,
) -> Path | None:
    """Extract a single level data bundle into a map folder.

    Returns the output folder path, or ``None`` on failure.
    """
    level_id = bundle_path.name  # file has no extension

    try:
        env = UnityPy.load(str(bundle_path))
    except Exception:
        logger.warning("Failed to load level bundle: %s", bundle_path)
        return None

    # Build path-ID -> TextAsset mapping.
    text_assets: dict[int, tuple[str, bytes]] = {}
    for obj in env.objects:
        if obj.type.name == "TextAsset":
            parsed = obj.parse_as_object()
            raw = _get_raw_bytes(parsed.m_Script)
            text_assets[obj.path_id] = (parsed.m_Name, raw)

    # Read the BeatmapLevelDataSO MonoBehaviour to get the definitive
    # characteristic -> difficulty -> TextAsset mapping.
    difficulty_sets: list[dict] | None = None
    for obj in env.objects:
        if obj.type.name != "MonoBehaviour":
            continue
        try:
            tree = obj.parse_as_dict()
        except Exception:
            continue
        if "BeatmapLevelData" in tree.get("m_Name", ""):
            difficulty_sets = tree.get("_difficultyBeatmapSets", [])
            break

    if difficulty_sets is None:
        logger.warning("No BeatmapLevelDataSO found in %s", bundle_path)
        return None

    # Resolve each difficulty's beatmap TextAsset.
    # Structure: [{characteristic, difficulty_int, beatmap_bytes, lightshow_bytes}]
    resolved: list[dict] = []
    for dset in difficulty_sets:
        characteristic = dset.get("_beatmapCharacteristicSerializedName", "Standard")
        for entry in dset.get("_difficultyBeatmaps", []):
            diff_int = entry.get("_difficulty", -1)
            diff_name = DIFFICULTY_INT_TO_NAME.get(diff_int)
            if diff_name is None:
                continue

            bm_path_id = entry.get("_beatmapAsset", {}).get("m_PathID")
            if bm_path_id is None or bm_path_id not in text_assets:
                continue

            _, bm_raw = text_assets[bm_path_id]
            resolved.append({
                "characteristic": characteristic,
                "difficulty": diff_name,
                "beatmap_bytes": bm_raw,
            })

    if not resolved:
        logger.warning("No beatmap data resolved for %s", bundle_path)
        return None

    # Read audio metadata TextAsset (e.g. "100Bills.audio.gz") for BPM info.
    audio_meta: dict | None = None
    for _, (asset_name, raw) in text_assets.items():
        if asset_name.endswith(".audio.gz") or asset_name.endswith(".audio"):
            try:
                decompressed = _decompress_if_gzip(raw)
                audio_meta = json.loads(decompressed)
            except Exception:
                pass
            break

    # Determine BPM.  Pack metadata is authoritative; audio metadata is backup.
    bpm = 0.0
    if pack_meta:
        bpm = pack_meta.get("bpm", 0.0)
    if bpm == 0.0 and audio_meta:
        # Derive BPM from bpmData if available.
        bpm_data = audio_meta.get("bpmData", [])
        freq = audio_meta.get("songFrequency", 44100)
        samples = audio_meta.get("songSampleCount", 0)
        if bpm_data and samples > 0:
            # bpmData contains cumulative beat counts at sample positions.
            # The last entry's eb / (ei / freq) * 60 gives BPM.
            last = bpm_data[-1]
            duration_s = last["ei"] / freq
            if duration_s > 0:
                bpm = last["eb"] / duration_s * 60.0

    # Build per-difficulty NJS/NJO lookup from pack metadata.
    njs_lookup: dict[str, tuple[float, float]] = {}
    if pack_meta:
        for d in pack_meta.get("difficulties", []):
            njs_lookup[d["difficulty"]] = (d["njs"], d["njo"])

    # Write output folder.
    out_folder = output_dir / level_id
    out_folder.mkdir(parents=True, exist_ok=True)

    # Write each beatmap .dat file (kept gzip-compressed -- our dat_reader
    # auto-detects gzip).  Filename convention: <Characteristic><Difficulty>.dat
    beatmap_filenames: list[tuple[str, str, str]] = []
    for item in resolved:
        char = item["characteristic"]
        diff = item["difficulty"]
        # For "Standard" we omit the prefix to match custom-map conventions.
        if char == "Standard":
            filename = f"{diff}.dat"
        else:
            filename = f"{char}{diff}.dat"

        out_path = out_folder / filename
        raw_bytes = item["beatmap_bytes"]
        # Ensure the file is gzip-compressed for consistency.
        decompressed = _decompress_if_gzip(raw_bytes)
        compressed = gzip.compress(decompressed)
        out_path.write_bytes(compressed)

        beatmap_filenames.append((char, diff, filename))

    # Extract AudioClip (song audio) as WAV file.
    # UnityPy decodes AudioClips to WAV via the samples property.
    audio_filename = ""
    for obj in env.objects:
        if obj.type.name == "AudioClip":
            try:
                clip = obj.parse_as_object()
                for clip_name, clip_data in clip.samples.items():
                    audio_out = out_folder / "song.wav"
                    audio_out.write_bytes(clip_data)
                    audio_filename = "song.wav"
                    logger.debug(
                        "Extracted audio: %s (%d bytes)", level_id, len(clip_data),
                    )
                    break
            except Exception:
                logger.warning("Failed to extract AudioClip from %s", bundle_path)
            break

    # Synthesise a v2-style Info.dat so parse_map_folder() can read the folder.
    info = _build_info_dat(
        level_id, pack_meta, bpm, njs_lookup, beatmap_filenames, audio_filename,
    )
    info_path = out_folder / "Info.dat"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    return out_folder


def _build_info_dat(
    level_id: str,
    pack_meta: dict | None,
    bpm: float,
    njs_lookup: dict[str, tuple[float, float]],
    beatmap_filenames: list[tuple[str, str, str]],
    audio_filename: str = "",
) -> dict:
    """Synthesise a v2-style Info.dat from extracted metadata."""
    song_name = ""
    song_sub = ""
    song_author = ""
    level_author = ""

    if pack_meta:
        song_name = pack_meta.get("song_name", "")
        song_sub = pack_meta.get("song_sub_name", "")
        song_author = pack_meta.get("song_author", "")
        level_author = pack_meta.get("level_author", "")

    # Group difficulties by characteristic.
    char_groups: dict[str, list[dict]] = {}
    for char, diff, filename in beatmap_filenames:
        njs, njo = njs_lookup.get(diff, (0.0, 0.0))
        entry = {
            "_difficulty": diff,
            "_difficultyRank": {
                "Easy": 1, "Normal": 3, "Hard": 5, "Expert": 7, "ExpertPlus": 9,
            }.get(diff, 0),
            "_beatmapFilename": filename,
            "_noteJumpMovementSpeed": njs,
            "_noteJumpStartBeatOffset": njo,
        }
        char_groups.setdefault(char, []).append(entry)

    diff_sets = []
    for char, entries in char_groups.items():
        diff_sets.append({
            "_beatmapCharacteristicName": char,
            "_difficultyBeatmaps": entries,
        })

    return {
        "_version": "2.0.0",
        "_songName": song_name,
        "_songSubName": song_sub,
        "_songAuthorName": song_author,
        "_levelAuthorName": level_author,
        "_beatsPerMinute": bpm,
        "_songTimeOffset": pack_meta.get("offset", 0.0) if pack_meta else 0.0,
        "_songFilename": audio_filename,
        "_difficultyBeatmapSets": diff_sets,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_bundles(
    bundles_dir: Path | None = None,
    beat_saber_path: Path = DEFAULT_BEAT_SABER_PATH,
) -> dict:
    """Enumerate all pack bundles and level-data bundles and return a summary.

    Args:
        bundles_dir: Path to ``StandaloneWindows64`` directory.  Derived
            from *beat_saber_path* when ``None``.
        beat_saber_path: Root Beat Saber installation directory.

    Returns:
        Dict with keys ``pack_bundles`` (list of names), ``level_bundles``
        (list of names), ``levels`` (dict of per-level metadata keyed by
        level ID), and ``level_bundle_dir`` (Path).
    """
    if bundles_dir is None:
        bundles_dir = beat_saber_path / BUNDLES_SUBPATH
    levels_dir = beat_saber_path / LEVELS_DATA_SUBPATH

    pack_bundle_names = sorted(
        b.name for b in bundles_dir.glob("*_pack_assets_all_*.bundle")
    )

    level_bundle_names: list[str] = []
    if levels_dir.exists():
        level_bundle_names = sorted(f.name for f in levels_dir.iterdir() if f.is_file())

    metadata = _read_pack_metadata(bundles_dir)

    return {
        "pack_bundles": pack_bundle_names,
        "level_bundles": level_bundle_names,
        "levels": metadata,
        "level_bundle_dir": str(levels_dir),
        "bundles_dir": str(bundles_dir),
    }


def _collect_level_bundles(
    beat_saber_path: Path,
) -> list[Path]:
    """Collect level bundle files from both BeatmapLevelsData and DLC/Levels."""
    bundles: list[Path] = []

    # Standard location: BeatmapLevelsData/<levelID>
    levels_dir = beat_saber_path / LEVELS_DATA_SUBPATH
    if levels_dir.exists():
        for f in sorted(levels_dir.iterdir()):
            if f.is_file():
                bundles.append(f)

    # DLC location: DLC/Levels/<LevelName>/<bundlefile>
    dlc_dir = beat_saber_path / DLC_LEVELS_SUBPATH
    if dlc_dir.exists():
        for level_folder in sorted(dlc_dir.iterdir()):
            if not level_folder.is_dir():
                continue
            for f in level_folder.iterdir():
                if f.is_file():
                    bundles.append(f)

    return bundles


def extract_official_maps(
    bundles_dir: Path | None = None,
    output_dir: Path = Path("data/official_extracted"),
    beat_saber_path: Path = DEFAULT_BEAT_SABER_PATH,
) -> list[Path]:
    """Extract beatmap data from Unity bundles into regular map folders.

    Each level becomes a folder with an ``Info.dat`` and per-difficulty
    ``.dat`` files (gzip-compressed v4 JSON), consumable by
    ``parse_map_folder()``.

    Scans both the standard ``BeatmapLevelsData`` directory and the
    ``DLC/Levels`` directory for level bundles.

    Args:
        bundles_dir: Path to ``StandaloneWindows64`` directory.  Derived
            from *beat_saber_path* when ``None``.
        output_dir: Where to write extracted map folders.
        beat_saber_path: Root Beat Saber installation directory.

    Returns:
        List of successfully extracted folder paths.
    """
    if bundles_dir is None:
        bundles_dir = beat_saber_path / BUNDLES_SUBPATH

    level_bundles = _collect_level_bundles(beat_saber_path)
    if not level_bundles:
        logger.error("No level bundles found in %s", beat_saber_path)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read pack-level metadata first.
    pack_metadata = _read_pack_metadata(bundles_dir)
    logger.info(
        "Read metadata for %d levels from %d pack bundles",
        len(pack_metadata),
        len(list(bundles_dir.glob("*_pack_assets_all_*.bundle"))),
    )

    # Pre-build case-insensitive metadata lookup
    meta_lookup: dict[str, dict] = {
        pid.lower(): pmeta for pid, pmeta in pack_metadata.items()
    }

    extracted: list[Path] = []

    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor() as executor:
        futures = {}
        for bundle_path in level_bundles:
            level_id = bundle_path.name
            meta = meta_lookup.get(level_id.lower())
            futures[executor.submit(
                _extract_level_bundle, bundle_path, meta, output_dir,
            )] = level_id

        for future in as_completed(futures):
            level_id = futures[future]
            try:
                result = future.result()
                if result is not None:
                    extracted.append(result)
                    logger.debug("Extracted: %s", level_id)
                else:
                    logger.warning("Skipped: %s", level_id)
            except Exception:
                logger.warning("Failed: %s", level_id, exc_info=True)

    logger.info(
        "Extracted %d / %d level bundles (BeatmapLevelsData + DLC)",
        len(extracted), len(level_bundles),
    )
    return extracted
