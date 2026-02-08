"""Detect Beat Saber map schema versions from JSON data.

Supports v2, v3, and v4 info and beatmap formats. Detection is based on
the presence of version-specific keys in the JSON structure.
"""


def detect_info_version(info_data: dict) -> str:
    """Detect the schema version of a Beat Saber info.dat file.

    Args:
        info_data: Parsed JSON dict from info.dat / Info.dat.

    Returns:
        Version string: "2", "3", or "4".
    """
    if "_version" in info_data:
        return "2"

    version_str = info_data.get("version", "")
    if version_str:
        try:
            major = int(version_str.split(".")[0])
            if major >= 4:
                return "4"
            else:
                return "3"
        except (ValueError, IndexError):
            pass

    return "2"


def detect_beatmap_version(beatmap_data: dict) -> str:
    """Detect the schema version of a Beat Saber beatmap file.

    Args:
        beatmap_data: Parsed JSON dict from a difficulty beatmap file.

    Returns:
        Version string: "2", "3", or "4".

    Raises:
        ValueError: If the beatmap format cannot be identified.
    """
    if "_version" in beatmap_data:
        return "2"

    if "colorNotesData" in beatmap_data:
        return "4"

    if "colorNotes" in beatmap_data:
        return "3"

    if "_notes" in beatmap_data:
        return "2"

    raise ValueError(
        "Unable to detect beatmap version: no recognized keys found. "
        "Expected one of: _version, colorNotesData, colorNotes, _notes."
    )
