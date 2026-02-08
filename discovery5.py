"""Examine full beatmap JSON structure to understand v4 format with indices."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
import UnityPy
import gzip
import json

levels_dir = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Beat Saber\Beat Saber_Data\StreamingAssets\BeatmapLevelsData")


def get_raw_bytes(raw):
    if isinstance(raw, bytes):
        return raw
    elif isinstance(raw, memoryview):
        return bytes(raw)
    elif isinstance(raw, str):
        return raw.encode("utf-8", errors="surrogateescape")
    else:
        return bytes(raw)


# Load one level and dump the full beatmap JSON for the Easy difficulty
level_file = levels_dir / "100bills"
env = UnityPy.load(str(level_file))

for obj in env.objects:
    if obj.type.name == "TextAsset":
        d = obj.parse_as_object()
        if d.m_Name == "100BillsEasy.beatmap.gz":
            raw = get_raw_bytes(d.m_Script)
            decompressed = gzip.decompress(raw)
            data = json.loads(decompressed)
            print("=== 100BillsEasy.beatmap.gz (full JSON) ===")
            print(json.dumps(data, indent=2))
            print()

        if d.m_Name == "100Bills.audio.gz":
            raw = get_raw_bytes(d.m_Script)
            decompressed = gzip.decompress(raw)
            data = json.loads(decompressed)
            print("=== 100Bills.audio.gz (full JSON) ===")
            print(json.dumps(data, indent=2))
            print()

# Also dump the MonoBehaviour to see the full difficulty set structure
for obj in env.objects:
    if obj.type.name == "MonoBehaviour":
        tree = obj.parse_as_dict()
        name = tree.get("m_Name", "")
        if "BeatmapLevelData" in name:
            print("=== MonoBehaviour: BeatmapLevelData (full) ===")
            # Print relevant fields
            print(f"version: {tree.get('_version')}")
            sets = tree.get("_difficultyBeatmapSets", [])
            for s in sets:
                char_name = s.get("_beatmapCharacteristicSerializedName", "?")
                print(f"\nCharacteristic: {char_name}")
                for d in s.get("_difficultyBeatmaps", []):
                    diff = d.get("_difficulty", "?")
                    print(f"  Difficulty {diff}:")
                    for k, v in d.items():
                        print(f"    {k}: {v}")
