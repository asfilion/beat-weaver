"""Check v4 colorNotesData defaults - what values can be missing?"""
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


# Look at colorNotesData entries to see which keys are present/missing
level_file = levels_dir / "100bills"
env = UnityPy.load(str(level_file))

for obj in env.objects:
    if obj.type.name == "TextAsset":
        d = obj.parse_as_object()
        if d.m_Name == "100BillsExpertPlus.beatmap.gz":
            raw = get_raw_bytes(d.m_Script)
            decompressed = gzip.decompress(raw)
            data = json.loads(decompressed)

            print("=== colorNotesData ===")
            for i, entry in enumerate(data.get("colorNotesData", [])):
                print(f"  [{i}]: {entry}")

            print(f"\n=== colorNotes (first 5) ===")
            for entry in data.get("colorNotes", [])[:5]:
                print(f"  {entry}")

            print(f"\n=== colorNotes entries missing 'i' key ===")
            missing_i = [e for e in data.get("colorNotes", []) if "i" not in e]
            print(f"  Count: {len(missing_i)}")
            if missing_i:
                print(f"  Example: {missing_i[0]}")

            print(f"\n=== bombNotesData ===")
            for i, entry in enumerate(data.get("bombNotesData", [])):
                print(f"  [{i}]: {entry}")

            print(f"\n=== obstaclesData ===")
            for i, entry in enumerate(data.get("obstaclesData", [])):
                print(f"  [{i}]: {entry}")

            break
