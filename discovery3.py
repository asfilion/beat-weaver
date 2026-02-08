"""Explore BeatmapLevelsData directory."""
from pathlib import Path
import UnityPy

levels_dir = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Beat Saber\Beat Saber_Data\StreamingAssets\BeatmapLevelsData")

level_files = sorted(levels_dir.iterdir())
print(f"BeatmapLevelsData: {len(level_files)} files")
for f in level_files[:10]:
    size = f.stat().st_size
    print(f"  {f.name} ({size / 1024:.1f} KB)")

# Try loading first few as UnityPy bundles, then as raw data
print("\n=== Trying to load first file ===")
first = level_files[0]
print(f"File: {first.name}")

# Check first bytes
with open(first, "rb") as fh:
    header = fh.read(200)
    print(f"First 200 bytes (hex): {header[:50].hex()}")
    print(f"First 200 bytes (ascii): {header[:100]}")

# Try UnityPy
try:
    env = UnityPy.load(str(first))
    types = {}
    for obj in env.objects:
        types[obj.type.name] = types.get(obj.type.name, 0) + 1
    print(f"UnityPy types: {types}")

    for obj in env.objects:
        if obj.type.name == "TextAsset":
            d = obj.parse_as_object()
            raw = d.m_Script
            if isinstance(raw, bytes):
                print(f"  TextAsset: {d.m_Name} ({len(raw)} bytes)")
                print(f"    Preview: {raw[:200]}")
            else:
                print(f"  TextAsset: {d.m_Name} ({len(raw)} chars)")
                print(f"    Preview: {raw[:200]}")
except Exception as e:
    print(f"UnityPy failed: {e}")

# Check if it's just raw JSON or other format
try:
    import gzip
    with open(first, "rb") as fh:
        data = fh.read()
    if data[:2] == b'\x1f\x8b':
        print("File is gzip compressed!")
        decompressed = gzip.decompress(data)
        print(f"Decompressed: {decompressed[:200]}")
    else:
        # Try as UTF-8
        text = data.decode("utf-8", errors="replace")
        if "{" in text[:100]:
            print(f"Looks like text/JSON: {text[:200]}")
except Exception as e:
    print(f"Raw read failed: {e}")
