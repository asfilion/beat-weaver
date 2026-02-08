"""Deep dive into beatmap level bundle structure."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
import UnityPy
import gzip

levels_dir = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Beat Saber\Beat Saber_Data\StreamingAssets\BeatmapLevelsData")

# Explore one level bundle thoroughly
level_file = levels_dir / "100bills"
env = UnityPy.load(str(level_file))


def get_raw_bytes(raw):
    """Convert m_Script to bytes regardless of type."""
    if isinstance(raw, bytes):
        return raw
    elif isinstance(raw, memoryview):
        return bytes(raw)
    elif isinstance(raw, str):
        return raw.encode("utf-8", errors="surrogateescape")
    else:
        return bytes(raw)


# List all objects
for obj in env.objects:
    t = obj.type.name
    if t == "TextAsset":
        d = obj.parse_as_object()
        raw = d.m_Script
        raw_bytes = get_raw_bytes(raw)
        is_gz = raw_bytes[:2] == b'\x1f\x8b'
        print(f"TextAsset: {d.m_Name} ({len(raw_bytes)} bytes, gzip={is_gz})")
        if is_gz:
            try:
                decompressed = gzip.decompress(raw_bytes)
                preview = decompressed[:500].decode("utf-8", errors="replace")
                print(f"  Decompressed ({len(decompressed)} bytes): {preview}")
            except Exception as e:
                print(f"  Decompress failed: {e}")
        else:
            preview = raw_bytes[:500].decode("utf-8", errors="replace")
            print(f"  Raw preview: {preview}")
    elif t == "MonoBehaviour":
        try:
            tree = obj.parse_as_dict()
            name = tree.get("m_Name", "<unnamed>")
            keys = list(tree.keys())
            print(f"MonoBehaviour: {name} keys={keys}")
            for k, v in tree.items():
                if not k.startswith("m_") or k == "m_Name":
                    if not isinstance(v, dict) or "m_PathID" not in v:
                        val_str = repr(v)
                        if len(val_str) > 200:
                            val_str = val_str[:200] + "..."
                        print(f"  {k}: {val_str}")
        except Exception as e:
            print(f"MonoBehaviour: parse failed: {e}")
    elif t == "AudioClip":
        d = obj.parse_as_object()
        print(f"AudioClip: {d.m_Name} (channels={d.m_Channels}, freq={d.m_Frequency}, length={d.m_Length})")
    elif t == "AssetBundle":
        d = obj.parse_as_object()
        print(f"AssetBundle: {d.m_Name}")
    elif t == "MonoScript":
        d = obj.parse_as_object()
        print(f"MonoScript: {d.m_Name} (class={d.m_ClassName}, namespace={d.m_Namespace})")
