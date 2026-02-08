"""Explore pack bundle metadata and correlation with level bundles."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
import UnityPy
import json

bundles_dir = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Beat Saber\Beat Saber_Data\StreamingAssets\aa\StandaloneWindows64")

# Explore one pack bundle fully
bundle_path = list(bundles_dir.glob("ostvol1_pack_assets_all_*.bundle"))[0]
env = UnityPy.load(str(bundle_path))

for obj in env.objects:
    if obj.type.name == "MonoBehaviour":
        try:
            tree = obj.parse_as_dict()
            name = tree.get("m_Name", "")
            if "BeatmapLevel" in name and "Pack" not in name and "Product" not in name and "Leaderboard" not in name and "Promo" not in name:
                print(f"=== {name} ===")
                print(f"  levelID: {tree.get('_levelID')}")
                print(f"  songName: {tree.get('_songName')}")
                print(f"  songAuthorName: {tree.get('_songAuthorName')}")
                print(f"  bpm: {tree.get('_beatsPerMinute')}")
                print(f"  songDuration: {tree.get('_songDuration')}")
                print(f"  songTimeOffset: {tree.get('_songTimeOffset')}")
                sets = tree.get("_previewDifficultyBeatmapSets", [])
                for s in sets:
                    for d in s.get("_previewDifficultyBeatmaps", []):
                        diff = d.get("_difficulty", "?")
                        print(f"    Difficulty {diff}: noteJumpSpeed={d.get('_noteJumpMovementSpeed')}, noteJumpOffset={d.get('_noteJumpStartBeatOffset')}")
        except Exception as e:
            print(f"MonoBehaviour: parse failed: {e}")

# Also check the PackDefinition
for obj in env.objects:
    if obj.type.name == "MonoBehaviour":
        try:
            tree = obj.parse_as_dict()
            name = tree.get("m_Name", "")
            if "PackDefinition" in name:
                print(f"\n=== {name} ===")
                pack = tree.get("_beatmapLevelPack", {})
                print(f"  pack keys: {list(pack.keys()) if isinstance(pack, dict) else type(pack)}")
                if isinstance(pack, dict):
                    for k, v in pack.items():
                        val_str = repr(v)
                        if len(val_str) > 300:
                            val_str = val_str[:300] + "..."
                        print(f"    {k}: {val_str}")
        except Exception as e:
            print(f"PackDefinition parse failed: {e}")
