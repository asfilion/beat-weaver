"""Map all pack bundles to their contained level IDs."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
import UnityPy

bundles_dir = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Beat Saber\Beat Saber_Data\StreamingAssets\aa\StandaloneWindows64")

DIFFICULTY_INT_TO_NAME = {0: "Easy", 1: "Normal", 2: "Hard", 3: "Expert", 4: "ExpertPlus"}

pack_bundles = sorted(bundles_dir.glob("*_pack_assets_all_*.bundle"))
print(f"Total pack bundles: {len(pack_bundles)}")

all_level_metadata = {}

for pb in pack_bundles:
    pack_name = pb.name.split("_pack_assets_all_")[0]
    env = UnityPy.load(str(pb))

    for obj in env.objects:
        if obj.type.name != "MonoBehaviour":
            continue
        try:
            tree = obj.parse_as_dict()
            name = tree.get("m_Name", "")
            level_id = tree.get("_levelID")
            if level_id and "BeatmapLevel" in name and "Pack" not in name and "Product" not in name and "Leaderboard" not in name:
                sets = tree.get("_previewDifficultyBeatmapSets", [])
                diff_info = []
                for s in sets:
                    char_ref = s.get("_beatmapCharacteristic", {})
                    for d in s.get("_previewDifficultyBeatmaps", []):
                        diff_int = d.get("_difficulty", -1)
                        diff_name = DIFFICULTY_INT_TO_NAME.get(diff_int, f"Unknown({diff_int})")
                        diff_info.append({
                            "difficulty": diff_name,
                            "noteJumpSpeed": d.get("_noteJumpMovementSpeed", 0.0),
                            "noteJumpOffset": d.get("_noteJumpStartBeatOffset", 0.0),
                        })

                all_level_metadata[level_id] = {
                    "pack": pack_name,
                    "song_name": tree.get("_songName", ""),
                    "song_author": tree.get("_songAuthorName", ""),
                    "bpm": tree.get("_beatsPerMinute", 0.0),
                    "duration": tree.get("_songDuration", 0.0),
                    "offset": tree.get("_songTimeOffset", 0.0),
                }
        except Exception:
            pass

print(f"\nTotal levels found in pack bundles: {len(all_level_metadata)}")
for lid, meta in sorted(all_level_metadata.items()):
    print(f"  {lid}: {meta['song_name']} by {meta['song_author']} (BPM={meta['bpm']}, pack={meta['pack']})")

# Check how many match level bundle files
levels_dir = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Beat Saber\Beat Saber_Data\StreamingAssets\BeatmapLevelsData")
level_bundle_names = set(f.name.lower() for f in levels_dir.iterdir())
print(f"\nLevel bundles: {len(level_bundle_names)}")
matched = sum(1 for lid in all_level_metadata if lid.lower() in level_bundle_names)
print(f"Matched to bundle files: {matched}")
unmatched_meta = [lid for lid in all_level_metadata if lid.lower() not in level_bundle_names]
if unmatched_meta:
    print(f"Unmatched metadata (no bundle file): {unmatched_meta[:10]}")
unmatched_bundle = [bn for bn in sorted(level_bundle_names) if bn not in set(lid.lower() for lid in all_level_metadata)]
if unmatched_bundle:
    print(f"Unmatched bundles (no metadata): {unmatched_bundle[:10]}")
