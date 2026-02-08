"""List all level bundles and count their contents."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
import UnityPy
import gzip
import json

levels_dir = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Beat Saber\Beat Saber_Data\StreamingAssets\BeatmapLevelsData")

level_files = sorted(levels_dir.iterdir())
print(f"Total level bundles: {len(level_files)}")
print()

# Summarize all level bundles
for lf in level_files:
    size_kb = lf.stat().st_size / 1024
    try:
        env = UnityPy.load(str(lf))
        text_assets = []
        audio_clips = []
        mono_behaviours = []
        for obj in env.objects:
            if obj.type.name == "TextAsset":
                d = obj.parse_as_object()
                text_assets.append(d.m_Name)
            elif obj.type.name == "AudioClip":
                d = obj.parse_as_object()
                audio_clips.append(d.m_Name)
            elif obj.type.name == "MonoBehaviour":
                try:
                    tree = obj.parse_as_dict()
                    mono_behaviours.append(tree.get("m_Name", "?"))
                except:
                    mono_behaviours.append("?")

        beatmap_count = sum(1 for t in text_assets if ".beatmap." in t)
        lightshow_count = sum(1 for t in text_assets if ".lightshow." in t)
        audio_count = sum(1 for t in text_assets if ".audio." in t)

        print(f"{lf.name}: {size_kb:.0f}KB | {beatmap_count} beatmaps, {lightshow_count} lightshows, {audio_count} audio, {len(audio_clips)} AudioClips")
    except Exception as e:
        print(f"{lf.name}: {size_kb:.0f}KB | ERROR: {e}")
