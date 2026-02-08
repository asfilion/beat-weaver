"""Explore more deeply for beatmap data."""
from pathlib import Path
import UnityPy

bundles_dir = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Beat Saber\Beat Saber_Data\StreamingAssets\aa\StandaloneWindows64")

# Check what's in the parent directories for beatmap data
aa_dir = bundles_dir.parent
print("=== Contents of aa/ directory ===")
for item in sorted(aa_dir.iterdir()):
    if item.is_dir():
        count = sum(1 for _ in item.iterdir())
        print(f"  DIR: {item.name}/ ({count} items)")
    else:
        print(f"  FILE: {item.name} ({item.stat().st_size / 1024:.1f} KB)")

# Check StreamingAssets for other dirs
sa_dir = aa_dir.parent
print("\n=== Contents of StreamingAssets/ ===")
for item in sorted(sa_dir.iterdir()):
    if item.is_dir():
        count = sum(1 for _ in item.iterdir())
        print(f"  DIR: {item.name}/ ({count} items)")
    else:
        print(f"  FILE: {item.name} ({item.stat().st_size / 1024:.1f} KB)")

# Look for any .dat or .json files that could be beatmaps
data_dir = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Beat Saber\Beat Saber_Data")
print("\n=== Contents of Beat Saber_Data/ (top level) ===")
for item in sorted(data_dir.iterdir()):
    if item.is_dir():
        print(f"  DIR: {item.name}/")
    else:
        print(f"  FILE: {item.name} ({item.stat().st_size / 1024:.1f} KB)")
