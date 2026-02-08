"""Temporary discovery script - delete after use."""
from pathlib import Path
import re

bundles_dir = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Beat Saber\Beat Saber_Data\StreamingAssets\aa\StandaloneWindows64")

all_bundles = list(bundles_dir.glob("*.bundle"))
print(f"Total bundles: {len(all_bundles)}")

patterns = {}
for b in all_bundles:
    match = re.match(r"^(.+?)_[a-f0-9]{20,}\.bundle$", b.name)
    if match:
        prefix = match.group(1)
        if prefix not in patterns:
            patterns[prefix] = []
        patterns[prefix].append(b)
    else:
        print(f"No match: {b.name}")

for p in sorted(patterns.keys()):
    bundles = patterns[p]
    total_size = sum(b.stat().st_size for b in bundles) / 1024 / 1024
    print(f"  {p}: {len(bundles)} bundles ({total_size:.1f} MB)")
