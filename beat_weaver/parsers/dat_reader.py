"""Utility to read Beat Saber .dat files with automatic gzip detection."""

import gzip
import json
from pathlib import Path

GZIP_MAGIC = b'\x1f\x8b'


def read_dat_file(filepath: Path) -> dict:
    """Read a .dat file, auto-detecting gzip compression. Returns parsed JSON dict."""
    raw = filepath.read_bytes()
    if raw[:2] == GZIP_MAGIC:
        raw = gzip.decompress(raw)
    return json.loads(raw)
