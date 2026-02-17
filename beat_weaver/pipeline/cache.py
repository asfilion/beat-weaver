"""Track downloaded and processed maps to avoid re-work."""

import json
from pathlib import Path


class ProcessingCache:
    """Tracks which maps have been downloaded and processed."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._downloaded_path = cache_dir / "beatsaver_index.json"
        self._processed_path = cache_dir / "processed_hashes.json"
        self.downloaded: dict[str, dict] = self._load(self._downloaded_path)
        self.processed: set[str] = set(self._load(self._processed_path).keys())

    @staticmethod
    def _load(path: Path) -> dict:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    def is_downloaded(self, beatsaver_id: str) -> bool:
        return beatsaver_id in self.downloaded

    def mark_downloaded(self, beatsaver_id: str, hash_val: str, score: float = 0.0) -> None:
        self.downloaded[beatsaver_id] = {"hash": hash_val, "score": score}

    def is_processed(self, content_hash: str) -> bool:
        return content_hash in self.processed

    def mark_processed(self, content_hash: str) -> None:
        self.processed.add(content_hash)

    def save(self) -> None:
        self._downloaded_path.write_text(json.dumps(self.downloaded, indent=2), encoding="utf-8")
        processed_dict = {h: True for h in self.processed}
        self._processed_path.write_text(json.dumps(processed_dict, indent=2), encoding="utf-8")
