import io
import json
import logging
import time
import zipfile
from pathlib import Path
from typing import Iterator

import requests

BASE_URL = "https://api.beatsaver.com"
DEFAULT_MIN_SCORE = 0.75
DEFAULT_MIN_UPVOTES = 5
DEFAULT_PAGE_SIZE = 20
REQUEST_DELAY = 1.0  # seconds between API requests

logger = logging.getLogger(__name__)


class BeatSaverClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "BeatWeaver/0.1.0"

    def search_maps(
        self,
        min_score: float = DEFAULT_MIN_SCORE,
        min_upvotes: int = DEFAULT_MIN_UPVOTES,
        max_pages: int = 5000,
        automapper: bool = False,
    ) -> Iterator[dict]:
        """Paginate through BeatSaver search results, yielding map docs.

        Default filters (score >= 0.75, upvotes >= 5, automapper=False)
        yield ~55,000 maps from ~115,000 total on BeatSaver. See
        LEARNINGS.md "Training Data Quality Analysis" for rationale.
        """
        total_found = 0
        page = 0

        while page < max_pages:
            response = self.session.get(
                f"{BASE_URL}/search/text/{page}",
                params={"sortOrder": "Rating"},
            )
            response.raise_for_status()
            data = response.json()

            below_threshold = 0
            for doc in data.get("docs", []):
                stats = doc.get("stats", {})
                score = stats.get("score", 0)
                upvotes = stats.get("upvotes", 0)

                if doc.get("automapper") != automapper:
                    continue
                if score < min_score:
                    below_threshold += 1
                    continue
                if upvotes < min_upvotes:
                    continue
                total_found += 1
                yield doc

            total_pages = data.get("totalPages", 0) if "totalPages" in data else 0
            page += 1
            logger.info(
                "Fetched page %d, total maps found so far: %d", page, total_found
            )

            # Stop if we've gone past the score threshold (results are
            # sorted by rating, so once a full page is below min_score
            # we won't find more matches)
            if below_threshold >= DEFAULT_PAGE_SIZE:
                logger.info(
                    "All maps on page %d below min_score=%.2f, stopping",
                    page, min_score,
                )
                break

            if page >= total_pages:
                break

            time.sleep(REQUEST_DELAY)

    def download_map(self, map_info: dict, dest_dir: Path) -> Path | None:
        """Download and extract a single map zip to dest_dir/<hash>."""
        try:
            map_hash = map_info["versions"][0]["hash"]
            target_dir = dest_dir / map_hash

            if target_dir.exists():
                return target_dir

            download_url = map_info["versions"][0]["downloadURL"]
            if download_url.startswith("/"):
                download_url = f"https://beatsaver.com{download_url}"

            response = self.session.get(download_url)
            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                zf.extractall(target_dir)

            meta_path = target_dir / "_beatsaver_meta.json"
            meta_path.write_text(json.dumps(map_info, indent=2), encoding="utf-8")

            return target_dir

        except Exception:
            logger.warning(
                "Failed to download map %s",
                map_info.get("id", "unknown"),
                exc_info=True,
            )
            return None

    def download_maps(
        self,
        dest_dir: Path,
        min_score: float = DEFAULT_MIN_SCORE,
        min_upvotes: int = DEFAULT_MIN_UPVOTES,
        max_maps: int = 100,
    ) -> list[Path]:
        """Search and download maps, returning list of extracted directories."""
        from tqdm import tqdm

        dest_dir.mkdir(parents=True, exist_ok=True)
        downloaded = []

        with tqdm(total=max_maps, desc="Downloading maps") as pbar:
            for map_info in self.search_maps(
                min_score=min_score, min_upvotes=min_upvotes,
            ):
                result = self.download_map(map_info, dest_dir)
                if result is not None:
                    downloaded.append(result)
                    pbar.update(1)

                if len(downloaded) >= max_maps:
                    break

        return downloaded


def load_beatsaver_meta(map_dir: Path) -> dict | None:
    """Read _beatsaver_meta.json from a map directory."""
    meta_path = map_dir / "_beatsaver_meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))
