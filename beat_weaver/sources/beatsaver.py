import io
import json
import logging
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
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

            info = data.get("info", {})
            total_pages = info.get("pages", 0)
            page += 1
            logger.info(
                "Fetched page %d / %d, total maps found so far: %d",
                page, total_pages, total_found,
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

            if total_pages > 0 and page >= total_pages:
                break

            time.sleep(REQUEST_DELAY)

    def download_map(self, map_info: dict, dest_dir: Path) -> Path | None:
        """Download and extract a single map zip to dest_dir/<hash>.

        Thread-safe: uses a fresh GET request instead of the shared session.
        """
        try:
            map_hash = map_info["versions"][0]["hash"]
            target_dir = dest_dir / map_hash

            if target_dir.exists():
                return target_dir

            download_url = map_info["versions"][0]["downloadURL"]
            if download_url.startswith("/"):
                download_url = f"https://beatsaver.com{download_url}"

            response = requests.get(
                download_url,
                headers={"User-Agent": "BeatWeaver/0.1.0"},
            )
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
        max_maps: int = 0,
        workers: int = 8,
    ) -> list[Path]:
        """Search and download maps, returning list of extracted directories.

        Args:
            dest_dir: Directory to extract maps into (each map gets its own subfolder).
            min_score: Minimum BeatSaver rating score (0.0-1.0).
            min_upvotes: Minimum number of upvotes.
            max_maps: Maximum maps to download. 0 means unlimited (all qualifying maps).
            workers: Number of parallel download threads (default 8).

        API pagination is sequential (to respect rate limits), but zip
        file downloads run in parallel for throughput.  Already-downloaded
        maps (detected by hash folder existence) are skipped automatically,
        making this safe to resume after interruption.
        """
        from tqdm import tqdm

        dest_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []
        skipped = 0
        newly_downloaded = 0

        desc = "Downloading maps" if max_maps == 0 else f"Downloading maps (max {max_maps})"
        pbar = tqdm(total=max_maps or None, desc=desc)

        # Collect map_info objects that need downloading, then download in
        # parallel batches.  Already-present maps are counted immediately.
        batch: list[dict] = []

        def _flush_batch() -> None:
            nonlocal newly_downloaded
            if not batch:
                return
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self.download_map, info, dest_dir): info
                    for info in batch
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        newly_downloaded += 1
                        downloaded.append(result)
                        pbar.update(1)
                        pbar.set_postfix(new=newly_downloaded, skipped=skipped)
            batch.clear()

        try:
            for map_info in self.search_maps(
                min_score=min_score, min_upvotes=min_upvotes,
            ):
                map_hash = map_info["versions"][0]["hash"]
                target_dir = dest_dir / map_hash

                if target_dir.exists():
                    skipped += 1
                    downloaded.append(target_dir)
                    pbar.update(1)
                    pbar.set_postfix(new=newly_downloaded, skipped=skipped)
                else:
                    batch.append(map_info)
                    if len(batch) >= workers * 2:
                        _flush_batch()

                if max_maps > 0 and len(downloaded) >= max_maps:
                    break

            # Flush remaining
            _flush_batch()
        finally:
            pbar.close()

        logger.info(
            "Download complete: %d total (%d new, %d already present)",
            len(downloaded), newly_downloaded, skipped,
        )
        return downloaded


def load_beatsaver_meta(map_dir: Path) -> dict | None:
    """Read _beatsaver_meta.json from a map directory."""
    meta_path = map_dir / "_beatsaver_meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))
