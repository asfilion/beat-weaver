"""Orchestrate the full data pipeline across all sources."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from beat_weaver.pipeline.cache import ProcessingCache
from beat_weaver.pipeline.processor import process_map_folder
from beat_weaver.schemas.normalized import NormalizedBeatmap
from beat_weaver.sources.local_custom import iter_local_custom_maps
from beat_weaver.storage.writer import write_parquet

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    beat_saber_path: Path = Path(
        r"C:\Program Files (x86)\Steam\steamapps\common\Beat Saber"
    )
    raw_dir: Path = Path("data/raw")
    output_dir: Path = Path("data/processed")
    cache_dir: Path = Path("data/cache")
    include_local: bool = True
    include_beatsaver: bool = True
    include_official: bool = True
    min_score: float = 0.7
    max_beatsaver_maps: int = 100


@dataclass
class PipelineResult:
    total_songs: int = 0
    total_beatmaps: int = 0
    total_notes: int = 0
    errors: list[str] = field(default_factory=list)


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run the full pipeline: gather from all sources, normalize, write output."""
    all_beatmaps: list[NormalizedBeatmap] = []
    result = PipelineResult()
    cache = ProcessingCache(config.cache_dir)

    # 1. Local custom levels
    if config.include_local:
        logger.info("Processing local custom levels...")
        for bm in iter_local_custom_maps(config.beat_saber_path):
            all_beatmaps.append(bm)

    # 2. BeatSaver downloads
    if config.include_beatsaver:
        logger.info("Processing BeatSaver maps...")
        from beat_weaver.sources.beatsaver import BeatSaverClient

        client = BeatSaverClient()
        downloaded = client.download_maps(
            dest_dir=config.raw_dir / "beatsaver",
            min_score=config.min_score,
            max_maps=config.max_beatsaver_maps,
        )
        for folder in downloaded:
            beatmaps = process_map_folder(folder, "beatsaver", folder.name)
            # Attach BeatSaver quality metadata
            from beat_weaver.sources.beatsaver import load_beatsaver_meta

            meta = load_beatsaver_meta(folder)
            if meta:
                for bm in beatmaps:
                    stats = meta.get("stats", {})
                    bm.metadata.score = stats.get("score")
                    bm.metadata.upvotes = stats.get("upvotes")
                    bm.metadata.downvotes = stats.get("downvotes")
            all_beatmaps.extend(beatmaps)

    # 3. Official maps
    if config.include_official:
        logger.info("Processing official maps...")
        from beat_weaver.sources.unity_extractor import extract_official_maps

        bundles_dir = (
            config.beat_saber_path
            / "Beat Saber_Data"
            / "StreamingAssets"
            / "aa"
            / "StandaloneWindows64"
        )
        official_dir = config.raw_dir / "official"
        extracted = extract_official_maps(bundles_dir, official_dir)
        for folder in extracted:
            beatmaps = process_map_folder(folder, "official", folder.name)
            all_beatmaps.extend(beatmaps)

    # 4. Write output
    logger.info("Writing %d beatmaps to %s...", len(all_beatmaps), config.output_dir)
    write_parquet(all_beatmaps, config.output_dir)

    # 5. Collect stats
    seen_songs: set[str] = set()
    for bm in all_beatmaps:
        key = f"{bm.metadata.source}:{bm.metadata.source_id}"
        seen_songs.add(key)
        result.total_notes += len(bm.notes)
    result.total_songs = len(seen_songs)
    result.total_beatmaps = len(all_beatmaps)

    cache.save()
    logger.info(
        "Pipeline complete: %d songs, %d beatmaps, %d notes",
        result.total_songs,
        result.total_beatmaps,
        result.total_notes,
    )
    return result
