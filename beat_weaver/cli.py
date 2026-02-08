"""Command-line interface for Beat Weaver data pipeline."""

import argparse
import logging
from pathlib import Path


def cmd_download(args: argparse.Namespace) -> None:
    from beat_weaver.sources.beatsaver import BeatSaverClient

    client = BeatSaverClient()
    downloaded = client.download_maps(
        dest_dir=Path(args.output),
        min_score=args.min_score,
        max_maps=args.max_maps,
    )
    print(f"Downloaded {len(downloaded)} maps to {args.output}")


def cmd_extract_official(args: argparse.Namespace) -> None:
    from beat_weaver.sources.unity_extractor import extract_official_maps

    bundles_dir = (
        Path(args.beat_saber)
        / "Beat Saber_Data"
        / "StreamingAssets"
        / "aa"
        / "StandaloneWindows64"
    )
    extracted = extract_official_maps(bundles_dir, Path(args.output))
    print(f"Extracted {len(extracted)} map folders to {args.output}")


def cmd_process(args: argparse.Namespace) -> None:
    from beat_weaver.pipeline.processor import process_map_folder
    from beat_weaver.storage.writer import write_parquet

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    all_beatmaps = []

    for folder in sorted(input_dir.rglob("Info.dat")):
        map_folder = folder.parent
        beatmaps = process_map_folder(
            map_folder, source="raw", source_id=map_folder.name
        )
        all_beatmaps.extend(beatmaps)

    write_parquet(all_beatmaps, output_dir)
    print(f"Processed {len(all_beatmaps)} beatmaps to {output_dir}")


def cmd_run(args: argparse.Namespace) -> None:
    from beat_weaver.pipeline.batch import PipelineConfig, run_pipeline

    config = PipelineConfig(
        beat_saber_path=Path(args.beat_saber),
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output),
        cache_dir=Path(args.cache_dir),
        include_local=not args.no_local,
        include_beatsaver=not args.no_beatsaver,
        include_official=not args.no_official,
        min_score=args.min_score,
        max_beatsaver_maps=args.max_maps,
    )
    result = run_pipeline(config)
    print(f"Done: {result.total_songs} songs, {result.total_beatmaps} beatmaps, {result.total_notes} notes")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="beat-weaver",
        description="Beat Weaver training data pipeline",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    sub = parser.add_subparsers(dest="command")

    # download
    dl = sub.add_parser("download", help="Download custom maps from BeatSaver")
    dl.add_argument("--min-score", type=float, default=0.7)
    dl.add_argument("--max-maps", type=int, default=100)
    dl.add_argument("--output", default="data/raw/beatsaver")

    # extract-official
    ext = sub.add_parser("extract-official", help="Extract maps from Unity bundles")
    ext.add_argument(
        "--beat-saber",
        default=r"C:\Program Files (x86)\Steam\steamapps\common\Beat Saber",
    )
    ext.add_argument("--output", default="data/raw/official")

    # process
    proc = sub.add_parser("process", help="Normalize raw maps into Parquet")
    proc.add_argument("--input", default="data/raw")
    proc.add_argument("--output", default="data/processed")

    # run (full pipeline)
    run = sub.add_parser("run", help="Run full pipeline")
    run.add_argument(
        "--beat-saber",
        default=r"C:\Program Files (x86)\Steam\steamapps\common\Beat Saber",
    )
    run.add_argument("--raw-dir", default="data/raw")
    run.add_argument("--output", default="data/processed")
    run.add_argument("--cache-dir", default="data/cache")
    run.add_argument("--min-score", type=float, default=0.7)
    run.add_argument("--max-maps", type=int, default=100)
    run.add_argument("--no-local", action="store_true")
    run.add_argument("--no-beatsaver", action="store_true")
    run.add_argument("--no-official", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    commands = {
        "download": cmd_download,
        "extract-official": cmd_extract_official,
        "process": cmd_process,
        "run": cmd_run,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
