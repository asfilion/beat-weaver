[![CI](https://github.com/asfilion/beat-weaver/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/asfilion/beat-weaver/actions/workflows/ci.yml)

# beat-weaver

AI-powered Beat Saber track generator — feed in a song, get a playable custom map.

## What is this?

Beat Weaver is a mod/tool for [Beat Saber](https://beatsaber.com/) that uses machine learning to automatically generate note maps from audio files. Instead of manually placing blocks, you provide a song and the model outputs block positions, orientations, and timing for both sabers.

## Planned Features

- **Audio-to-map generation** — drop in an audio file, get a playable Beat Saber map
- **Seeded generation** — use a fixed seed for repeatable tracks, or randomize for a fresh experience every time
- **In-game feedback capture** — rate generated tracks to continuously improve the model
- **Windowed evaluation** — score individual segments of a track for finer-grained training data

## Project Status

**Data pipeline complete** — can extract, parse, and normalize Beat Saber maps from local custom levels, BeatSaver community maps, and official Unity asset bundles into ML-ready Parquet format. Next phase: ML model design and training.

## License

TBD
