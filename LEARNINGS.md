# Beat Weaver Research Library

This document is the project's curated knowledge base, capturing research findings from each phase of development. It serves as a durable reference for Beat Saber map formats, data sources, extraction techniques, and prior art in automatic map generation. New sections are appended as research progresses — existing content is never removed.

---

## Table of Contents

### Map Format Specification
- [File Format Overview](#file-format-overview) — Schema versions v2/v3/v4 and their differences
- [The Note Grid](#the-note-grid) — 4x3 grid coordinate system
- [Color Values](#color-values) — Red (0) and Blue (1) saber mapping
- [Cut Direction Values](#cut-direction-values) — Direction integers 0-8
- [Timing](#timing) — Beat-based timing and BPM conversion
- [Difficulty Ranks](#difficulty-ranks) — Easy through ExpertPlus rank values
- [Characteristics](#characteristics) — Standard, OneSaber, NoArrows, 360/90 Degree

### File Structures by Version
- [v2 Info.dat Structure](#v2-infodat-structure-most-common) — Underscore-prefixed metadata format
- [v2 Beatmap File Structure](#v2-beatmap-file-structure) — `_notes` and `_obstacles` arrays
- [v3 Beatmap File Structure](#v3-beatmap-file-structure) — Abbreviated single-letter field names
- [v4 Beatmap File Structure](#v4-beatmap-file-structure) — Index-based compression with data arrays

### Data Sources and Access
- [File Locations on Disk (Steam)](#file-locations-on-disk-steam) — Where custom and official maps live
- [Local Installation Structure](#local-installation-structure) — Detailed Beat Saber directory layout
- [BeatSaver API](#beatsaver-api) — Community map download endpoints, filtering, rate limits
- [Unity Asset Extraction](#unity-asset-extraction) — Tools for extracting official maps from bundles

### Implementation Reference
- [Parsing Strategy](#parsing-strategy) — Step-by-step approach to reading map data
- [Existing Tools and Libraries](#existing-tools-and-libraries) — Python, TypeScript, and Rust parsers
- [Key Insight for Training Data](#key-insight-for-training-data) — Custom vs official map accessibility

### Prior Art
- [Existing ML Automapper Projects](#existing-ml-automapper-projects) — DeepSaber, InfernoSaber, BeatMapSynthesizer, Beat Sage

### Verified Extraction Results
- [Unity Bundle Internal Structure](#unity-bundle-internal-structure-verified-via-extraction) — Two-location architecture, BPM resolution, extraction stats

### ML Model Research
- [Audio Feature Extraction](#audio-feature-extraction) — Mel spectrograms, onset detection, librosa parameters
- [Model Architecture](#model-architecture) — Encoder-decoder transformer, prior automapper architectures
- [Output Representation](#output-representation) — Beat-quantized event tokens, compound tokens, vocabulary design
- [Evaluation and Quality Metrics](#evaluation-and-quality-metrics) — Playability heuristics, rhythm alignment, flow, training losses
- [Map Export](#map-export) — Writing playable v2 custom maps, audio requirements, packaging

---

## File Format Overview

Beat Saber maps are collections of **JSON files** in a folder (distributed as `.zip` via BeatSaver). Four major schema versions exist:

| Schema | Introduced | Key Traits |
|--------|-----------|------------|
| **v2** | 1.0.0 | Underscore-prefixed fields (`_time`, `_lineIndex`). Files use `.dat`. Most custom maps use this. |
| **v3** | 1.20.0 | Abbreviated single-letter fields (`b`, `x`, `y`, `c`, `d`). Separate collections for bombs, arcs, chains. |
| **v4** | 1.34.5 | Index-based compression: object data in separate metadata arrays, referenced by index. Beatmap and lightshow split into separate files. Supports gzip. |

## File Locations on Disk (Steam)

### Custom Maps
```
<Beat Saber>/Beat Saber_Data/CustomLevels/<hash> (Song Name - Mapper)/
    Info.dat
    Easy.dat / Normal.dat / Hard.dat / Expert.dat / ExpertPlus.dat
    song.ogg (or song.egg)
    cover.jpg (or cover.png)
```

### Official/DLC Maps
Official OST and DLC maps are bundled inside **Unity asset files** in `Beat Saber_Data/`. As of v1.34.5, they use gzip-compressed v4 JSON. They are NOT accessible as plain folders — they're packed into Unity asset bundles.

## The Note Grid

4-column x 3-row grid:
```
        Col 0    Col 1    Col 2    Col 3
        (Left)                     (Right)
Row 2   [0,2]    [1,2]    [2,2]    [3,2]    (Top)
Row 1   [0,1]    [1,1]    [2,1]    [3,1]    (Middle)
Row 0   [0,0]    [1,0]    [2,0]    [3,0]    (Bottom)
```

- **Line Index (x):** 0-3, left to right
- **Line Layer (y):** 0-2, bottom to top

## Color Values

| Value | Meaning |
|-------|---------|
| 0 | Red (Left Saber) |
| 1 | Blue (Right Saber) |

## Cut Direction Values

| Value | Direction |
|-------|-----------|
| 0 | Up |
| 1 | Down |
| 2 | Left |
| 3 | Right |
| 4 | Up-Left |
| 5 | Up-Right |
| 6 | Down-Left |
| 7 | Down-Right |
| 8 | Any (dot note) |

## Timing

All timing is in **beats** (float), not seconds. Convert: `time_seconds = beat / BPM * 60`

## Difficulty Ranks

| Difficulty | Rank |
|-----------|------|
| Easy | 1 |
| Normal | 3 |
| Hard | 5 |
| Expert | 7 |
| ExpertPlus | 9 |

## Characteristics

`Standard`, `NoArrows`, `OneSaber`, `360Degree`, `90Degree`, `Legacy`

## v2 Info.dat Structure (Most Common)

```json
{
  "_version": "2.1.0",
  "_songName": "Song Title",
  "_songSubName": "feat. Artist",
  "_songAuthorName": "Artist Name",
  "_levelAuthorName": "Mapper Name",
  "_beatsPerMinute": 120.0,
  "_songTimeOffset": 0,
  "_shuffle": 0,
  "_shufflePeriod": 0.5,
  "_previewStartTime": 30.0,
  "_previewDuration": 10.0,
  "_songFilename": "song.ogg",
  "_coverImageFilename": "cover.jpg",
  "_environmentName": "DefaultEnvironment",
  "_difficultyBeatmapSets": [
    {
      "_beatmapCharacteristicName": "Standard",
      "_difficultyBeatmaps": [
        {
          "_difficulty": "Easy",
          "_difficultyRank": 1,
          "_beatmapFilename": "Easy.dat",
          "_noteJumpMovementSpeed": 10.0,
          "_noteJumpStartBeatOffset": 0.0
        }
      ]
    }
  ]
}
```

## v2 Beatmap File Structure

```json
{
  "_version": "2.6.0",
  "_notes": [
    {
      "_time": 10.0,
      "_lineIndex": 1,
      "_lineLayer": 0,
      "_type": 0,
      "_cutDirection": 1
    }
  ],
  "_obstacles": [
    {
      "_time": 15.0,
      "_lineIndex": 0,
      "_lineLayer": 0,
      "_type": 0,
      "_duration": 2.0,
      "_width": 2
    }
  ],
  "_events": [],
  "_sliders": [],
  "_waypoints": [],
  "_customData": {}
}
```

**Note `_type` values:** 0 = Red, 1 = Blue, 3 = Bomb

**Obstacle `_type` values:** 0 = Full-height wall, 1 = Crouch wall

## v3 Beatmap File Structure

```json
{
  "version": "3.3.0",
  "colorNotes": [
    { "b": 10.0, "x": 1, "y": 0, "c": 0, "d": 1, "a": 0 }
  ],
  "bombNotes": [
    { "b": 12.0, "x": 2, "y": 1 }
  ],
  "obstacles": [
    { "b": 15.0, "x": 0, "y": 0, "d": 2.0, "w": 2, "h": 5 }
  ],
  "sliders": [],
  "burstSliders": [],
  "bpmEvents": [],
  "rotationEvents": [],
  "basicBeatmapEvents": [],
  "colorBoostBeatmapEvents": [],
  "customData": {}
}
```

**v3 field key:** `b`=beat, `x`=column, `y`=row, `c`=color, `d`=cut direction, `a`=angle offset

## v4 Beatmap File Structure

Uses index-based compression — objects reference metadata by index into separate arrays:

```json
{
  "version": "4.0.0",
  "colorNotes": [
    { "b": 10.0, "r": 0, "i": 0 }
  ],
  "colorNotesData": [
    { "x": 1, "y": 0, "c": 0, "d": 1, "a": 0 }
  ],
  "bombNotes": [],
  "bombNotesData": [],
  "obstacles": [],
  "obstaclesData": [],
  "arcs": [],
  "arcsData": [],
  "chains": [],
  "chainsData": []
}
```

Multiple objects can reference the same metadata index for deduplication.

## Parsing Strategy

1. Read `Info.dat` as JSON. Detect schema version from `_version` or `version`.
2. Extract difficulty filenames from the appropriate path per version.
3. Read each difficulty `.dat` file as JSON.
4. Parse notes/obstacles/events from the version-appropriate arrays.
5. Convert beat timing to seconds: `seconds = beat * 60.0 / BPM`.
6. Handle gzip: if `.dat` starts with magic bytes `1f 8b`, decompress first.

## Existing Tools and Libraries

### Python
- Format is plain JSON — `json` module works directly
- [InfernoSaber](https://github.com/fred-brenner/InfernoSaber---BeatSaber-Automapper) — Python automapper
- [BeatMapSynthesizer](https://github.com/wvsharber/BeatMapSynthesizer) — Automatic mapper using Pandas
- [SimSaber](https://github.com/MetaGuard/SimSaber) — Replay simulator that parses maps

### TypeScript — bsmap
- [github.com/KivalEvan/BeatSaber-JSMap](https://github.com/KivalEvan/BeatSaber-JSMap)
- Supports v1.5, v2.6, v3.3, v4.1 schemas

### Rust — beat_saber_map
- [docs.rs/beat_saber_map](https://docs.rs/beat_saber_map/latest/beat_saber_map/)

### Map Distribution
- [BeatSaver](https://beatsaver.com/) — primary custom map repository (`.zip` downloads)
- [BeastSaber](https://bsaber.com/) — curated listings

## Key Insight for Training Data

Custom maps (v2 format, JSON) are freely available from BeatSaver and trivially parseable. Official/DLC maps are locked inside Unity asset bundles and harder to extract. The bulk of accessible training data will come from the custom map community.

---

## Local Installation Structure

Explored: `C:\Program Files (x86)\Steam\steamapps\common\Beat Saber`

### Top-Level
```
Beat Saber.exe
Beat Saber_Data/
    Managed/              # .NET assemblies (BeatmapCore.dll, etc.)
    StreamingAssets/       # Addressable assets (official maps)
    CustomLevels/         # Plain JSON custom/built-in maps
```

### Built-In Custom Levels (2 tracks)
Located in `Beat Saber_Data/CustomLevels/`:
- `Jaroslav Beck - Beat Saber (Built in)` — v2 Info.dat, v2 beatmaps, 166 BPM
- `Jaroslav Beck - Magic (Built in)` — v2 Info.dat, **v3 beatmaps**, 208 BPM

Each folder contains: `Info.dat`, multiple `{Difficulty}.dat` files (Easy through ExpertPlus, plus OneSaber/NoArrows/360/90 variants), `song.ogg`, `cover.png`.

**Important edge case:** Magic has v2 `Info.dat` (`_version: 2.0.0`) but v3 difficulty files (`version: 3.0.0`). Info.dat and beatmap versions must be detected independently per file.

### Official/DLC Maps — Unity Asset Bundles
Located in `Beat Saber_Data/StreamingAssets/aa/StandaloneWindows64/`:

- **`BeatmapLevelsData/`** — ~60 binary files named by track ID (e.g., `100bills`, `beatsaber`, `crabrave`, `crystallized`). 6-22 MB each. NOT JSON — binary compressed.
- **`.bundle` files** — 140+ Unity asset bundles (676 MB total). Named like `{pack}_assets_all__{hash}.bundle` (e.g., `ostvol1_pack_assets_all_*.bundle`, `billieeilish_pack_assets_all_*.bundle`). 38 pack bundles covering 8 OST volumes + ~30 DLC packs.
- **`catalog.json`** (841 KB) — Unity Addressables master catalog referencing all content bundles.

## Unity Asset Extraction

### UnityPy (Recommended Python Library)
- **Install:** `pip install UnityPy`
- Load any `.assets`, `.bundle`, folder, or bytes
- Key asset types for maps: `TextAsset` (map JSON data), `MonoBehaviour` (serialized C# classes), `AudioClip` (song audio)
- For `TextAsset`: access `data.m_Script`, decode with `encode("utf-8", "surrogateescape")`
- For gzip-compressed content: check for magic bytes `1f 8b`, decompress before JSON parsing
- MonoBehaviour may need `TypeTreeGenerator` if type trees are missing from bundles

### Other Extraction Tools (GUI, for exploration)
- [AssetRipper](https://github.com/AssetRipper/AssetRipper) — C# GUI, full project reconstruction
- [UABE/UABEA](https://github.com/SeriousCache/UABE) — C# GUI, manual asset inspection
- [AssetStudio](https://github.com/Perfare/AssetStudio) — C# GUI, visual exploration and bulk export

**Recommended workflow:** Use AssetRipper/AssetStudio first to visually identify which bundle files contain map data, then automate with UnityPy.

## BeatSaver API

### Endpoints
- Base URL: `https://api.beatsaver.com`
- [Swagger docs](https://api.beatsaver.com/docs/)
- `GET /search/text/{page}` — full-text search with filters (sortOrder, minRating, pageSize=20)
- `GET /maps/id/{id}` — get map by BeatSaver ID
- `GET /maps/hash/{hash}` — get map by hash
- `GET /maps/latest` — latest maps (paginated)
- Download URL in response: `versions[0].downloadURL` (zip file)

### Filtering
- `score >= 0.7` (70%+ user rating) filters ~20k-40k usable maps from ~115k total
- `automapper=false` to exclude AI-generated maps
- Response includes `stats.score`, `stats.upvotes`, `stats.downvotes`, `versions[0].diffs[*].nps`

### Rate Limiting
- Be polite: 1 request/second
- Respect `Retry-After` headers

### Python Wrappers
- [BeatSaver.py](https://github.com/Sirspam/BeatSaver.py) — PyPI package
- [beatsaver-python](https://github.com/megamaz/beatsaver-python) — alternative

### Bulk Download Tools
- [ARBSMapDo](https://github.com/Luux/ARBSMapDo) — filter by ScoreSaber rank + BeatSaver metadata
- [Map Pack Downloader](https://github.com/medAndro/Beatsaber-Map-Pack-Downloader) — download from playlists

## Existing ML Automapper Projects

### DeepSaber (OxAI Labs)
- **Repo:** [github.com/oxai/deepsaber](https://github.com/oxai/deepsaber)
- **Data:** 765 songs (613 train / 80 val / 72 test), ~40 hours audio, 903k training actions
- **Dataset:** [MEGA download](https://mega.nz/#!sABVnYYJ!ZWImW0OSCD_w8Huazxs3Vr0p_2jCqmR44IB9DCKWxac)
- **Source:** Scraped from BeatSaver/BeastSaber
- **Audio features:** `librosa` mel spectrograms (multi-resolution, size 80 and 100)
- **Model:** Two-stage: WaveNet/DDC for block timing, then beam search (beam_size=17) for block selection
- **Pre-trained weights:** [MEGA](https://mega.nz/#!tJBxTC5C!nXspSCKfJ6PYJjdKkFVzIviYEhr0BSg8zXINBqC5rpA)

### InfernoSaber
- **Repo:** [github.com/fred-brenner/InfernoSaber---BeatSaber-Automapper](https://github.com/fred-brenner/InfernoSaber---BeatSaber-Automapper)
- **Data:** Hundreds of user-created maps from BeatSaver/BSaber (excludes modded/out-of-bounds maps)
- **Model:** Four consecutive models: convolutional autoencoder → TCN for timing → DNN for note placement → DNN for lighting
- **Requirements:** 10-20 GB RAM, 8-15 GB VRAM per 50 maps
- **Pre-trained:** [Hugging Face](https://huggingface.co/BierHerr/InfernoSaber)

### BeatMapSynthesizer
- **Repo:** [github.com/wvsharber/BeatMapSynthesizer](https://github.com/wvsharber/BeatMapSynthesizer)
- **Data:** ~8,000 maps from BeatSaver (filtered to >70% rating from 20k+ available)
- **Pipeline:** BeatSaver API download → extract audio + JSON → `librosa` beat detection + spectral features → align with block data → store as Pandas DataFrames
- **Model:** Hidden Markov Models (5 hidden states per difficulty) using `markovify`
- **Blog:** [Medium writeup](https://medium.com/swlh/beatmapsynth-an-automatic-song-mapper-for-beat-saber-aa9e59f093f8)

### Beat Sage
- **Website:** [beatsage.com](https://beatsage.com/)
- **Model:** Two neural networks: timing network (spectrogram → block placement) + block assignment network (timestamp → note properties)
- **Data:** Curated official + community maps (not open-sourced)

---

## Unity Bundle Internal Structure (Verified via Extraction)

### Three-Location Architecture
Official maps are stored across three locations:

1. **Pack bundles** (`StandaloneWindows64/<pack>_pack_assets_all_*.bundle`):
   - 38 pack bundles total (OST volumes + DLC packs)
   - Contain `MonoBehaviour` objects with song metadata per level: `_levelID`, `_songName`, `_songAuthorName`, `_beatsPerMinute`, `_songDuration`, difficulty preview info with NJS/NJO
   - 328 levels with metadata extracted across all packs

2. **Level data bundles** (`BeatmapLevelsData/<levelID>` — no file extension):
   - 65 UnityFS bundles, one per level (base game / OST)
   - Contain:
     - `MonoBehaviour` ("BeatmapLevelDataSO") mapping characteristics+difficulties to `TextAsset` path IDs
     - Gzipped `TextAsset` files: `<Name><Diff>.beatmap.gz` (v4 JSON), `<Name>.audio.gz` (BPM/sample data), `<Name>.lightshow.gz`
     - `AudioClip` (song audio) — extractable via `obj.parse_as_object().samples` → `{filename: bytes}` WAV data

3. **DLC level bundles** (`DLC/Levels/<LevelName>/<bundlefile>`):
   - 149 individual bundles, one per DLC level
   - Same internal structure as BeatmapLevelsData bundles (MonoBehaviour + TextAssets + AudioClip)
   - Container paths reference `packages/com.beatgames.beatsaber.packs.<pack-name>/so/<levelname>/`
   - Pack metadata may not exist for all DLC levels; BPM falls back to audio.gz derivation

### AudioClip Extraction
UnityPy can decode AudioClip assets directly. `clip.samples` returns a dict of `{filename: bytes}` containing decoded WAV audio. All 214 level bundles (65 base + 149 DLC) contain an AudioClip asset. Total extracted audio: ~7.9 GB WAV. Audio is written as `song.wav` alongside the beatmap files, and `_songFilename` is set in the synthesized `Info.dat`.

### Level ID Matching
Pack metadata uses `_levelID` (e.g., "100Bills"), level bundles use filename (e.g., "100bills"). Match is case-insensitive.

### BPM Resolution
Pack metadata `_beatsPerMinute` is authoritative. Fallback: derive from `audio.gz`'s `bpmData` array using `last_entry.eb / (last_entry.ei / songFrequency) * 60`.

### Extraction Output
Each level extracted to a standard map folder with synthesized v2-style `Info.dat` + gzip-compressed v4 `.dat` files. The existing `parse_map_folder()` + `dat_reader` (auto-gzip) handles these seamlessly.

### Verified Stats
- 214 levels extracted successfully (65 base + 149 DLC, 100% success rate)
- Example "100bills": 12 beatmaps (Standard 5 diffs + OneSaber + NoArrows + 360/90 Degree)
- v4 format with gzip compression confirmed across all extracted levels
- Total extracted audio: ~7.9 GB WAV

---

## Audio Feature Extraction

### Recommended Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Sample rate | 22,050 Hz | Music standard, used by all prior automappers |
| n_mels | 80 | Whisper / ISMIR 2023 / DeepSaber standard |
| n_fft | 2048 | Good frequency resolution at 22050 Hz (93ms window) |
| hop_length | 512 | ~43 frames/sec, sufficient for note placement |
| Window | Hann | Universal default |
| Scale | Log-magnitude | Standard for deep learning audio |
| Input format | Frame-level (not patch) | Better for seq-to-seq generation |

### Primary Feature: Log-Mel Spectrogram

Used by every successful prior system. At default parameters, a 3-minute song produces `(80, ~7,756)` frames. Captures full spectral content in a compact, perceptually-motivated representation.

### Auxiliary Features (Optional)

- **Onset strength envelope** (1 channel) — strong signal for note timing, nearly free to compute
- **Beat indicator** (1 channel) — binary/soft indicator of beat positions
- Multi-channel stacking: `[mel_spec (80), onset_strength (1), beat_indicator (1)] = 82 channels`

### What Prior Projects Used

| Project | Audio Features |
|---------|---------------|
| DeepSaber | Multi-resolution mel spectrograms (size 80, 100) + MFCCs |
| Beat Sage | Log-mel spectrogram in temporal windows |
| InfernoSaber | Deep convolutional autoencoder on raw audio |
| BeatMapSynthesizer | librosa beat detection + harmonic/percussive separation |
| Mapperatorinator (osu!) | Whisper encoder on mel spectrogram (80 bins, 16kHz) |
| ISMIR 2023 paper | 80-bin log-mel, beat-aligned hop size (1/48 beat) |

### Key Finding: Beat Alignment

The ISMIR 2023 paper ([arXiv 2311.13687](https://arxiv.org/abs/2311.13687)) demonstrated that **beat alignment of training data is vital for successful transformer training**. Aligning spectrogram frames to musical beats rather than fixed time windows significantly improves results.

### Feature Relevance Ranking

1. Mel Spectrogram (essential)
2. Onset Detection (essential for timing)
3. Beat Tracking (essential for alignment)
4. Tempogram (useful for tempo-varying content)
5. Spectral Contrast (moderate — texture changes)
6. Chroma (moderate — structural awareness)
7. MFCCs (largely redundant with mel spectrogram)

---

## Model Architecture

### Recommended: Encoder-Decoder Transformer

```
Audio (wav) → Log-Mel Spectrogram → Beat-Aligned Framing → Transformer Encoder
                                                                    |
                                                               [cross-attention]
                                                                    |
[DIFFICULTY] [BOS] → Transformer Decoder → Token Sequence → Beat Saber Map
```

**Encoder:** 6-8 layers, beat-aligned mel spectrogram input (4-8 beat context windows), relative positional encoding. Optionally initialize from pretrained AST weights.

**Decoder:** 6-8 layers, causal self-attention + cross-attention to encoder. Autoregressive token generation.

**Model size:** Start at ~20-60M parameters (similar to MT3 T5-small). Scale up if data permits.

**Training:** Cross-entropy on next-token prediction + auxiliary losses (beat alignment, note density regression).

### Why Encoder-Decoder (Not Decoder-Only)

- Audio understanding benefits from **bidirectional** context (encoder)
- Note generation is inherently **sequential/causal** (decoder)
- Cross-attention lets the decoder attend to any part of the audio at every generation step
- This is what MT3, the ISMIR 2023 paper, and Mapperatorinator all use successfully

### Prior Automapper Architecture Summary

| Project | Year | Architecture | Approach |
|---------|------|-------------|----------|
| DeepSaber | 2019 | CNN + Multi-LSTM + beam search | Two-stage: DDC timing → LSTM block selection |
| Beat Sage | 2020 | Two neural networks | Two-stage: timing from audio → block assignment |
| InfernoSaber | 2022-25 | Autoencoder + TCN + 2 DNNs | Four-stage pipeline |
| BeatMapSynthesizer | 2020 | Hidden Markov Models | Statistical (not neural) |
| BeatLearning | 2024 | Transformer (NanoGPT-style) | Masked encoder + left-to-right decoder |
| ISMIR 2023 | 2023 | Encoder-decoder Transformer | Beat-aligned spectrogram-to-sequence |
| Mapperatorinator | 2024 | Whisper-based enc-dec (219M) | Mel spectrogram encoder, event token decoder |

### Two-Stage vs End-to-End

**Two-stage** (DeepSaber, Beat Sage, DDC): Timing model → placement model. Easier to debug, but suffers from error propagation and "repetitive, incoherent local patterns" (DDC finding).

**End-to-end** (ISMIR 2023, BeatLearning, TaikoNation): Single model outputs timing + properties jointly. Better pattern coherence, but harder to train.

**Recommended:** End-to-end transformer with beat-aligned preprocessing. Gets the benefits of joint optimization while encoding musical structure as an inductive bias.

### Difficulty Conditioning

Prepend a difficulty token (`[EASY]` through `[EXPERT_PLUS]`) to the decoder sequence. The transformer learns to condition its entire output distribution via self-attention. Can also add difficulty embedding to encoder frames for stronger conditioning.

### Key References

- [MT3: Music Transcription Transformer](https://github.com/magenta/mt3) — T5 enc-dec, spectrogram → MIDI tokens
- [Music Transformer](https://openreview.net/pdf?id=rJe4ShAcF7) — Relative self-attention for temporal patterns
- [AST: Audio Spectrogram Transformer](https://github.com/YuanGongND/ast) — Pretrained audio encoder
- [Nested Music Transformer](https://arxiv.org/abs/2408.01180) — Compound token sub-decoding
- [Beat-Aligned Spec2Seq](https://arxiv.org/abs/2311.13687) — Closest to our target architecture
- [Mapperatorinator](https://github.com/OliBomby/Mapperatorinator) — Whisper-based rhythm game mapper
- [BeatLearning](https://github.com/sedthh/BeatLearning) — Transformer for rhythm games
- [EDGE](https://github.com/Stanford-TML/EDGE) — Diffusion + transformer for dance generation

---

## Output Representation

### Recommended: Beat-Quantized Event Tokens with Compound Notes

**Scope decision:** The model generates **color notes only** — no bombs, walls, arcs, or chains. Rationale: bombs/walls are rare in training data (10-50x fewer than notes), arcs/chains are v3+ only (absent from most community maps), and color notes are the core gameplay. Additional elements can be layered on later via post-processing or model expansion.

**Beat quantization:** Snap all note times to 1/16th note subdivisions (4 slots per beat). This aligns with how maps are authored — the BSMG Wiki confirms 90%+ of notes fall on standard subdivisions. Validated as "vital" by the ISMIR 2023 paper.

**Compound tokens with fixed hand slots:** At each active beat position, emit `[LEFT_TOKEN] [RIGHT_TOKEN]` where each is either EMPTY or a compound encoding of `(x, y, direction)`.

### Token Vocabulary (~290 tokens)

| Category | Count | Description |
|----------|-------|-------------|
| Position-in-bar | 64 | 1/16th subdivisions in 4/4 time |
| BAR | 1 | Bar boundary marker |
| Left hand | 109 | 108 note configs (12 positions × 9 directions) + EMPTY |
| Right hand | 109 | Same as left hand |
| Difficulty | 5 | Easy through ExpertPlus |
| Special | 3 | START, END, PAD |

### Sequence Format Example

```
START DIFF_EXPERT BAR POS_0 LEFT_EMPTY RIGHT_2_1_DOWN POS_4 LEFT_1_1_UP RIGHT_EMPTY BAR POS_0 ...
```

### Sequence Length Estimates

| Scenario | Tokens |
|----------|--------|
| 3-min Expert (2,000 notes, 120 BPM) | ~2,500-4,000 |
| 3-min Easy (500 notes) | ~800-1,200 |
| 5-min Expert+ (4,000 notes) | ~5,000-7,000 |

All well within transformer context windows.

### Why This Approach

- **Beat quantization** is empirically validated as critical for transformers in rhythm games
- **Compound tokens** cut sequence length in half vs factored tokens (Compound Word Transformer: 5-10x faster convergence)
- **Fixed hand slots** avoid ordering ambiguity of simultaneous events
- **~305 tokens** is manageable — small embedding table, efficient softmax
- **Event-based** (no empty frames) avoids the severe sparsity of frame-based approaches

### Alternatives Considered

| Approach | Vocab | Seq Length (3min Expert) | Sparsity | Proven? |
|----------|-------|------------------------|----------|---------|
| Frame-based 20 Hz | ~34 outputs/frame | 3,600 | ~40% active | DDC, Beat Sage |
| Frame-based 50 Hz | ~34 outputs/frame | 9,000 | ~16% active | DDC |
| Event factored | ~75-225 | 8,000-10,000 | None | Music Transformer |
| **Event compound (recommended)** | **~290** | **2,500-4,000** | **None** | **Compound Word Transformer** |
| Beat-quantized factored | ~100-200 | 5,000-7,000 | None | ISMIR 2023 |

---

## Evaluation and Quality Metrics

### During Training

| Loss | Purpose | Notes |
|------|---------|-------|
| **Cross-entropy** (primary) | Next-token prediction | Standard for autoregressive transformers |
| **Focal loss** for timing | Handle class imbalance | γ=2, α tuned to positive class frequency |
| **Density regression** (auxiliary) | Match NPS to difficulty | Regularize difficulty conditioning |
| **Beat alignment** (auxiliary) | Penalize off-grid notes | Encourage musical alignment |

**Multi-task weighting:** timing > direction > position > color (suggested 4:2:1:1).

### Post-Generation Quality Score

```
Quality = w1·Onset_F1 + w2·Beat_Alignment + w3·(1-Parity_Violation_Rate)
        + w4·(1-Vision_Block_Rate) + w5·Flow_Score + w6·NPS_Accuracy
        + w7·Pattern_Diversity
```

### Metric Definitions

**Onset Alignment F1** — Treat notes as predicted onsets, audio onsets as ground truth. TP = note within ±40ms of audio onset. Standard MIR evaluation.

**Beat Alignment Score** — `mean(min_k(|t_note - t_beat_k|))` where `t_beat_k` are beat grid positions. Lower is better.

**Parity Violation Rate** — Track forehand/backhand swing state per hand. Each note implies a swing direction; violations occur when consecutive same-hand notes require the same swing type. Target: <5% for Expert+. Tools: [JoshaParity](https://github.com/Joshabi/JoshaParity), [bs-parity](https://github.com/GalaxyMaster2/bs-parity).

**Vision Block Rate** — Notes at center positions (x=1-2, y=1) that obscure subsequent notes. Target: minimize.

**NPS Accuracy** — `1 - |NPS_generated - NPS_target| / NPS_target`.

**Pattern Diversity** — Fraction of unique 4-note subsequences. Low diversity = repetitive output.

### NPS Ranges by Difficulty

| Difficulty | NPS Range | Rhythm Subdivision |
|------------|-----------|-------------------|
| Easy | 1.0 – 2.0 | 1/1 (on-beat) |
| Normal | 2.0 – 3.5 | 1/2 |
| Hard | 3.0 – 5.0 | 1/4 |
| Expert | 4.5 – 7.0 | 1/4, 1/8 |
| Expert+ | 6.0 – 10.0+ | 1/8, 1/16 |

Official hardest: "Power of the Saber Blade" at 10.66 NPS.

### Playability Heuristics (Algorithmically Detectable)

- **Vision blocks** — notes at (1,1) or (2,1) obscuring subsequent notes
- **Parity violations** — unnatural forehand/backhand sequences
- **Double directionals** — consecutive same-direction same-hand notes
- **Arm crossing** — left hand at x≥2 and right hand at x≤1 simultaneously
- **Excessive dot notes** — overuse of cut_direction=8 above Normal
- **Bomb placement** — bombs in natural follow-through swing paths

### Validation Tools

| Tool | What It Checks |
|------|----------------|
| [BS Map Check](https://github.com/KivalEvan/BeatSaber-MapCheck) | Schema validity, vision blocks, ranking criteria |
| [bs-parity](https://github.com/GalaxyMaster2/bs-parity) | Parity errors, swing resets |
| [JoshaParity](https://github.com/Joshabi/JoshaParity) | Swing prediction, parity analysis |
| [bs-analysis](https://github.com/officialMECH/bs-analysis) | NPS statistics, difficulty spread |

---

## Map Export

### Target Format: v2

v2 is the oldest supported schema and provides maximum compatibility with Beat Saber, all community tools, BeatSaver uploads, and mods. The game maintains an internal compatibility layer.

### Model Output → v2 JSON (1:1 Mapping)

| Model Output | v2 Field |
|-------------|----------|
| `beat` | `_time` |
| `x` (0-3) | `_lineIndex` |
| `y` (0-2) | `_lineLayer` |
| `color` (0-1) | `_type` |
| `cut_direction` (0-8) | `_cutDirection` |

No transformation needed — direct field rename.

### Required File Structure

```
CustomLevelFolder/
  Info.dat              ← must be exactly "Info.dat"
  song.ogg              ← OGG Vorbis, 44100 Hz, stereo
  cover.jpg             ← square, 512x512 recommended
  ExpertStandard.dat    ← one or more difficulty files
```

### Audio Requirements

- **Format:** OGG Vorbis (`.ogg`)
- **Sample rate:** 44100 Hz
- **Channels:** Stereo
- **Conversion:** `ffmpeg -i input.mp3 -c:a libvorbis -ar 44100 -q:a 6 song.ogg`
- Add ≥2 seconds silence before first note to avoid hot-start issues

### Custom Map Loading

Beat Saber **natively supports custom levels without mods**. Place folder in:
- **PC (Steam):** `Beat Saber_Data/CustomLevels/`
- **Testing:** `Beat Saber_Data/CustomWIPLevels/` (appears in WIP Maps, Practice mode only)

### Map Hash Computation

SHA-1 over concatenation of `Info.dat` bytes + all difficulty `.dat` file bytes (UTF-8). Audio file is NOT included in v2 hash.

### BeatSaver Upload

- **Zip limit:** 15 MB total
- Files must be at zip root (no containing folder)
- Cover image required for upload
- AI-generated maps must be disclosed

### Recommended NJS by Difficulty

| Difficulty | NJS |
|------------|-----|
| Easy | 10 |
| Normal | 10 |
| Hard | 12 |
| Expert | 16 |
| Expert+ | 18 |

---

## Training Data Quality Analysis

Research conducted February 2026 to determine download criteria for BeatSaver custom maps.

### BeatSaver Catalog Overview

- **Total maps:** ~114,900
- **Rating system:** Wilson score (0.0-1.0) computed from upvotes/downvotes
- **Key stats fields:** `score`, `upvotes`, `downvotes` (also `plays`, `downloads` but these return 0)
- **Automapper flag:** Boolean indicating AI-generated maps

### Score Distribution (estimated from full catalog)

| Min Score | Est. Maps | % of Total |
|-----------|-----------|------------|
| >= 0.50 | ~96,000 | 84% |
| >= 0.60 | ~83,000 | 72% |
| >= 0.65 | ~76,000 | 66% |
| >= 0.70 | ~69,000 | 60% |
| >= 0.75 | ~61,000 | 53% |
| >= 0.80 | ~50,000 | 44% |
| >= 0.85 | ~39,000 | 34% |
| >= 0.90 | ~23,000 | 20% |
| >= 0.95 | ~5,500 | 5% |

Calibration points from rating-sorted page depth scan:
- Rank 0: score 0.99, Rank 2,000: 0.962, Rank 10,000: 0.935
- Rank 20,000: 0.909, Rank 40,000: 0.847, Rank 60,000: 0.756
- Rank 80,000: 0.625, Rank 100,000: 0.472

### Score vs Engagement Correlation (3,700-map sample)

| Score Range | Median Upvotes | % with 10+ Upvotes | Interpretation |
|-------------|---------------|---------------------|----------------|
| 0.50 - 0.60 | 0 | 3.8% | Unrated/untested noise |
| 0.60 - 0.70 | 2 | 13.3% | Low engagement |
| 0.70 - 0.80 | 8 | 45.7% | Moderate quality |
| 0.80 - 0.90 | 23 | 89.8% | Good quality |
| 0.90+ | 91 | 100% | Excellent quality |

The 0.50-0.60 score bucket is a "default/unrated" zone: maps with 0-1 votes receive a score around 0.5 from the Wilson score formula. These have no quality signal.

### Upvotes Distribution (human-made maps)

| Upvotes | % of Maps |
|---------|-----------|
| 0 | 19% |
| 1-4 | 19% |
| 5-10 | 13% |
| 11-50 | 27% |
| 51-250 | 14% |
| 250+ | 8% |

### Download Criteria Decision

**Selected: score >= 0.75, upvotes >= 5, automapper = false**

Rationale:
- Filters out the unrated noise bucket (0.50-0.60 scores with zero engagement)
- Yields ~55,000 maps (each with multiple difficulties = 100K+ training examples)
- Large enough dataset for supervised pretraining
- Can fine-tune later on premium tier (score >= 0.90, ~23K maps) for polish

Three tiers defined for flexibility:

| Tier | Criteria | Est. Maps | Use Case |
|------|----------|-----------|----------|
| Primary | score >= 0.75, upvotes >= 5 | ~55,000 | Main training set |
| Strict | score >= 0.80, upvotes >= 10 | ~40,000 | Higher quality subset |
| Premium | score >= 0.90 | ~23,000 | Fine-tuning / validation |

Additional filters applied in processing (not download):
- Standard characteristic only (exclude 360/90/OneSaber)
- Must have at least one difficulty with color notes

---

## Parquet Storage Format

### Row Group Strategy
Notes, bombs, and obstacles are written to numbered Parquet files (`notes_0000.parquet`, `notes_0001.parquet`, etc.) with **one row group per song_hash**. This allows readers to push down predicates and skip irrelevant songs when loading a subset (e.g. only train-split hashes).

### File Splitting
When a Parquet file exceeds 1 GB (configurable via `max_file_bytes`), a new numbered file is started. This keeps individual files manageable for memory-mapped I/O and cloud storage. The reader (`read_notes_parquet`) transparently handles both:
- **Multi-file layout:** `notes_0000.parquet`, `notes_0001.parquet`, ...
- **Legacy single-file:** `notes.parquet` (backward compat)

### Performance Profile (214 official songs, 907K notes)
- Full table load: ~1s, ~100MB RAM (happens once at training start)
- Current dataset fits in a single file (~5MB compressed)
- Splitting becomes relevant at ~50K+ songs with millions of notes

### Baseline Training Results (5 epochs, 214 official songs)

| Config | Params | Epoch Time | Val Loss | Val Acc |
|--------|--------|-----------|----------|---------|
| Default (6L/512d) | 44.5M | ~456s | 2.7457 | 40.5% |
| Small (2L/128d) | 1.0M | ~15s | 2.7379 | 41.5% |

The small model matches the large model's quality on this dataset size, suggesting data volume (not model capacity) is the current bottleneck.

---

## Full-Dataset Training (23K Songs)

### Dataset Scale
- **23,588 songs** in audio manifest (23,375 BeatSaver custom + 213 official)
- **54,771 beatmaps** processed (multiple difficulties per song)
- **40.3M notes** in Parquet (notes_0000.parquet, ~600MB)
- **42,542 training samples**, 5,282 validation samples (80/10/10 split by song hash)
- **206 samples filtered** — mapping extension maps with all notes outside standard 4x3 grid

### Data Pipeline Issues Encountered

**v3.3.0 compact format:** Newer BeatSaver maps (v3.3.0+) omit JSON keys that have default values. Parser crashed with `KeyError: 'y'`. Fixed by using `.get()` with defaults in `v3.py` for all note, bomb, and obstacle fields.

**Mapping extensions (Noodle Extensions):** Some maps use coordinates far outside the standard grid (e.g., x=1000, y=3000) for visual effects. These overflow int8 in Arrow/Parquet. Fixed with `_clamp8()` in `writer.py` to clamp to [-128, 127], then filtered in `dataset.py` pre-tokenization (only keep notes where 0 <= x < 4, 0 <= y < 3, 0 <= cut_direction < 9, color in {0, 1}).

**Difficulty name inconsistencies:** BeatSaver maps use various casings (`"normal"`, `"Expert+"` instead of `"Normal"`, `"ExpertPlus"`). Fixed with case-insensitive lookup + alias map in `tokenizer.py`.

### Mel Spectrogram Pre-Caching

Computing mel spectrograms during training's first epoch was a severe bottleneck: ~35 songs/min single-threaded vs 18K+ unique songs = ~8 hours for first epoch.

**Solution:** `warm_mel_cache()` in `dataset.py`:
- Uses `ProcessPoolExecutor` (up to 8 workers) to compute mel spectrograms before training starts
- Cache key: `{song_hash}_{bpm}.npy` in `data/processed/mel_cache/`
- ~500 songs/min → 14K songs in ~25 minutes (14x faster than in-loop)
- Once warm, subsequent training starts in ~3 minutes (dataset init only)
- Cache files persist across runs — only new songs need computation

### Memory Optimization

**VRAM:** batch_size=128 maxed out 8GB VRAM (cross-attention maps scale as batch × heads × seq_len × audio_len). batch_size=32 uses ~4-5GB with comfortable headroom.

**RAM:** The dataset stored 42K samples' raw note dicts (list of dicts with 8 keys each) in memory alongside pre-tokenized tokens. Freed raw note dicts after pre-tokenization with `sample.pop("notes", None)`, reducing RAM from ~100% to ~75%.

**DataFrame cleanup:** Explicitly `del df, table, grouped` after pandas groupby loop to free the 40M-row DataFrame and Arrow table immediately.

### Training Results (Small Config: 1M params, batch_size=32)

| Epoch | Train Loss | Val Loss | Val Acc | Notes |
|-------|-----------|----------|---------|-------|
| 1 | 3.163 | 3.047 | 35.8% | |
| 2 | 2.644 | 2.807 | 44.5% | |
| 3 | 2.467 | 2.657 | 48.3% | |
| 4 | 2.374 | 2.609 | 49.2% | |
| 5 | 2.334 | 2.603 | 49.4% | End of first run |
| 6 | 2.263 | 2.319 | 55.6% | Resumed; LR scheduler reset |
| 7 | 2.158 | 2.184 | 57.4% | |
| 8 | 2.114 | 2.216 | 57.4% | Val loss bounce |
| 9 | 2.091 | 2.191 | 57.9% | |
| 10 | 2.083 | 2.124 | 59.0% | |
| 11 | 2.082 | 2.099 | 59.8% | |
| 12 | 2.091 | 2.063 | 60.4% | |
| **13** | **2.085** | **2.055** | **60.6%** | **Best model** |
| 14 | NaN | NaN | 0.0% | Diverged (LR too high) |

**Key observations:**
- 60.6% token accuracy on 291-vocab task is solid for a 1M param model
- Epochs 6-13 benefited from LR scheduler restarting on resume (fresh warmup)
- Training diverged at epoch 14 — cosine LR schedule hit a problematic region after warmup
- Train loss ≈ val loss throughout (2.08 vs 2.06 at best) — no overfitting, thanks to label smoothing + dropout
- Each epoch: ~590s (9.8 min), 1,330 batches at batch_size=32, 71 samples/s

### Checkpoint Resume Bug: Missing Training State

Resuming from a checkpoint caused NaN on the first epoch. Three pieces of training state were not saved:

1. **GradScaler** (root cause of NaN): Fresh scaler starts at scale=65536. During training, the scaler backs off to a much lower value. On resume, the 100x+ scale jump causes overflow in mixed-precision backward passes → NaN immediately.

2. **LR Scheduler**: Cosine schedule restarts from warmup, ramping LR to 3e-4 when the model was running at ~1.2e-4. This compounds the scaler issue.

3. **Checkpoint overwrite hazard**: Resuming from `epoch_013` retrains epoch 12, which saves back to `epoch_013` — overwriting the original clean checkpoint with NaN weights. Always resume from `best/` (only overwritten when val_loss improves).

**Fix (implemented):**
- `save_checkpoint()` now saves `scheduler.pt` and `scaler.pt` alongside model/optimizer
- `load_checkpoint()` restores scaler state; falls back to `init_scale=1.0` for old checkpoints
- `restore_scheduler()` restores scheduler state; falls back to fast-forwarding to `global_step`

### Resumed Training Results (after checkpoint fix)

Resumed from best checkpoint (epoch 11, val_loss=2.055) with conservative scaler (init_scale=1.0) and fast-forwarded LR scheduler:

| Epoch | Train Loss | Val Loss | Val Acc | Notes |
|-------|-----------|----------|---------|-------|
| 12 | 2.323 | 2.403 | 60.3% | Resumed; recovering |
| 13 | 2.428 | 2.162 | 58.8% | |
| 14 | 2.277 | 2.127 | 59.3% | Past previous NaN point |
| 15 | 2.235 | 2.113 | 59.6% | |
| 16 | 2.190 | 2.080 | 60.0% | |
| 16 (re) | 2.145 | 2.063 | 60.3% | Clean resume with scaler.pt |

**Conclusion:** Model plateaued at val_loss ~2.06, matching the original best of 2.055. The 1M param model has saturated on 42K samples. Further improvement requires scaling up model size.

### Grammar Constraint: Strictly Increasing POS

The model was generating multiple notes of the same color at the same beat (e.g., two red notes pointing in opposite directions = impossible to hit). Root cause: the grammar mask allowed any POS token after a RIGHT token, including repeats of the same position.

**Fix:** Track `last_pos_in_bar` during generation. After a RIGHT token, only allow POS tokens with offset strictly greater than the last one in the current bar. Reset on BAR token.

**Result:** Zero duplicate same-color-same-beat notes. Also reduced NPS from 10.6 to 4.5 (more realistic) and extended generated duration from 35s to 79s.

### Generation Quality (Epoch 13 Best Model)

Generated "Boom Kitty - Bassgasm" on Expert difficulty:
- 352 notes, 4.5 NPS, 79 seconds
- Playable in-game — verified by user
- **Issues:** Color imbalance (77% red, 23% blue), audio truncated to ~95s (max_audio_len=4096)
- **Strengths:** Uses full grid, correct note structure, reasonable difficulty spread
