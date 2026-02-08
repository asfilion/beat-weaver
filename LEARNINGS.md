# Beat Saber Map Format Research

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
