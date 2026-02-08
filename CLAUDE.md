# Beat Saber Track Generator

An AI-powered mod for Beat Saber that generates custom note maps from audio files.

## Project Overview

Given an audio file as input, the system generates block positions and orientations (for both left and right sabers) at a regular sample rate (~20 Hz). The goal is a machine learning model that produces playable, enjoyable Beat Saber tracks automatically.

## Key Components

### 1. Beat Saber Map Format Research
- Reverse-engineer how the game stores official/purchased song maps
- Understand the data format for block positions, orientations, timing, and difficulty levels
- This data becomes the foundation for training data

### 2. Custom Track Loading
- Figure out how to create custom tracks that load into Beat Saber
- Support the standard custom map format so generated tracks are playable in-game

### 3. ML Model for Track Generation
- **Input:** Audio file (waveform, spectrogram features, etc.)
- **Output:** Sequence of block positions and orientations per timestep
- **Model type:** TBD
- **Design options:**
  - Random seed for repeatable tracks
  - Randomized generation for a unique experience every play

### 4. Training Data Pipeline
- **Primary sources:** Official store maps + community-created custom maps
- **Challenge:** Likely insufficient training data from existing maps alone
- **Solution:** Use the feedback capture system (below) to generate additional labeled data

### 5. Feedback Capture System (Later Phase)
- In-game mechanism to collect player feedback on generated tracks
- Enables continuous improvement of the model through real gameplay data
- Additional source of training data beyond what's publicly available

## Open Questions

### Scoring and Labeling Strategy
- How do we score/evaluate a generated track?
- How do we label training data effectively?
- **Windowed evaluation:** Score individual segments of a track rather than the whole track
  - A single track may have good sequences and bad sequences
  - Segment-level labeling lets us mark portions as good/bad independently
  - More granular labels = better training signal

### Model Architecture
- What features to extract from audio?
- What model architecture is best suited (sequence-to-sequence, transformer, etc.)?
- What sample rate / temporal resolution for block placement?
