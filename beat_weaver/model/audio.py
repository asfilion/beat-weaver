"""Audio preprocessing — mel spectrograms and beat-aligned framing.

Depends on librosa and soundfile (optional ML dependencies).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def load_audio(path: Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load an audio file and resample to target sample rate.

    Returns (audio_mono, sample_rate) where audio_mono is float32 1-D.
    """
    path = Path(path)
    audio, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)
    # Mix to mono
    audio = audio.mean(axis=1)
    # Resample if needed
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    return audio, sr


def compute_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 22050,
    n_mels: int = 80,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute log-mel spectrogram.

    Returns float32 array of shape (n_mels, T).
    """
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, window="hann",
    )
    # Convert to dB scale (log-magnitude), ref=max
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def beat_align_spectrogram(
    mel: np.ndarray,
    sr: int,
    hop_length: int,
    bpm: float,
    subdivisions_per_beat: int = 16,
) -> np.ndarray:
    """Resample spectrogram frames to align with beat subdivisions.

    Each output frame corresponds to one 1/16th note position.

    Args:
        mel: Log-mel spectrogram of shape (n_mels, T_frames).
        sr: Audio sample rate.
        hop_length: STFT hop length used for mel.
        bpm: Beats per minute.
        subdivisions_per_beat: Number of subdivisions per beat (default 16).

    Returns:
        Float32 array of shape (n_mels, T_beats) where T_beats is the total
        number of beat subdivisions covered by the audio.
    """
    n_mels, n_frames = mel.shape

    # Time of each spectrogram frame
    frame_times = librosa.frames_to_time(
        np.arange(n_frames), sr=sr, hop_length=hop_length,
    )

    # Duration of the audio
    duration = frame_times[-1] if n_frames > 0 else 0.0

    # Total beat subdivisions
    beats_per_second = bpm / 60.0
    subs_per_second = beats_per_second * subdivisions_per_beat
    total_subs = int(np.ceil(duration * subs_per_second))

    if total_subs == 0:
        return np.zeros((n_mels, 0), dtype=np.float32)

    # Time of each subdivision
    sub_times = np.arange(total_subs) / subs_per_second

    # Interpolate: for each sub_time, find the nearest frame
    # Use linear interpolation across the time axis
    frame_indices = np.interp(sub_times, frame_times, np.arange(n_frames))

    # Interpolate each mel bin
    aligned = np.zeros((n_mels, total_subs), dtype=np.float32)
    x_coords = np.arange(n_frames)
    for i in range(n_mels):
        aligned[i] = np.interp(frame_indices, x_coords, mel[i])

    return aligned


def detect_bpm(
    audio: np.ndarray, sr: int = 22050, default: float = 120.0,
) -> float:
    """Estimate the BPM of an audio signal using librosa beat tracking.

    Returns the estimated tempo as a float.  Falls back to *default*
    (120 BPM) when beat tracking cannot determine a tempo (e.g. for very
    short or non-rhythmic audio).
    """
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    # librosa may return an array; extract scalar
    if hasattr(tempo, "__len__"):
        tempo = float(tempo[0]) if len(tempo) > 0 else default
    tempo = float(tempo)
    if tempo <= 0:
        return default
    return tempo


def compute_onset_envelope(
    audio: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute onset strength envelope.

    Returns float32 array of shape (1, T).
    """
    onset = librosa.onset.onset_strength(
        y=audio, sr=sr, hop_length=hop_length,
    )
    return onset.astype(np.float32).reshape(1, -1)


# ── Audio manifest ──────────────────────────────────────────────────────────

_AUDIO_EXTENSIONS = {".ogg", ".egg", ".wav", ".mp3", ".flac"}


def _hash_folder(folder: Path) -> str:
    """Compute a content hash for a map folder.

    Uses the same algorithm as ``beat_weaver.pipeline.processor.compute_map_hash``
    so the audio manifest keys match the song hashes in the Parquet data.
    """
    from beat_weaver.pipeline.processor import compute_map_hash

    return compute_map_hash(folder)


def build_audio_manifest(raw_dirs: list[Path]) -> dict[str, str]:
    """Scan raw map folders and build a hash → audio file path mapping.

    Looks for Info.dat files and their referenced audio filenames.
    Falls back to scanning for common audio extensions.
    """
    manifest: dict[str, str] = {}

    for raw_dir in raw_dirs:
        raw_dir = Path(raw_dir)
        if not raw_dir.exists():
            logger.warning("Raw directory not found: %s", raw_dir)
            continue

        for info_file in raw_dir.rglob("Info.dat"):
            folder = info_file.parent
            song_hash = _hash_folder(folder)

            # Try to find audio file referenced in Info.dat
            audio_path = _find_audio_in_folder(folder, info_file)
            if audio_path:
                manifest[song_hash] = str(audio_path)

        # Also check for case-insensitive info.dat
        for info_file in raw_dir.rglob("info.dat"):
            if info_file.name == "Info.dat":
                continue  # already handled above
            folder = info_file.parent
            song_hash = _hash_folder(folder)
            if song_hash not in manifest:
                audio_path = _find_audio_in_folder(folder, info_file)
                if audio_path:
                    manifest[song_hash] = str(audio_path)

    logger.info("Built audio manifest: %d entries", len(manifest))
    return manifest


def _find_audio_in_folder(folder: Path, info_file: Path) -> Path | None:
    """Find the audio file in a map folder."""
    # Try parsing Info.dat for the audio filename
    try:
        import json as _json
        info = _json.loads(info_file.read_text(encoding="utf-8-sig"))
        # v2/v3: _songFilename, v4: audio.songFilename or song.songFilename
        audio_name = (
            info.get("_songFilename")
            or info.get("audio", {}).get("songFilename")
            or info.get("song", {}).get("songFilename")
        )
        if audio_name:
            audio_path = folder / audio_name
            if audio_path.exists():
                return audio_path
    except Exception:
        pass

    # Fallback: scan for common audio files
    for ext in _AUDIO_EXTENSIONS:
        for f in folder.glob(f"*{ext}"):
            return f
    return None


def save_manifest(manifest: dict[str, str], path: Path) -> None:
    """Save audio manifest to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def load_manifest(path: Path) -> dict[str, str]:
    """Load audio manifest from JSON."""
    return json.loads(Path(path).read_text())
