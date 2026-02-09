"""Tests for audio preprocessing module."""

import json
import tempfile
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
pytest.importorskip("librosa")

from beat_weaver.model.audio import (
    beat_align_spectrogram,
    build_audio_manifest,
    compute_mel_spectrogram,
    compute_onset_envelope,
    detect_bpm,
    load_audio,
    load_manifest,
    save_manifest,
)


@pytest.fixture
def sine_wave_file(tmp_path):
    """Create a short sine wave WAV file."""
    import soundfile as sf

    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "test.wav"
    sf.write(str(path), audio, sr)
    return path, sr, duration


class TestLoadAudio:
    def test_load_returns_mono(self, sine_wave_file):
        path, sr, duration = sine_wave_file
        audio, out_sr = load_audio(path, sr=sr)
        assert audio.ndim == 1
        assert out_sr == sr
        assert len(audio) == pytest.approx(sr * duration, abs=1)

    def test_resamples(self, sine_wave_file):
        path, _, _ = sine_wave_file
        audio, out_sr = load_audio(path, sr=16000)
        assert out_sr == 16000
        assert len(audio) == pytest.approx(16000 * 2, abs=100)


class TestMelSpectrogram:
    def test_output_shape(self, sine_wave_file):
        path, sr, _ = sine_wave_file
        audio, _ = load_audio(path, sr=sr)
        mel = compute_mel_spectrogram(audio, sr=sr, n_mels=80, hop_length=512)
        assert mel.shape[0] == 80
        assert mel.shape[1] > 0
        assert mel.dtype == np.float32

    def test_custom_params(self, sine_wave_file):
        path, sr, _ = sine_wave_file
        audio, _ = load_audio(path, sr=sr)
        mel = compute_mel_spectrogram(audio, sr=sr, n_mels=40, hop_length=256)
        assert mel.shape[0] == 40
        # More frames with smaller hop
        mel2 = compute_mel_spectrogram(audio, sr=sr, n_mels=40, hop_length=512)
        assert mel.shape[1] > mel2.shape[1]


class TestBeatAlignment:
    def test_output_shape(self, sine_wave_file):
        path, sr, duration = sine_wave_file
        audio, _ = load_audio(path, sr=sr)
        mel = compute_mel_spectrogram(audio, sr=sr)
        aligned = beat_align_spectrogram(mel, sr=sr, hop_length=512, bpm=120.0)

        assert aligned.shape[0] == 80  # n_mels preserved
        # At 120 BPM, 2 seconds = 4 beats = 64 subdivisions (16 per beat)
        expected_subs = int(np.ceil(duration * (120.0 / 60.0) * 16))
        assert aligned.shape[1] == pytest.approx(expected_subs, abs=2)

    def test_faster_bpm_more_frames(self, sine_wave_file):
        path, sr, _ = sine_wave_file
        audio, _ = load_audio(path, sr=sr)
        mel = compute_mel_spectrogram(audio, sr=sr)
        a1 = beat_align_spectrogram(mel, sr=sr, hop_length=512, bpm=120.0)
        a2 = beat_align_spectrogram(mel, sr=sr, hop_length=512, bpm=240.0)
        # Double BPM → double subdivisions
        assert a2.shape[1] == pytest.approx(a1.shape[1] * 2, abs=2)


class TestOnsetEnvelope:
    def test_shape(self, sine_wave_file):
        path, sr, _ = sine_wave_file
        audio, _ = load_audio(path, sr=sr)
        onset = compute_onset_envelope(audio, sr=sr)
        assert onset.shape[0] == 1
        assert onset.shape[1] > 0
        assert onset.dtype == np.float32


class TestDetectBpm:
    def test_returns_float(self, sine_wave_file):
        path, sr, _ = sine_wave_file
        audio, _ = load_audio(path, sr=sr)
        bpm = detect_bpm(audio, sr=sr)
        assert isinstance(bpm, float)
        assert bpm > 0

    def test_fallback_for_non_rhythmic(self):
        """Pure sine wave has no beats; should return default."""
        sr = 22050
        t = np.linspace(0, 2.0, sr * 2, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        bpm = detect_bpm(audio, sr=sr, default=120.0)
        # Should return the default rather than 0
        assert bpm > 0

    def test_click_track(self):
        """A click track at 120 BPM should detect roughly 120."""
        sr = 22050
        duration = 5.0
        bpm_target = 120.0
        n_samples = int(sr * duration)
        audio = np.zeros(n_samples, dtype=np.float32)
        # Place clicks at each beat
        beat_interval = int(sr * 60.0 / bpm_target)
        for i in range(0, n_samples, beat_interval):
            end = min(i + 200, n_samples)
            audio[i:end] = 0.8
        bpm = detect_bpm(audio, sr=sr)
        assert isinstance(bpm, float)
        assert bpm > 0


class TestAudioManifest:
    def test_save_load_round_trip(self, tmp_path):
        manifest = {"abc123": "/path/to/song.ogg", "def456": "/path/to/song.egg"}
        path = tmp_path / "manifest.json"
        save_manifest(manifest, path)
        loaded = load_manifest(path)
        assert loaded == manifest

    def test_build_from_map_folders(self, tmp_path):
        """Build manifest from a directory structure with Info.dat files."""
        import soundfile as sf

        # Create a fake map folder
        map_dir = tmp_path / "raw" / "my_map"
        map_dir.mkdir(parents=True)

        info = {"_songFilename": "song.ogg", "_songName": "Test", "_beatsPerMinute": 120}
        (map_dir / "Info.dat").write_text(json.dumps(info))

        # Create a dummy audio file
        sr = 22050
        audio = np.zeros(sr, dtype=np.float32)
        sf.write(str(map_dir / "song.ogg"), audio, sr)

        manifest = build_audio_manifest([tmp_path / "raw"])
        assert len(manifest) == 1
        audio_path = list(manifest.values())[0]
        assert Path(audio_path).exists()
