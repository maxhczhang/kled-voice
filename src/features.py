from __future__ import annotations

import numpy as np
import librosa

from src.audio_io import DEFAULT_SAMPLE_RATE


def extract_features(
    waveform: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_mfcc: int = 20,
    n_mels: int = 64,
) -> dict[str, float]:
    """Extract a compact set of baseline acoustic features."""
    if waveform.size == 0:
        raise ValueError("Cannot extract features from an empty waveform")

    features: dict[str, float] = {}

    mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=n_mfcc)
    for idx, row in enumerate(mfcc, start=1):
        features[f"mfcc_{idx:02d}_mean"] = float(np.mean(row))
        features[f"mfcc_{idx:02d}_std"] = float(np.std(row))

    mel = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    _add_summary(features, "mel_db", mel_db)

    centroid = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)
    rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sample_rate)
    zcr = librosa.feature.zero_crossing_rate(waveform)
    rms = librosa.feature.rms(y=waveform)

    _add_summary(features, "spectral_centroid", centroid)
    _add_summary(features, "spectral_rolloff", rolloff)
    _add_summary(features, "zero_crossing_rate", zcr)
    _add_summary(features, "rms_energy", rms)

    duration = waveform.size / sample_rate
    features["duration_seconds"] = float(duration)
    features["peak_amplitude"] = float(np.max(np.abs(waveform)))

    return features


def feature_names(features: dict[str, float]) -> list[str]:
    """Return deterministic feature ordering for vectorization."""
    return sorted(features)


def vectorize_features(features: dict[str, float], names: list[str]) -> np.ndarray:
    """Convert a feature dictionary into a dense vector using a fixed order."""
    return np.asarray([features[name] for name in names], dtype=np.float32)


def _add_summary(features: dict[str, float], prefix: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float32)
    features[f"{prefix}_mean"] = float(np.mean(values))
    features[f"{prefix}_std"] = float(np.std(values))
    features[f"{prefix}_min"] = float(np.min(values))
    features[f"{prefix}_max"] = float(np.max(values))
