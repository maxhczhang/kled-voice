from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


DEFAULT_SAMPLE_RATE = 16_000


def load_audio(path: str | Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32 waveform at the target sample rate."""
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    waveform, loaded_sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    waveform = np.asarray(waveform, dtype=np.float32)
    waveform = normalize_audio(waveform)
    return waveform, loaded_sr


def normalize_audio(waveform: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Peak-normalize audio while preserving silence."""
    if waveform.ndim != 1:
        waveform = np.mean(waveform, axis=0)

    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak < eps:
        return waveform.astype(np.float32)

    return (waveform / peak).astype(np.float32)
