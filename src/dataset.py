from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.audio_io import DEFAULT_SAMPLE_RATE, load_audio
from src.features import extract_features, feature_names, vectorize_features


@dataclass(frozen=True)
class AudioExample:
    path: Path
    label: str
    source: str


def load_manifest(manifest_path: str | Path) -> list[AudioExample]:
    """Load a CSV manifest with path,label,source columns."""
    manifest = Path(manifest_path)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    examples: list[AudioExample] = []
    with manifest.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "label", "source"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(f"Manifest missing required columns: {missing_cols}")

        for row in reader:
            audio_path = Path(row["path"])
            if not audio_path.is_absolute():
                audio_path = Path.cwd() / audio_path

            examples.append(
                AudioExample(
                    path=audio_path,
                    label=row["label"].strip(),
                    source=row["source"].strip(),
                )
            )

    if not examples:
        raise ValueError(f"Manifest has no examples: {manifest}")

    return examples


def build_feature_matrix(
    examples: list[AudioExample],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load audio examples and return X, y, feature_names."""
    rows: list[np.ndarray] = []
    labels: list[str] = []
    names: list[str] | None = None

    for example in examples:
        waveform, sr = load_audio(example.path, sample_rate=sample_rate)
        features = extract_features(waveform, sample_rate=sr)

        if names is None:
            names = feature_names(features)

        rows.append(vectorize_features(features, names))
        labels.append(example.label)

    return np.vstack(rows), np.asarray(labels), names or []
