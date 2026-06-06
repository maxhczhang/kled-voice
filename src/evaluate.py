from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib

from src.audio_io import load_audio
from src.features import extract_features, vectorize_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Score one audio file with a trained baseline model.")
    parser.add_argument("audio_path", help="Path to an audio file to score.")
    parser.add_argument("--model", default="models/baseline.joblib", help="Path to a trained model bundle.")
    args = parser.parse_args()

    bundle_path = Path(args.model)
    if not bundle_path.exists():
        raise SystemExit(f"Model not found: {bundle_path}. Train one with src/train.py first.")

    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    names = bundle["feature_names"]

    start = time.perf_counter()
    waveform, sr = load_audio(args.audio_path)
    features = extract_features(waveform, sample_rate=sr)
    X = vectorize_features(features, names).reshape(1, -1)
    prediction = model.predict(X)[0]
    elapsed_ms = (time.perf_counter() - start) * 1_000

    label = label_encoder.inverse_transform([prediction])[0]
    print(f"label: {label}")
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        for class_name, probability in zip(label_encoder.classes_, probabilities):
            print(f"probability_{class_name}: {probability:.4f}")
    print(f"latency_ms: {elapsed_ms:.2f}")


if __name__ == "__main__":
    main()
