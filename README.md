# Audio Integrity Lab

Audio Integrity Lab is an audio fraud detection prototype for identifying synthetic, cloned, replayed, or otherwise suspicious voice submissions in user-generated audio workflows.

The project is motivated by a product trust problem: as platforms expand from image-based inputs into voice, they need a way to flag potentially fraudulent audio before it affects users, moderation systems, or downstream automation. The goal is not to claim perfect deepfake detection. The goal is to build a defensible baseline system with clear metrics, documented tradeoffs, and a demo that explains why an audio clip was considered risky.

## Current Status

This repository currently contains an audio exploration script that:

- generates a test WAV file
- computes a mel spectrogram
- saves a waveform/spectrogram visualization
- loads a pretrained Wav2Vec2 model for speech embeddings

It also includes the first baseline detector pipeline:

- audio loading, mono conversion, resampling, and normalization
- acoustic feature extraction
- CSV manifest loading
- sklearn baseline training and evaluation scripts

The current pipeline shape is:

```text
audio file -> preprocessing -> acoustic features / speech embeddings -> classifier -> risk score + explanation
```

The starter manifest only contains one generated test-tone sample, so it is not enough to train a real classifier yet. The next milestone is adding real and synthetic speech examples.

## Planned System

Audio Integrity Lab will score uploaded audio using a mix of classical acoustic features and pretrained speech representations.

Initial feature set:

- MFCC statistics
- mel-spectrogram statistics
- spectral centroid and rolloff
- zero-crossing rate
- RMS energy
- Wav2Vec2 embeddings

Initial model targets:

- logistic regression baseline
- random forest baseline
- calibrated risk score for review workflows

Evaluation targets:

- accuracy
- precision and recall
- F1 score
- ROC-AUC
- confusion matrix
- inference latency

## Product Framing

The intended output is a trust-and-safety style risk assessment, not a hard automatic ban decision. A production version of this kind of system should support human review, threshold tuning, and continuous retraining as synthetic voice models improve.

The demo will aim to show:

- audio upload and playback
- predicted label: authentic, suspicious, or synthetic
- fraud risk score
- spectrogram visualization
- explanation signals for why the clip was flagged

## Repository Layout

```text
data/
  manifest.csv          Starter labeled manifest
  samples/              Small local sample audio files
docs/
  research_writeup.md   Living design and research notes
notebooks/
  01_audio_basics.py    Initial audio feature and Wav2Vec2 exploration
src/
  audio_io.py           Audio loading and normalization
  dataset.py            Manifest loading and feature matrix construction
  evaluate.py           Single-file scoring with a trained model
  features.py           Baseline acoustic feature extraction
  train.py              Baseline model training and metrics
```

## Usage

Extract features and attempt baseline training from the starter manifest:

```bash
python -m src.train --manifest data/manifest.csv
```

The command currently exits with a clear message because the starter manifest has only one label. Once real and synthetic speech examples are added, it will train a baseline model and save it to `models/baseline.joblib`.

Score a file after training:

```bash
python -m src.evaluate path/to/audio.wav --model models/baseline.joblib
```

## Roadmap

1. Add real and synthetic speech clips to the dataset manifest.
2. Train and evaluate the first simple classifier.
3. Report baseline metrics in the README and research writeup.
4. Add Wav2Vec2 embedding features.
5. Build an interactive demo with risk scoring and spectrogram visualization.
6. Document results, limitations, and future work.
