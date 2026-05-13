# Audio Integrity Lab

Audio Integrity Lab is an audio fraud detection prototype for identifying synthetic, cloned, replayed, or otherwise suspicious voice submissions in user-generated audio workflows.

The project is motivated by a product trust problem: as platforms expand from image-based inputs into voice, they need a way to flag potentially fraudulent audio before it affects users, moderation systems, or downstream automation. The goal is not to claim perfect deepfake detection. The goal is to build a defensible baseline system with clear metrics, documented tradeoffs, and a demo that explains why an audio clip was considered risky.

## Current Status

This repository currently contains an audio exploration script that:

- generates a test WAV file
- computes a mel spectrogram
- saves a waveform/spectrogram visualization
- loads a pretrained Wav2Vec2 model for speech embeddings

The next milestone is turning this exploration into an end-to-end baseline detector:

```text
audio file -> preprocessing -> acoustic features / speech embeddings -> classifier -> risk score + explanation
```

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
  samples/              Small local sample audio files
docs/
  research_writeup.md   Living design and research notes
notebooks/
  01_audio_basics.py    Initial audio feature and Wav2Vec2 exploration
src/                    Detector implementation will live here
```

## Roadmap

1. Build audio loading and normalization utilities.
2. Create a labeled dataset manifest for real and synthetic clips.
3. Extract baseline acoustic features.
4. Train and evaluate a simple classifier.
5. Add Wav2Vec2 embedding features.
6. Build an interactive demo with risk scoring and spectrogram visualization.
7. Document results, limitations, and future work.
