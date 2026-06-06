# Project State

## Current Goal

Build a useful demo that scores uploaded voice clips for synthetic/fraud risk, focused on low-resource languages from Kled's voice launch: Malay, Indonesian, and Filipino/Tagalog.

## Why This Exists

This is both a product-minded audio fraud prototype and a pitch/portfolio artifact. The sharp version of the project is: benchmark and improve synthetic speech detection for low-resource voice marketplace uploads, then package the results as a repo, writeup, and demo.

## Current Status

- Baseline audio pipeline exists in `src/`.
- `src.audio_io` loads, resamples, converts to mono, and normalizes audio.
- `src.features` extracts baseline acoustic features: MFCC stats, mel-spectrogram stats, spectral centroid, rolloff, zero-crossing rate, RMS energy, duration, and peak amplitude.
- `src.dataset` loads a CSV manifest and builds a feature matrix.
- `src.train` trains logistic regression or random forest baselines and prints metrics.
- `src.evaluate` scores one audio file with a trained model.
- `data/manifest.csv` currently has only one generated test tone labeled `synthetic`.
- Training is blocked on adding real and synthetic speech examples.

## Next Tasks

- [ ] Expand `data/manifest.csv` schema beyond `path,label,source`.
- [ ] Add a small real speech sample set.
- [ ] Add a small synthetic speech sample set.
- [ ] Train the first baseline classifier.
- [ ] Record baseline metrics in `README.md` and `docs/research_writeup.md`.
- [ ] Add one pretrained synthetic speech detector benchmark.
- [ ] Build a Streamlit demo for upload, playback, spectrogram, risk score, and review decision.

## Suggested Manifest Schema

```text
path,label,language,source,split,text,synthetic_engine,speaker_id
```

Suggested labels:

- `real`
- `synthetic`

Suggested languages:

- `en`
- `id`
- `ms`
- `tl`

## Last Verified

Command:

```bash
python -m src.train --manifest data/manifest.csv
```

Result:

- Loaded 1 example with label `synthetic`.
- Extracted 62 acoustic features.
- Exited with the expected message because at least two labels are required to train a classifier.

## Useful Commands

Train logistic regression baseline:

```bash
python -m src.train --manifest data/manifest.csv --model-type logistic_regression
```

Train random forest baseline:

```bash
python -m src.train --manifest data/manifest.csv --model-type random_forest
```

Score one file after training:

```bash
python -m src.evaluate path/to/audio.wav --model models/baseline.joblib
```

## Decisions

- Keep project-state tracking in this repo rather than only in Obsidian.
- Use Obsidian for motivation, pitch strategy, and higher-level reflection.
- Use this file as the operational handoff: read it first at the start of future sessions and update it at the end.
- Use Streamlit before building a custom frontend.
- Do not commit full datasets. Keep large data under ignored directories such as `data/common_voice/`, `data/asvspoof/`, and `data/synthetic/`.

## Blockers

- Need labeled real and synthetic speech data before model training produces meaningful metrics.
- Need to decide how synthetic target-language samples will be generated: ElevenLabs, OpenAI TTS, Coqui/XTTS, or another source.
- Need dataset access plan for real speech samples, likely Common Voice with small streamed/subset downloads.
