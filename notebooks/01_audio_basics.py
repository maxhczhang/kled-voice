import torch
import torchaudio
import librosa
import matplotlib.pyplot as plt
import numpy as np

print(">>> all imports done", flush=True)

# Generate a 3-second sine wave at 440Hz, sample rate 16000
sr = 16000
duration = 3.0
freq = 440
t = np.linspace(0, duration, int(sr * duration), endpoint=False)
y = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
print(f">>> generated test audio: {len(y)} samples at {sr}Hz", flush=True)

# Save it so we have a real file
import soundfile as sf
sample_path = "data/samples/test_tone.wav"
sf.write(sample_path, y, sr)
print(f">>> saved to {sample_path}", flush=True)

# Load it back with soundfile (simpler, no extra deps)
waveform, sr_loaded = sf.read(sample_path)
print(f">>> soundfile loaded: shape={waveform.shape}, sr={sr_loaded}", flush=True)

# Compute mel spectrogram with librosa
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
print(f">>> mel spectrogram shape: {mel_spec_db.shape}", flush=True)

# Save a plot
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
axes[0].plot(t, y)
axes[0].set_title("Waveform (440Hz sine wave)")
axes[0].set_xlabel("Time (s)")
img = axes[1].imshow(mel_spec_db, aspect="auto", origin="lower", cmap="viridis")
axes[1].set_title("Mel spectrogram")
plt.colorbar(img, ax=axes[1], format="%+2.0f dB")
plt.tight_layout()
plt.savefig("notebooks/01_first_audio.png", dpi=100)
plt.close()
print(">>> saved plot to notebooks/01_first_audio.png", flush=True)

# Now run it through Wav2Vec2
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
print(">>> loading Wav2Vec2 (downloads ~360MB on first run)...", flush=True)
extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
print(">>> Wav2Vec2 loaded", flush=True)

inputs = extractor(y, sampling_rate=sr, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
print(f">>> Wav2Vec2 embeddings shape: {outputs.last_hidden_state.shape}", flush=True)
print(">>> SUCCESS", flush=True)