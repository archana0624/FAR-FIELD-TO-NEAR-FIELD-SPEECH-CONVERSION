import os
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import soundfile as sf
import matplotlib.pyplot as plt

# Set parameters
MAX_EPOCHS = 10
BATCH_SIZE = 16
SR = 16000  # Sampling rate
N_FFT = 1024  # FFT size
HOP_LENGTH = 512  # Hop length

# Data loader with normalization
def load_audio_data(far_field_dir, near_field_dir):
    far_field_files = sorted(os.listdir(far_field_dir))
    near_field_files = sorted(os.listdir(near_field_dir))
    far_spectrograms, near_spectrograms = [], []

    for ff, nf in zip(far_field_files, near_field_files):
        ff_path, nf_path = os.path.join(far_field_dir, ff), os.path.join(near_field_dir, nf)
        ff_audio, _ = librosa.load(ff_path, sr=SR)  # Load MP3 far-field audio
        nf_audio, _ = librosa.load(nf_path, sr=SR)  # Load MP3 near-field audio

        # Convert to spectrogram
        ff_spec = librosa.stft(ff_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
        nf_spec = librosa.stft(nf_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)

        epsilon = 1e-6
        far_spectrograms.append(np.abs(ff_spec) / (np.max(np.abs(ff_spec)) + epsilon))
        near_spectrograms.append(np.abs(nf_spec) / (np.max(np.abs(nf_spec)) + epsilon))

    return np.array(far_spectrograms), np.array(near_spectrograms)

# TransformerModel
@tf.keras.utils.register_keras_serializable()
class TransformerModel(Model):
    def __init__(self, input_dim, num_heads=8, ff_dim=512, num_layers=4, **kwargs):
        super(TransformerModel, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers

        # Create layers
        self.encoder = [
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_dim)
            for _ in range(num_layers)
        ]
        self.ffn = [
            tf.keras.Sequential(
                [layers.Dense(ff_dim, activation="relu"), layers.Dense(input_dim)]
            )
            for _ in range(num_layers)
        ]
        self.norm_layers = [layers.LayerNormalization() for _ in range(num_layers)]
        self.output_layer = layers.Dense(input_dim)

    def call(self, x):
        for attn, ffn, norm in zip(self.encoder, self.ffn, self.norm_layers):
            attn_out = attn(x, x)
            x = norm(x + attn_out)  # Add & Norm
            ffn_out = ffn(x)
            x = norm(x + ffn_out)  # Add & Norm
        return self.output_layer(x)

# Load dataset
far_field_dir = r"C:\Users\Archana S\OneDrive\Desktop\Dataset\FF2"
near_field_dir = r"C:\Users\Archana S\OneDrive\Desktop\Dataset\NF2"
far_spectrograms, near_spectrograms = load_audio_data(far_field_dir, near_field_dir)

# Define model
input_dim = far_spectrograms.shape[-1]
model = TransformerModel(input_dim=input_dim)
model.compile(optimizer="adam", loss="mae")  # Use MAE for smoother output

# Callbacks for better training
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3)

# Training on the entire dataset
model.fit(
    far_spectrograms,
    near_spectrograms,
    validation_split=0.2,
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping, lr_scheduler],
)

# Specify a single far-field MP3 audio file for prediction
single_ff_file = r"C:\Users\Archana S\OneDrive\Desktop\Dataset\clip_670(FF).mp3"
ff_audio, _ = librosa.load(single_ff_file, sr=SR)  # Load the MP3 file

# Convert to spectrogram
ff_spec = librosa.stft(ff_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
ff_spec_mag = np.abs(ff_spec) / np.max(np.abs(ff_spec))  # Normalize

# Add batch dimension for prediction
ff_spec_mag = np.expand_dims(ff_spec_mag, axis=0)

# Predict near-field spectrogram
predicted_near = model.predict(ff_spec_mag)

# Convert predicted spectrogram back to audio using iSTFT
predicted_near_mag = predicted_near[0] * np.max(np.abs(ff_spec))  # Denormalize
predicted_near_audio = librosa.istft(predicted_near_mag, hop_length=HOP_LENGTH)

# Save predicted audio
output_dir = r"C:\Users\Archana S\OneDrive\Desktop\Dataset\Predicted_NF"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
output_file = os.path.join(output_dir, "p3.wav")
sf.write(output_file, predicted_near_audio, SR)

print(f"Predicted near-field audio saved to: {output_file}")

# Plot spectrograms for debugging
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(ff_spec, ref=np.max), sr=SR, hop_length=HOP_LENGTH)
plt.title("Far-Field Spectrogram (Input)")
plt.colorbar()

plt.subplot(2, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(predicted_near_mag, ref=np.max), sr=SR, hop_length=HOP_LENGTH)
plt.title("Predicted Near-Field Spectrogram")
plt.colorbar()
plt.tight_layout()
plt.show()
