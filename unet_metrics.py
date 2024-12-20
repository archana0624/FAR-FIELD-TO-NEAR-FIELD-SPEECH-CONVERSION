import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.metrics.pairwise import cosine_similarity
from mir_eval.separation import bss_eval_sources
from pystoi import stoi

# Load predicted and reference audio
predicted_audio_path = r"C:\Users\Archana S\OneDrive\Desktop\Dataset\output_audio6.wav"
reference_audio_path = r"C:\Users\Archana S\OneDrive\Desktop\Dataset\clip_670(NF).mp3"
predicted_audio, sr_pred = librosa.load(predicted_audio_path, sr=16000)


# Visualize original, predicted
reference_audio, sr_ref = librosa.load(reference_audio_path, sr=16000)
fig, ax = plt.subplots(2, 1, figsize=(10, 6))

# Original spectrogram
S_ref = librosa.amplitude_to_db(np.abs(librosa.stft(reference_audio)), ref=np.max)
img1 = librosa.display.specshow(S_ref, sr=sr_ref, x_axis='time', y_axis='hz', ax=ax[0])
ax[0].set_title("Original Near-Field Audio Spectrogram")
plt.colorbar(img1, ax=ax[0])

# Predicted spectrogram
S_pred = librosa.amplitude_to_db(np.abs(librosa.stft(predicted_audio)), ref=np.max)
img2 = librosa.display.specshow(S_pred, sr=sr_pred, x_axis='time', y_axis='hz', ax=ax[1])
ax[1].set_title("Predicted Near-Field Audio Spectrogram")
plt.colorbar(img2, ax=ax[1])

plt.tight_layout()
plt.show()



def compute_cosine_similarity(original, predicted):
    original_spectrogram = librosa.stft(original)
    predicted_spectrogram = librosa.stft(predicted)
    original_db = librosa.amplitude_to_db(abs(original_spectrogram), ref=np.max)
    predicted_db = librosa.amplitude_to_db(abs(predicted_spectrogram), ref=np.max)
    
    similarity = cosine_similarity(original_db.T, predicted_db.T)
    return np.mean(similarity)

# File paths
nf_audio_path = r"C:\Users\Archana S\OneDrive\Desktop\Dataset\clip_670(NF).mp3"
pred_nf_audio_path = r"C:\Users\Archana S\OneDrive\Desktop\Dataset\output_audio6.wav"
similarity = compute_cosine_similarity(nf_audio, pred_nf_audio)
print(f"Cosine Similarity: {similarity:.2f}")


def calculate_noise_reduction(original, predicted, frequency_threshold=10000):
    original_spectrogram = librosa.stft(original)
    predicted_spectrogram = librosa.stft(predicted)
    
    original_db = librosa.amplitude_to_db(abs(original_spectrogram), ref=np.max)
    predicted_db = librosa.amplitude_to_db(abs(predicted_spectrogram), ref=np.max)
    
    # Identify high-frequency regions (above the threshold)
    high_freq_indices = np.where(np.linspace(0, 22050, original_db.shape[0]) > frequency_threshold)[0]
    
    original_high_freq = np.mean(original_db[high_freq_indices])
    predicted_high_freq = np.mean(predicted_db[high_freq_indices])
    
    noise_reduction = (original_high_freq - predicted_high_freq) / original_high_freq * 100
    return noise_reduction

# File paths
nf_audio_path = r"C:\Users\Archana S\OneDrive\Desktop\Dataset\clip_670(NF).mp3"
pred_nf_audio_path = r"C:\Users\Archana S\OneDrive\Desktop\Dataset\output_audio6.wav"

noise_reduction_percentage = calculate_noise_reduction(nf_audio, pred_nf_audio)
print(f"Noise reduction: {noise_reduction_percentage:.2f}%")


def calculate_sdr(reference_audio, predicted_audio):
   
    # Ensure both signals have the same length
    min_len = min(len(reference_audio), len(predicted_audio))
    reference_audio = reference_audio[:min_len]
    predicted_audio = predicted_audio[:min_len]

    # Convert to 2D arrays as required by mir_eval
    reference_audio = np.expand_dims(reference_audio, axis=0)
    predicted_audio = np.expand_dims(predicted_audio, axis=0)

    # Calculate SDR
    sdr, _, _, _ = bss_eval_sources(reference_audio, predicted_audio)
    return sdr[0]  # Return the SDR for the single channel

# File paths
nf_audio_path = r"C:\Users\Archana S\OneDrive\Desktop\Dataset\clip_670(NF).mp3"
pred_nf_audio_path = r"C:\Users\Archana S\OneDrive\Desktop\Dataset\output_audio6.wav"

# Load the original near-field and predicted near-field audios
nf_audio, sr_nf = librosa.load(nf_audio_path, sr=None)
pred_nf_audio, sr_pred = librosa.load(pred_nf_audio_path, sr=None)

# Ensure the sampling rates match (if they differ, resample the predicted audio)
if sr_nf != sr_pred:
    pred_nf_audio = librosa.resample(pred_nf_audio, orig_sr=sr_pred, target_sr=sr_nf)

# Calculate SDR
sdr_value = calculate_sdr(nf_audio, pred_nf_audio)
print(f"SDR between original and predicted near-field audio: {sdr_value:.2f} dB")



def calculate_stoi(reference_audio_path, degraded_audio_path, sampling_rate=16000):
    """
    Calculate STOI (Short-Time Objective Intelligibility) between reference and degraded audio.
    float: STOI score (higher is better).
    """
    # Load the reference and degraded audio
    ref_audio, sr_ref = librosa.load(reference_audio_path, sr=sampling_rate)
    deg_audio, sr_deg = librosa.load(degraded_audio_path, sr=sampling_rate)

    # Ensure both signals have the same length
    min_len = min(len(ref_audio), len(deg_audio))
    ref_audio = ref_audio[:min_len]
    deg_audio = deg_audio[:min_len]

    # Calculate STOI
    stoi_score = stoi(ref_audio, deg_audio, sr_ref)

    return stoi_score

# File paths
reference_audio_path = r"C:\Users\Archana S\OneDrive\Desktop\Dataset\clip_670(NF).mp3"
degraded_audio_path = r"C:\Users\Archana S\OneDrive\Desktop\Dataset\output_audio6.wav"

# Calculate STOI
stoi_score = calculate_stoi(reference_audio_path, degraded_audio_path)
print(f"STOI Score: {stoi_score:.2f}")
