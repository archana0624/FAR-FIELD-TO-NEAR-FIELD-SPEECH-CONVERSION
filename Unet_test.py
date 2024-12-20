from tensorflow.keras.models import load_model
import librosa
import numpy as np
import soundfile as sf

# Load the pretrained model (assuming the model is saved as 'best_model.keras')
model = load_model('unet_audio_model3.keras')

def preprocess_audio(audio, sr, target_sr=16000, segment_length=16000):
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    audio = np.pad(audio, (0, max(0, segment_length - len(audio))), 'constant')
    segments = segment_audio(audio, segment_length)
    return segments

def segment_audio(audio, segment_length):
    segments = []
    for start in range(0, len(audio) - segment_length + 1, segment_length):
        segments.append(audio[start:start + segment_length])
    return np.array(segments)

# Test the model on a new audio file
test_audio_path = r"C:\Users\Archana S\OneDrive\Desktop\Dataset\clip_670(FF).mp3"
test_audio, sr = librosa.load(test_audio_path, sr=None)
preprocessed_test_audio = preprocess_audio(test_audio, sr)

# Make predictions on each segment and combine results
predicted_segments = []
for segment in preprocessed_test_audio:
    predicted_segment = model.predict(segment.reshape(1, -1, 1)).flatten()
    predicted_segments.append(predicted_segment)

# Combine the segments back into one audio file
predicted_near_audio = np.concatenate(predicted_segments)

# Save the predicted near-field audio
output_path = r"C:\Users\Archana S\OneDrive\Desktop\Dataset\output_audio6.wav"
sf.write(output_path, predicted_near_audio, 16000)

print(f"Predicted near-field audio saved to {output_path}")