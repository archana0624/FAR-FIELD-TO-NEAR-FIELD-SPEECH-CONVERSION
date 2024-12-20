import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import soundfile as sf


# Load and preprocess audio pairs
def load_audio_pairs(near_dir, far_dir):
    near_files = sorted([f for f in os.listdir(near_dir) if f.endswith('.mp3')])
    far_files = sorted([f for f in os.listdir(far_dir) if f.endswith('.mp3')])
    
    audio_pairs = []
    for near_file, far_file in zip(near_files, far_files):
        near_path = os.path.join(near_dir, near_file)
        far_path = os.path.join(far_dir, far_file)
        
        near_audio, sr = librosa.load(near_path, sr=None)
        far_audio, _ = librosa.load(far_path, sr=sr)  # Ensure both have the same sample rate
        
        audio_pairs.append((near_audio, far_audio, sr))
    
    return audio_pairs

# Preprocess and segment audio
def preprocess_audio(audio, sr, target_sr=16000, segment_length=16000):
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    audio = np.pad(audio, (0, max(0, segment_length - len(audio))), 'constant')
    return segment_audio(audio, segment_length)

def segment_audio(audio, segment_length):
    """Segment audio into chunks of specified length."""
    segments = []
    for start in range(0, len(audio) - segment_length + 1, segment_length):
        segments.append(audio[start:start + segment_length])
    return np.array(segments)

# Create dataset from audio pairs
def create_dataset(audio_pairs, target_sr=16000, segment_length=16000):
    X, y = [], []
    for near_audio, far_audio, sr in audio_pairs:
        segments_far = preprocess_audio(far_audio, sr, target_sr, segment_length)
        segments_near = preprocess_audio(near_audio, sr, target_sr, segment_length)
        
        min_len = min(len(segments_far), len(segments_near))
        X.extend(segments_far[:min_len])
        y.extend(segments_near[:min_len])
    
    X = np.array(X).reshape(-1, segment_length, 1)
    y = np.array(y).reshape(-1, segment_length, 1)
    
    return X, y

# Build U-Net model
def build_unet_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Encoder
    conv1 = Conv1D(32, kernel_size=3, activation='relu', padding='same')(inputs)
    conv1 = Conv1D(32, kernel_size=3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    
    conv2 = Conv1D(64, kernel_size=3, activation='relu', padding='same')(pool1)
    conv2 = Conv1D(64, kernel_size=3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    
    conv3 = Conv1D(128, kernel_size=3, activation='relu', padding='same')(pool2)
    conv3 = Conv1D(128, kernel_size=3, activation='relu', padding='same')(conv3)
    
    # Decoder
    up2 = UpSampling1D(size=2)(conv3)
    up2 = concatenate([up2, conv2])
    conv4 = Conv1D(64, kernel_size=3, activation='relu', padding='same')(up2)
    conv4 = Conv1D(64, kernel_size=3, activation='relu', padding='same')(conv4)
    
    up1 = UpSampling1D(size=2)(conv4)
    up1 = concatenate([up1, conv1])
    conv5 = Conv1D(32, kernel_size=3, activation='relu', padding='same')(up1)
    conv5 = Conv1D(32, kernel_size=3, activation='relu', padding='same')(conv5)
    
    outputs = Conv1D(1, kernel_size=1, activation='linear')(conv5)
    
    model = Model(inputs, outputs)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    
    return model

# Set paths
near_dir = r"C:\Users\Archana S\OneDrive\Desktop\Dataset\NF2"
far_dir = r"C:\Users\Archana S\OneDrive\Desktop\Dataset\FF2"

# Load audio pairs
audio_pairs = load_audio_pairs(near_dir, far_dir)

# Create dataset
X, y = create_dataset(audio_pairs)

# Build and compile model
input_shape = (16000, 1)
model = build_unet_model(input_shape)
model.summary()

# Train model
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('unet_audio_model3.keras', save_best_only=True)
]

history = model.fit(X, y, epochs=20, batch_size=16, validation_split=0.2, callbacks=callbacks)

# Plot training history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
