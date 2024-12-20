# FAR-FIELD-TO-NEAR-FIELD-SPEECH-CONVERSION

This repository contains the implementation of a machine learning pipeline for far-field to near-field speech conversion using a U-Net model. The project focuses on improving speech clarity and recognition for voice assistant systems.

File Structure:

Unet_final_Complete_code.ipynb: The complete Jupyter Notebook implementation of the U-Net model, including data preprocessing, model training, and evaluation.
Unet_test.py: Script to test the trained U-Net model on far-field audio files and generate predicted near-field outputs.
unet_metrics.py: Contains utility functions to calculate evaluation metrics such as SDR (Signal-to-Distortion Ratio) for model performance analysis.
unet_train.py: Script for training the U-Net model, including dataset handling, training pipeline, and checkpoints.

U-Net Model:

The U-Net model is adapted for audio signal processing to transform far-field speech into near-field speech. The architecture is designed to balance efficiency and accuracy, ensuring the output speech is both clear and computationally lightweight for real-time applications.

Key Features:

Preprocessing: Segments audio into manageable chunks for model input.
Training: Includes early stopping and checkpointing for optimal performance.
Metrics: Uses SDR and other metrics to evaluate the quality of converted audio.

How to Use:

Training the Model:
Run the unet_train.py script to train the U-Net model on your dataset. Adjust configurations as needed.
Replace the train code with far-field and corresponding near-field audio folder.

Testing the Model:
Use the Unet_test.py script to test the trained model on far-field audio files. The script generates near-field audio outputs and saves them.

Metrics Evaluation:
Evaluate the model's performance using the functions provided in unet_metrics.py.

Requirements:
Python 3.x
Libraries: TensorFlow, NumPy, Librosa, Matplotlib, and other dependencies listed in the code files.

