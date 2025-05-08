import matplotlib.pyplot as plt
import tempfile
import speech_recognition as sr
import librosa
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from pydub import AudioSegment
import soundfile as sf
import numpy as np
import wave
from streamlit_mic_recorder import mic_recorder
import streamlit as st
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


model_path = 'speech.keras'
model = load_model(model_path)
label_mapping = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise',
}



# Feature Extraction Function

def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_best')

    # Extract MFCC features
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    # Extract Chroma Features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, n_chroma=12)
    chroma_scaled_features = np.mean(chroma.T, axis=0)

    # Extract Mel Spectrogram Features
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    mel_scaled_features = np.mean(mel.T, axis=0)

    # Concatenate all features into a single array
    features = np.hstack((mfccs_scaled_features,chroma_scaled_features, mel_scaled_features))

    return features

# Function to predict emotions from audio segments
def predict_emotions(audio_path, interval):
    audio_data, samplerate = sf.read(audio_path)
    duration = len(audio_data) / samplerate
    emotions = []

    for start in np.arange(0, duration, interval):
        end = start + interval
        if end > duration:
            end = duration
        segment = audio_data[int(start*samplerate):int(end*samplerate)]
        segment_path = 'segment.wav'
        sf.write(segment_path, segment, samplerate)
        # Extract features
        feat = features_extractor(segment_path)
        if feat is not None:
            feat = feat.reshape(1, -1)
            predictions = model.predict(feat)

            # Format predictions
            # predicted_emotions = {label_mapping[i]: float(predictions[0][i]) for i in range(len(label_mapping))}
            predicted_emotions = {label_mapping[i]: round(
                float(predictions[0][i]), 4) for i in range(len(label_mapping))}

            emotions.append((start, end, predicted_emotions))

    return emotions


def main():
    st.title("🎤 Speech Emotion Recognition")

    st.header("Upload Audio")
    audio_file = st.file_uploader("Upload an audio file:", type=["mp3", "wav"])

    if st.button("Upload"):
        if audio_file:
            audio_data, samplerate = sf.read(audio_file)
            # Convert the audio file to WAV format and save it
            output_file_path = 'upload.wav'
            sf.write(output_file_path, audio_data, samplerate)

            st.audio(audio_file)
        else:
            st.error("Please upload an audio file.")

    interval = st.number_input(
        "Set the interval (0.00-15.00 seconds) for emotion detection segments:",
        min_value=0.00, max_value=15.00, value=3.00, step=0.01
    )

    if st.button("Predict"):
        if audio_file:
            emotions = predict_emotions('upload.wav', interval=interval)
            emotions_df = pd.DataFrame(emotions, columns=["Start", "End", "Emotion"])
            st.write(emotions_df)

            # Save emotions to a log file
            log_file_path = 'emotion_log.csv'
            emotions_df.to_csv(log_file_path, index=False)

            # Extrapolate major emotions (kept exactly as provided)
            major_emotion = emotions_df['Emotion'].mode().values[0]
            major_emotion = [key for key in major_emotion if major_emotion[key] == max(
                major_emotion.values())]
            st.write(f"Major emotion: {major_emotion}")

            st.success(f"Emotion log saved to {log_file_path}")

            # Add download button for the emotion log file
            with open(log_file_path, "rb") as file:
                st.download_button(
                    label="Download Emotion Log", 
                    data=file, 
                    file_name='emotion_log.csv', 
                    mime='text/csv'
                )

if __name__ == '__main__':
    main()


# Run -> terminal -> open virtual environment - myenv\Scripts\activate 
# streamlit run app.py
# upload audio file 
# predict the emotion