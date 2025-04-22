import librosa
import numpy as np
import joblib

# Load model, scaler, and encoder
model = joblib.load('age_predictor_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Function to extract features
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)  # load first 30 seconds
    features = []
    features.append(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.rms(y=y)))
    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.zero_crossing_rate(y)))

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for mfcc in mfccs:
        features.append(np.mean(mfcc))

    return np.array(features).reshape(1, -1)

# Predict function
def predict_age_group(file_path):
    features = extract_features(file_path)
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    predicted_label = label_encoder.inverse_transform(prediction)
    return predicted_label[0]

# Example usage
song_path = 'converted_wav/Mesmerizing Nights A Dreamy Jazz & Blues Love Song_converted.wav'
predicted_age = predict_age_group(song_path)
print(f"The predicted age group is: {predicted_age}")
