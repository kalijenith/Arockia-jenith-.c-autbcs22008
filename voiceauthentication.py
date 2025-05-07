# --------------------------------------------
# ✅ Required Libraries:
# pip install sounddevice soundfile librosa numpy scikit-learn
# --------------------------------------------

import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
from sklearn.metrics.pairwise import cosine_similarity

# 📌 Record and save audio
def record_audio(file_name, seconds=5, sample_rate=44100):
    try:
        print("🎙️ Recording...")
        audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        sf.write(file_name, audio, sample_rate)
        print(f"✅ Recorded and saved as: {file_name}")
    except Exception as e:
        print(f"❌ Recording failed: {e}")

# 📌 Get MFCC features from file
def get_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return None

# 📌 Enroll new user
def enroll(username):
    file_name = f"{username}_enroll.wav"
    record_audio(file_name)
    features = get_features(file_name)
    if features is not None:
        np.save(f"{username}_features.npy", features)
        print("🆗 Enrolled successfully!")

# 📌 Authenticate user
def authenticate(username):
    feature_file = f"{username}_features.npy"
    if not os.path.exists(feature_file):
        print("❌ User not found. Please enroll first.")
        return

    test_file = f"{username}_test.wav"
    record_audio(test_file)
    test_features = get_features(test_file)

    if test_features is None:
        print("❌ Couldn't extract features from test voice.")
        return

    saved_features = np.load(feature_file)

    score = cosine_similarity([saved_features], [test_features])[0][0]
    print(f"🔍 Similarity: {score:.2f}")

    if score > 0.85:
        print("✅ Access Granted!")
    else:
        print("❌ Access Denied!")

# 📌 Main Menu
if __name__ == "__main__":
    print("\n🎤 Voice Authentication System")
    print("1. Enroll")
    print("2. Authenticate")
    option = input("Choose (1/2): ")

    if option == "1":
        user = input("Enter username: ")
        enroll(user)

    elif option == "2":
        user = input("Enter username: ")
        authenticate(user)

    else:
        print("⚠️ Invalid option.")
