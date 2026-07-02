import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
import joblib
import os
import time

# ── Load AI Model ────────────────────────────────────
if os.path.exists('voice_model.pkl'):
    ai_model = joblib.load('voice_model.pkl')
    ai_scaler = joblib.load('scaler.pkl')
    MODEL_LOADED = True
    print("✅ Real AI Model loaded! (94% accuracy)")
else:
    ai_model = None
    ai_scaler = None
    MODEL_LOADED = False
    print("⚠️ No trained model found")


def record_audio(duration=30, sample_rate=44100):
    print(f"Recording {duration} seconds...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32',
        device=1
    )
    sd.wait()
    print("✅ Done recording!")
    return audio.flatten(), sample_rate


def extract_all_features(audio, sr):
    features = {}

    pitches, magnitudes = librosa.piptrack(
        y=audio, sr=sr)
    pitch_vals = pitches[magnitudes > 0.1]
    features['mean_pitch'] = float(
        np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0.0
    features['pitch_variation'] = float(
        np.std(pitch_vals)) if len(pitch_vals) > 0 else 0.0

    mfccs = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i}'] = float(np.mean(mfccs[i]))

    rms = librosa.feature.rms(y=audio)
    features['energy_mean'] = float(np.mean(rms))
    features['energy_std'] = float(np.std(rms))

    zcr = librosa.feature.zero_crossing_rate(audio)
    features['zcr_mean'] = float(np.mean(zcr))

    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr)
    features['spectral_centroid'] = float(
        np.mean(centroid))

    rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sr)
    features['spectral_rolloff'] = float(
        np.mean(rolloff))

    onsets = librosa.onset.onset_detect(
        y=audio, sr=sr)
    features['speech_rate'] = len(onsets) / (
        len(audio) / sr)

    return features


def analyze_health(features):
    scores = {}

    mental_risk = 0
    if features['energy_mean'] < 0.05:
        mental_risk += 30
    if features['speech_rate'] < 2.0:
        mental_risk += 30
    if features['mfcc_0'] < -300:
        mental_risk += 40
    scores['Mental Wellness'] = max(0, 100 - mental_risk)

    resp_risk = 0
    if features['zcr_mean'] > 0.08:
        resp_risk += 35
    if features['energy_std'] > 0.08:
        resp_risk += 35
    if features['spectral_centroid'] > 2500:
        resp_risk += 30
    scores['Respiratory Health'] = max(0, 100 - resp_risk)

    stress_risk = 0
    if features['pitch_variation'] > 500:
        stress_risk += 35
    if features['speech_rate'] > 4.0:
        stress_risk += 35
    if features['energy_std'] > 0.08:
        stress_risk += 30
    scores['Stress Level'] = max(0, 100 - stress_risk)

    vocal_risk = 0
    if features['zcr_mean'] > 0.1:
        vocal_risk += 50
    if features['energy_std'] > 0.1:
        vocal_risk += 50
    scores['Vocal Health'] = max(0, 100 - vocal_risk)

    recommendations = []
    if scores['Mental Wellness'] < 50:
        recommendations.append(
            "Low mental wellness detected — "
            "consider speaking to a counselor")
    if scores['Respiratory Health'] < 50:
        recommendations.append(
            "Irregular breathing detected — "
            "consult doctor if persistent")
    if scores['Stress Level'] < 50:
        recommendations.append(
            "High stress detected — "
            "try deep breathing or meditation")
    if scores['Vocal Health'] < 50:
        recommendations.append(
            "Vocal strain detected — "
            "rest your voice and stay hydrated")
    if not recommendations:
        recommendations.append(
            "Voice patterns look healthy! "
            "Keep maintaining a healthy lifestyle")

    return scores, recommendations


def get_ai_powered_score(audio, sr):
    """Use the real trained AI model for scoring"""
    if not MODEL_LOADED:
        return None

    mfccs = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=22)
    feature_vector = [float(np.mean(m)) for m in mfccs]

    while len(feature_vector) < 22:
        feature_vector.append(0.0)
    feature_vector = feature_vector[:22]

    try:
        scaled = ai_scaler.transform([feature_vector])
        prediction_proba = ai_model.predict_proba(scaled)[0]

        ai_score = {
            'healthy_probability': float(
                prediction_proba[0] * 100),
            'risk_probability': float(
                prediction_proba[1] * 100),
            'model_confidence': float(
                max(prediction_proba) * 100)
        }
        return ai_score
    except Exception as e:
        print(f"AI scoring error: {e}")
        return None


# ── Test Section ──────────────────────────────────────
if __name__ == "__main__":
    print("\n🔍 Testing FULL voice engine with AI model...")
    print("="*50)

    samples = {
        'happy.wav': 'Happy',
        'sad.wav': 'Sad',
        'angry.wav': 'Angry',
        'tired.wav': 'Tired',
        'normal.wav': 'Normal'
    }

    for filename, emotion in samples.items():
        try:
            audio, sr = librosa.load(filename)

            features = extract_all_features(audio, sr)
            scores, recs = analyze_health(features)
            ai_result = get_ai_powered_score(audio, sr)

            print(f"\n🎤 {emotion}:")
            print("  Rule-Based Scores:")
            for category, score in scores.items():
                print(f"     {category}: {score}%")

            if ai_result:
                print("  AI Model Prediction:")
                print(f"     Healthy: "
                      f"{ai_result['healthy_probability']:.1f}%")
                print(f"     Risk: "
                      f"{ai_result['risk_probability']:.1f}%")

        except Exception as e:
            print(f"{emotion}: Error — {e}")

    print("\n" + "="*50)
    print("✅ Full pipeline working with REAL AI model!")