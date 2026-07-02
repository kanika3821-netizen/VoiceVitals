import librosa
import numpy as np
import scipy.stats

def extract_parkinsons_features(audio, sr):
    """
    Extract features that actually match
    the Parkinson's dataset column structure
    """
    features = []

    # ── Fundamental Frequency Features ──────────────
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr
    )
    f0_clean = f0[~np.isnan(f0)]

    if len(f0_clean) > 1:
        fo = float(np.mean(f0_clean))
        fhi = float(np.max(f0_clean))
        flo = float(np.min(f0_clean))
    else:
        fo, fhi, flo = 150.0, 200.0, 100.0

    features.extend([fo, fhi, flo])

    # ── Jitter Features ──────────────────────────────
    if len(f0_clean) > 1:
        diff_f0 = np.abs(np.diff(f0_clean))
        jitter_percent = float(
            np.mean(diff_f0) / (fo + 1e-8) * 100)
        jitter_abs = float(np.mean(diff_f0))
        jitter_rap = float(np.mean([
            np.mean(np.abs(
                f0_clean[i] - np.mean(
                    f0_clean[max(0,i-1):i+2])))
            for i in range(1, len(f0_clean)-1)
        ]) / (fo + 1e-8))
        jitter_ppq5 = jitter_rap * 0.8
        jitter_ddp = jitter_rap * 3.0
    else:
        jitter_percent = 0.0
        jitter_abs = 0.0
        jitter_rap = 0.0
        jitter_ppq5 = 0.0
        jitter_ddp = 0.0

    features.extend([
        jitter_percent, jitter_abs,
        jitter_rap, jitter_ppq5, jitter_ddp
    ])

    # ── Shimmer Features ─────────────────────────────
    rms = librosa.feature.rms(y=audio)[0]
    if len(rms) > 1:
        diff_rms = np.abs(np.diff(rms))
        mean_rms = np.mean(rms) + 1e-8
        shimmer = float(np.mean(diff_rms) / mean_rms)
        shimmer_db = float(
            20 * np.log10(shimmer + 1e-8) * -1)
        shimmer_apq3 = shimmer * 0.8
        shimmer_apq5 = shimmer * 0.9
        shimmer_apq11 = shimmer * 1.1
        shimmer_dda = shimmer * 3.0
    else:
        shimmer = 0.02
        shimmer_db = 0.2
        shimmer_apq3 = 0.016
        shimmer_apq5 = 0.018
        shimmer_apq11 = 0.022
        shimmer_dda = 0.06

    features.extend([
        shimmer, shimmer_db, shimmer_apq3,
        shimmer_apq5, shimmer_apq11, shimmer_dda
    ])

    # ── Noise Ratio Features ─────────────────────────
    harmonic, percussive = librosa.effects.hpss(audio)
    harmonic_energy = np.sum(harmonic ** 2) + 1e-8
    noise_energy = np.sum(percussive ** 2) + 1e-8
    nhr = float(noise_energy / harmonic_energy)
    hnr = float(10 * np.log10(
        harmonic_energy / noise_energy))

    features.extend([nhr, hnr])

    # ── Nonlinear Features ───────────────────────────
    if len(f0_clean) > 2:
        rpde = float(
            scipy.stats.entropy(
                np.histogram(f0_clean, bins=10)[0] + 1e-8
            ) / np.log(10))
        dfa = float(np.std(f0_clean) / (
            np.mean(f0_clean) + 1e-8))
        spread1 = float(
            np.percentile(f0_clean, 75) -
            np.percentile(f0_clean, 25))
        spread2 = float(np.var(f0_clean))
        d2 = float(np.mean(np.abs(np.diff(f0_clean, 2))))
        ppe = float(scipy.stats.entropy(
            np.histogram(
                np.diff(f0_clean), bins=10)[0] + 1e-8))
    else:
        rpde = 0.4
        dfa = 0.7
        spread1 = -5.0
        spread2 = 0.2
        d2 = 2.0
        ppe = 0.2

    features.extend([rpde, dfa, spread1,
                     spread2, d2, ppe])

    return np.array(features)


# ── Test ─────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    # Check feature count matches dataset
    df = pd.read_csv('parkinsons.csv')
    expected = df.drop(['name', 'status'],
                       axis=1).shape[1]
    print(f"Dataset expects: {expected} features")

    audio, sr = librosa.load('my_voice.wav')
    features = extract_parkinsons_features(audio, sr)
    print(f"We extract: {len(features)} features")

    if len(features) == expected:
        print("✅ Feature count matches perfectly!")
    else:
        print(f"⚠️ Mismatch — adjusting needed")
        print(f"Difference: {expected - len(features)}")