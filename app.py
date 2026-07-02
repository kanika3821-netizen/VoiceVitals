import streamlit as st
import librosa
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import time
import joblib
import json
import os
import scipy.stats
from datetime import datetime
from extract_features_v2 import extract_parkinsons_features
from generate_report import generate_pdf_report

st.set_page_config(
    page_title="VoiceVitals",
    page_icon="🎙️",
    layout="centered"
)

# ── Load Model ───────────────────────────────────────
model = joblib.load('voice_model.pkl')
scaler = joblib.load('scaler.pkl')

# ── Session State ────────────────────────────────────
if 'healthy_pct' not in st.session_state:
    st.session_state.healthy_pct = None
if 'risk_pct' not in st.session_state:
    st.session_state.risk_pct = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None


# ── History Functions ────────────────────────────────
def save_result(risk_pct, healthy_pct):
    history = []
    if os.path.exists('voice_history.json'):
        with open('voice_history.json', 'r') as f:
            history = json.load(f)
    history.append({
        'date': datetime.now().strftime('%d %b %Y %H:%M'),
        'risk_score': round(risk_pct, 1),
        'healthy_score': round(healthy_pct, 1)
    })
    with open('voice_history.json', 'w') as f:
        json.dump(history, f)
    return history


def load_history():
    if os.path.exists('voice_history.json'):
        with open('voice_history.json', 'r') as f:
            return json.load(f)
    return []


# ── Styling ──────────────────────────────────────────
st.markdown("""
<style>
.main { background-color: #0E1117; }
h1 {
    background: linear-gradient(135deg, #1E88E5, #64B5F6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.8rem;
}
h2, h3 { color: #64B5F6; font-weight: 600; }
.stButton > button {
    background: linear-gradient(135deg, #1E88E5, #0D47A1);
    color: white;
    border-radius: 12px;
    padding: 0.8rem 2rem;
    font-weight: 700;
    border: none;
    width: 100%;
}
[data-testid="metric-container"] {
    background: #1E2130;
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid #2D3748;
}
.disclaimer {
    background: #1A1A2E;
    border-radius: 10px;
    padding: 1rem;
    border-left: 4px solid #FF9800;
    font-size: 0.85rem;
    color: #aaa;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────
st.title("🎙️ VoiceVitals")
st.markdown(
    "**Early neurological voice screening** — "
    "detect vocal biomarkers linked to motor disorders "
    "using only your phone microphone."
)

with st.expander("ℹ️ What does VoiceVitals actually do?"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **What it measures:**
        - Pitch frequency stability
        - Voice tremor (jitter)
        - Amplitude variation (shimmer)
        - Vocal fold irregularities
        - Speech rhythm patterns
        """)
    with col2:
        st.markdown("""
        **Validated on:**
        - UCI Parkinson's Voice Dataset
        - 195 real clinical recordings
        - 92% test accuracy
        - 97% at-risk detection rate
        """)

st.divider()

# ── Recording ────────────────────────────────────────
st.subheader("🎙️ Start Your Screening")
st.write(
    "Speak naturally for the selected duration. "
    "Read anything aloud or describe your day."
)

duration = st.select_slider(
    "Recording duration",
    options=[10, 15, 20, 25, 30],
    value=15
)

if st.button("🎙️ Begin Voice Analysis",
             use_container_width=True,
             type="primary"):

    progress_bar = st.progress(0)
    status = st.empty()
    countdown = st.empty()
    status.info("🔴 Recording in progress — speak naturally")

    sample_rate = 44100
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32',
        device=1
    )

    for i in range(duration):
        time.sleep(1)
        progress_bar.progress((i + 1) / duration)
        remaining = duration - i - 1
        if remaining > 0:
            countdown.caption(
                f"⏱️ {remaining} seconds remaining")

    sd.wait()
    audio = audio_data.flatten()
    countdown.empty()
    status.success("✅ Recording complete — analyzing...")

    with st.spinner("🔬 Analyzing vocal biomarkers..."):
        time.sleep(1.5)
        try:
            feature_vector = extract_parkinsons_features(
                audio, sample_rate)
            expected = 22
            if len(feature_vector) < expected:
                feature_vector = np.pad(
                    feature_vector,
                    (0, expected - len(feature_vector)))
            feature_vector = feature_vector[:expected]
            scaled = scaler.transform([feature_vector])
            proba = model.predict_proba(scaled)[0]
            st.session_state.healthy_pct = proba[0] * 100
            st.session_state.risk_pct = proba[1] * 100
            st.session_state.analysis_done = True
            st.session_state.audio_data = audio.tolist()
        except Exception as e:
            st.error(f"Analysis error: {e}")
            st.stop()

    save_result(
        st.session_state.risk_pct,
        st.session_state.healthy_pct)

# ── Results ───────────────────────────────────────────
if st.session_state.analysis_done:
    healthy_pct = st.session_state.healthy_pct
    risk_pct = st.session_state.risk_pct

    st.divider()
    st.markdown("## 📊 Screening Result")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🟢 Healthy Pattern Match",
                  f"{healthy_pct:.1f}%")
    with col2:
        st.metric("🔴 At-Risk Pattern Match",
                  f"{risk_pct:.1f}%")

    st.progress(healthy_pct / 100)

    if risk_pct > 65:
        st.warning(
            "⚠️ Your voice pattern shows similarity to "
            "at-risk profiles in our clinical training data. "
            "This does NOT mean you have Parkinson's disease. "
            "We recommend consulting a neurologist.")
    elif risk_pct > 40:
        st.info(
            "ℹ️ Your voice pattern shows some at-risk "
            "indicators. Monitor your voice over time.")
    else:
        st.success(
            "✅ Your voice pattern closely matches healthy "
            "profiles in our clinical training data.")

    # ── Waveform ─────────────────────────────────────
    if st.session_state.audio_data is not None:
        st.divider()
        st.markdown("### 🌊 Your Voice Pattern")
        audio = np.array(st.session_state.audio_data)
        sample_rate = 44100
        fig, ax = plt.subplots(figsize=(10, 2.5))
        time_axis = np.linspace(
            0, 15, len(audio[:sample_rate * 5]))
        ax.plot(time_axis, audio[:sample_rate * 5],
                color='#1E88E5', linewidth=0.6, alpha=0.8)
        ax.fill_between(time_axis,
                        audio[:sample_rate * 5],
                        alpha=0.2, color='#1E88E5')
        ax.set_facecolor('#0E1117')
        fig.patch.set_facecolor('#0E1117')
        ax.set_xlabel("Time (seconds)", color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')
        st.pyplot(fig)

    # ── Trend Chart ───────────────────────────────────
    history = load_history()
    if len(history) > 1:
        st.divider()
        st.markdown("### 📈 Your Trend Over Time")
        dates = [h['date'] for h in history]
        risk_scores = [h['risk_score'] for h in history]
        healthy_scores = [
            h['healthy_score'] for h in history]
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(range(len(dates)), risk_scores,
                 marker='o', color='#EF5350',
                 label='At-Risk', linewidth=2)
        ax2.plot(range(len(dates)), healthy_scores,
                 marker='o', color='#66BB6A',
                 label='Healthy', linewidth=2)
        ax2.set_xticks(range(len(dates)))
        ax2.set_xticklabels(
            dates, rotation=45,
            ha='right', fontsize=8, color='white')
        ax2.set_ylabel("Score %", color='white')
        ax2.set_facecolor('#0E1117')
        fig2.patch.set_facecolor('#0E1117')
        ax2.tick_params(colors='white')
        ax2.legend(
            facecolor='#1E2130', labelcolor='white')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#333')
        plt.tight_layout()
        st.pyplot(fig2)

    # ── PDF Report ────────────────────────────────────
    st.divider()
    st.markdown("### 📄 Download Your Report")
    patient_name = st.text_input(
        "Your name (optional)",
        placeholder="Leave blank for anonymous"
    )

    if st.button("📄 Generate PDF Report"):
        with st.spinner("Generating report..."):
            name = (patient_name
                    if patient_name else "Anonymous")
            try:
                filename = generate_pdf_report(
                    healthy_pct, risk_pct, name)
                with open(filename, 'rb') as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label="⬇️ Download Your Report",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf"
                )
                st.success(
                    "✅ Click button above to download!")
            except Exception as e:
                st.error(f"Report error: {e}")

    # ── Disclaimer ────────────────────────────────────
    st.divider()
    st.markdown("""
    <div class="disclaimer">
    ⚕️ <b>Medical Disclaimer:</b> VoiceVitals is a
    screening and awareness tool only. It is NOT a
    medical diagnosis. Always consult a qualified
    neurologist for clinical evaluation.
    </div>
    """, unsafe_allow_html=True)

# ── Past Screenings ───────────────────────────────────
st.divider()
history = load_history()
if len(history) > 0:
    with st.expander(
            f"📋 Past Screenings ({len(history)})"):
        for h in reversed(history):
            st.write(
                f"**{h['date']}** — "
                f"Healthy: {h['healthy_score']}% | "
                f"At-Risk: {h['risk_score']}%"
            )