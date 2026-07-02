# 🎙️ VoiceVitals
### Early Neurological Voice Screening App

VoiceVitals analyzes acoustic patterns in your voice —
pitch stability, jitter, shimmer — linked to early 
neurological motor symptoms associated with Parkinson's.

---

## 📊 Validation
- Dataset: UCI Parkinson's Voice Dataset
- 195 real clinical recordings from 31 patients
- Test Accuracy: 92%
- At-Risk Detection Rate: 97%

---

## ✨ Features
- 🎙️ 15-30 second voice recording
- 🤖 AI analysis using trained clinical model
- 📊 Healthy vs At-Risk pattern matching
- 📈 Trend tracking across multiple screenings
- 📄 PDF report download for doctor consultation

---

## 🛠️ Tech Stack
- Python
- Librosa — audio feature extraction
- Scikit-learn — machine learning model
- Streamlit — web interface
- ReportLab — PDF generation

---

## 🚀 How To Run

1. Clone this repository
2. Install dependencies:
   pip install -r requirements.txt
3. Download UCI Parkinson's dataset from Kaggle
   and save as parkinsons.csv
4. Train the model:
   python retrain_model.py
5. Run the app:
   streamlit run app.py

---

## ⚕️ Disclaimer
VoiceVitals is a screening and awareness tool only.
It is NOT a medical diagnosis. Results are based on
acoustic pattern matching against a clinical dataset.
Always consult a qualified neurologist for evaluation.

---

## 👨‍💻 Built By
Kanika — First Year Student
Built for Samsung Solve for Tomorrow 2026

---

## 📚 Research Backing
- MIT Media Lab — vocal biomarker research
- Mayo Clinic — voice and neurological conditions
- Carnegie Mellon University — acoustic health markers
- UCI Machine Learning Repository — Parkinson's dataset
