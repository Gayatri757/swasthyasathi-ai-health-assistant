# 🩺 SwasthyaSathi — AI-Powered Health Assistant

SwasthyaSathi predicts likely diseases from user-reported symptoms, explains results in plain language, estimates severity/urgency, helps users book a consultation, and finds nearby hospitals in real time.

**🔗 Live App:** https://swasthyasathi-ai-health-assistant.streamlit.app/

> ⚠️ **Disclaimer:** This is an educational/portfolio project, not a medical device. Predictions, severity scores, doctor profiles, and booking are for demonstration purposes and should never replace professional medical advice.

---

## ✨ Features

### Core ML Pipeline
- **Disease prediction** using an `SGDClassifier` trained on 246,945 medical records, achieving **87% accuracy** — outperforming a Random Forest baseline (39%).
- **Specialist recommendation** — maps predicted conditions to the right type of doctor (Cardiologist, Psychiatrist, Dermatologist, etc.).
- **Severity scoring (0–100) and risk classification** (🟢 Low / 🟡 Medium / 🔴 High) with real-time alerts on next steps.

### Generative AI Layer (Llama 3.3 via Groq — free tier)
- **Natural-language symptom intake** — describe symptoms in plain English (e.g. *"I've had a headache and feel dizzy for two days"*) and the AI extracts and matches them to the model's known symptom list, then runs prediction automatically in the same step.
- **AI-generated result summaries** — a plain-language explanation of the prediction, generated per-request rather than a static template.
- **Context-aware follow-up chat** — ask questions like *"what does acute stress reaction mean?"* and get answers grounded in your actual results.

### Doctor Booking (Simulated)
- Browse doctors by recommended specialist, with ratings, review counts, experience, and consultation fees.
- Book a demo appointment (date + time slot) and view a simulated booking confirmation with a generated booking ID.
- *Note: doctor profiles, fees, and availability are mock data for demonstration — there is no real payment gateway or live doctor network.*

### Hospital Finder
- Enter a city and search radius to find nearby hospitals via the **Overpass API** (OpenStreetMap), with fallback search via Nominatim if Overpass is unavailable.
- View results on an interactive map, get sorted distances, and open turn-by-turn directions in Google Maps.
- Simulated emergency ambulance dispatch flow with ETA estimate.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Scikit-learn (`SGDClassifier`), joblib |
| GenAI | Groq API (Llama 3.3 70B) |
| Frontend / App | Streamlit |
| Geolocation & Maps | Nominatim, Overpass API, Google Maps (directions) |
| Data | NumPy, Pandas, SciPy (sparse matrices) |

---

## 🚀 Running Locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/Gayatri757/swasthyasathi-ai-health-assistant.git
   cd swasthyasathi-ai-health-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your free Groq API key**

   Create `.streamlit/secrets.toml` (already gitignored — never commit this file):
   ```toml
   GROQ_API_KEY = "your-groq-key-here"
   ```
   Get a free key (no credit card required) at [console.groq.com](https://console.groq.com).

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## 📂 Model Files

The app expects these pre-trained artifacts in the project root:
- `disease_model.pkl` — trained `SGDClassifier`
- `label_encoder.pkl` — `LabelEncoder` for disease labels
- `symptoms_list.pkl` — ordered list of known symptom strings used as model features

---

## 🧭 How It Works (High Level)

```
User Input (free text OR manual symptom selection)
        │
        ▼
[GenAI: symptom extraction]  ──(if free text)──▶  matched symptoms
        │
        ▼
Feature vector → SGDClassifier → Top-3 predicted diseases
        │
        ▼
Severity scoring + specialist mapping
        │
        ▼
[GenAI: plain-language summary]  +  [GenAI: follow-up chat]
        │
        ▼
Doctor booking (simulated)  +  Nearby hospital search (Overpass/Nominatim)
```

---

## 🔭 Possible Future Improvements

- Replace mock doctor directory with a real practitioner database
- Add multi-language support for symptom input
- Voice-based symptom input (speech-to-text)
- Persist booking history and prediction logs per user (currently session-only)

---

## 📄 License

This project is for educational and portfolio purposes.
