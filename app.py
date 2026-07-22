import streamlit as st
import joblib
import requests
import pandas as pd
from scipy.sparse import csr_matrix
from math import radians, sin, cos, sqrt, atan2
import time
import json
from groq import Groq

# ----------------------------
# GEN AI CLIENT (CACHED) — Groq: free tier, no credit card required
# ----------------------------

@st.cache_resource
def get_ai_client():
    api_key = st.secrets.get("GROQ_API_KEY", None)
    if not api_key:
        return None
    return Groq(api_key=api_key)

ai_client = get_ai_client()
AI_MODEL = "llama-3.3-70b-versatile"

# ----------------------------
# NOMINATIM HELPERS (cached + retry-on-429)
# ----------------------------

def safe_get(url, params, headers, retries=2, backoff=1.5):
    """GET with basic retry on 429 (rate limit) responses."""
    last_resp = None
    for attempt in range(retries + 1):
        resp = requests.get(url, params=params, headers=headers, timeout=20)
        last_resp = resp
        if resp.status_code == 429:
            time.sleep(backoff)
            continue
        resp.raise_for_status()
        return resp
    # If we got here, all retries hit 429 — raise the last response's error
    last_resp.raise_for_status()
    return last_resp


@st.cache_data(ttl=3600, show_spinner=False)
def geocode_location(location):
    """Geocode a city/place name to (lat, lon). Cached for 1 hour to avoid
    re-hitting Nominatim on repeated searches for the same location."""
    headers = {
        "User-Agent": "SwasthyaSathi-App/1.0 (contact: gayatriadatiya344@gmail.com)"
    }
    resp = safe_get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": location, "format": "json"},
        headers=headers
    )
    return resp.json()


@st.cache_data(ttl=3600, show_spinner=False)
def geocode_hospital_fallback(location):
    """Fallback hospital search via Nominatim, cached for 1 hour."""
    headers = {
        "User-Agent": "SwasthyaSathi-App/1.0 (contact: gayatriadatiya344@gmail.com)"
    }
    resp = safe_get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": "hospital " + location, "format": "json", "limit": 10},
        headers=headers
    )
    return resp.json()

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="SwasthyaSathi",
    page_icon="🩺",
    layout="wide"
)

# ----------------------------
# CUSTOM CSS
# ----------------------------

st.markdown("""
<style>

.stApp{
background: linear-gradient(to right,#eef5ff,#f8fbff);
}

.main-title{
font-size:38px;
font-weight:800;
text-align:center;
color:#1f4db8;
margin-bottom:10px;
}

.subtitle{
text-align:center;
font-size:18px;
color:#555;
margin-bottom:30px;
}

.card{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 6px 18px rgba(0,0,0,0.12);
text-align:center;

min-height:260px;
height:auto;

display:flex;
flex-direction:column;
justify-content:center;
align-items:center;

transition:0.3s;

margin-bottom:20px;

overflow-wrap:break-word;
word-wrap:break-word;
}

.card:hover{
transform:scale(1.02);
box-shadow:0px 8px 22px rgba(0,0,0,0.18);
}

.hospital-card{
background:white;
padding:18px;
border-radius:12px;
box-shadow:0px 3px 10px rgba(0,0,0,0.08);
margin-bottom:12px;
}

section[data-testid="stSidebar"]{
background:#ffffff;
border-right:1px solid #eee;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD MODELS (CACHED — fixes memory crash from reloading on every rerun)
# ----------------------------

@st.cache_resource
def load_models():
    clf = joblib.load("disease_model.pkl")
    le = joblib.load("label_encoder.pkl")
    symptoms_list = joblib.load("symptoms_list.pkl")
    return clf, le, symptoms_list

clf, le, symptoms_list = load_models()

num_features = len(symptoms_list)

# ----------------------------
# GEN AI: EXTRACT SYMPTOMS FROM FREE TEXT
# ----------------------------

def ai_extract_symptoms(user_text, symptoms_list):
    """Use Claude to map a free-text description to entries in symptoms_list."""
    if ai_client is None:
        return [], "AI features are not configured (missing API key)."

    prompt = f"""You are a medical symptom-matching assistant. A user described their
symptoms in their own words. Match their description to entries from this
exact list of known symptoms (only use exact strings from the list, do not
invent new ones):

{json.dumps(symptoms_list)}

User's description: "{user_text}"

Return ONLY a JSON array of matching strings from the list, nothing else.
If nothing matches, return an empty array []."""

    try:
        response = ai_client.chat.completions.create(
            model=AI_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        matched = json.loads(text)
        matched = [s for s in matched if s in symptoms_list]
        return matched, None
    except Exception as e:
        return [], f"Could not process with AI: {e}"


# ----------------------------
# GEN AI: NATURAL-LANGUAGE RESULT SUMMARY
# ----------------------------

def ai_generate_summary(disease_results):
    """Generate a friendly, plain-language summary of the prediction results."""
    if ai_client is None:
        return None

    results_text = "\n".join([
        f"- {d['disease']}: severity {d['severity']}/100 ({d['risk']}), "
        f"recommended specialist: {d['doctor']}"
        for d in disease_results
    ])

    prompt = f"""You are a calm, clear health assistant (not a doctor). Based on
this AI model's output, write a short (4-6 sentence) plain-language summary
for the user. Explain what the results suggest in simple terms, give general
non-prescriptive precautions, and remind them this is not a diagnosis and they
should consult a real doctor for confirmation. Do not invent new medical facts
beyond what's given.

Model output:
{results_text}"""

    try:
        response = ai_client.chat.completions.create(
            model=AI_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(AI summary unavailable: {e})"


# ----------------------------
# GEN AI: FOLLOW-UP CHAT
# ----------------------------

def ai_chat_reply(chat_history, context):
    """Generate a chat reply grounded in the current prediction context."""
    if ai_client is None:
        return "AI chat is not configured (missing API key)."

    system_prompt = f"""You are a calm, helpful health information assistant
embedded in an app called SwasthyaSathi. You are NOT a doctor and must never
give a definitive diagnosis or prescribe medication. Always recommend
consulting a real doctor for anything serious. Keep answers short (3-5
sentences) and easy to understand.

Context from the user's recent symptom check: {context}"""

    messages = [{"role": "system", "content": system_prompt}]
    messages += [{"role": m["role"], "content": m["content"]} for m in chat_history]

    try:
        response = ai_client.chat.completions.create(
            model=AI_MODEL,
            max_tokens=400,
            messages=messages
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(AI chat unavailable: {e})"

# ----------------------------
# SPECIALIST DETECTION
# ----------------------------

def get_specialist(disease):

    d = disease.lower()

    if any(x in d for x in ["stress", "anxiety", "panic", "depression"]):
        return "Psychiatrist"

    elif any(x in d for x in ["heart", "cardio", "hypertension"]):
        return "Cardiologist"

    elif any(x in d for x in ["lung", "asthma", "pneumonia", "respiratory"]):
        return "Pulmonologist"

    elif any(x in d for x in ["brain", "migraine", "epilepsy", "neuro"]):
        return "Neurologist"

    elif any(x in d for x in ["skin", "acne", "eczema", "dermat"]):
        return "Dermatologist"

    elif any(x in d for x in ["kidney", "urinary", "bladder"]):
        return "Urologist"

    elif any(x in d for x in ["pregnancy", "vaginal", "uterus", "menstrual"]):
        return "Gynecologist"

    elif any(x in d for x in ["diabetes", "thyroid", "hormone"]):
        return "Endocrinologist"

    else:
        return "General Physician"

# ----------------------------
# DOCTOR DIRECTORY (SIMULATED DATA)
# ----------------------------
# NOTE: This is demo/mock data for portfolio purposes — not a real doctor
# network, live availability system, or payment gateway.

DOCTOR_DIRECTORY = {
    "General Physician": [
        {"name": "Dr. Anjali Deshmukh", "rating": 4.7, "reviews": 312, "experience": 9, "fee": 400},
        {"name": "Dr. Rohan Kulkarni", "rating": 4.5, "reviews": 198, "experience": 6, "fee": 300},
        {"name": "Dr. Sunita Patil", "rating": 4.8, "reviews": 456, "experience": 14, "fee": 500},
    ],
    "Psychiatrist": [
        {"name": "Dr. Meera Nair", "rating": 4.9, "reviews": 521, "experience": 12, "fee": 800},
        {"name": "Dr. Karan Malhotra", "rating": 4.6, "reviews": 214, "experience": 7, "fee": 650},
        {"name": "Dr. Priya Shah", "rating": 4.7, "reviews": 289, "experience": 10, "fee": 700},
    ],
    "Cardiologist": [
        {"name": "Dr. Vikram Rao", "rating": 4.8, "reviews": 401, "experience": 16, "fee": 900},
        {"name": "Dr. Neha Joshi", "rating": 4.6, "reviews": 178, "experience": 8, "fee": 750},
    ],
    "Pulmonologist": [
        {"name": "Dr. Sameer Khan", "rating": 4.7, "reviews": 233, "experience": 11, "fee": 700},
        {"name": "Dr. Ayesha Sheikh", "rating": 4.5, "reviews": 156, "experience": 7, "fee": 600},
    ],
    "Neurologist": [
        {"name": "Dr. Arvind Menon", "rating": 4.9, "reviews": 367, "experience": 15, "fee": 950},
        {"name": "Dr. Kavita Iyer", "rating": 4.6, "reviews": 201, "experience": 9, "fee": 800},
    ],
    "Dermatologist": [
        {"name": "Dr. Riya Kapoor", "rating": 4.7, "reviews": 289, "experience": 8, "fee": 500},
        {"name": "Dr. Aditya Verma", "rating": 4.5, "reviews": 145, "experience": 6, "fee": 450},
    ],
    "Urologist": [
        {"name": "Dr. Nikhil Bhatt", "rating": 4.6, "reviews": 167, "experience": 10, "fee": 700},
    ],
    "Gynecologist": [
        {"name": "Dr. Shalini Reddy", "rating": 4.8, "reviews": 398, "experience": 13, "fee": 650},
        {"name": "Dr. Pooja Agarwal", "rating": 4.7, "reviews": 245, "experience": 9, "fee": 600},
    ],
    "Endocrinologist": [
        {"name": "Dr. Manish Gupta", "rating": 4.7, "reviews": 210, "experience": 11, "fee": 750},
    ],
}


def get_doctors_for_specialist(specialist):
    return DOCTOR_DIRECTORY.get(specialist, DOCTOR_DIRECTORY["General Physician"])


def star_display(rating):
    full_stars = int(rating)
    return "⭐" * full_stars + f" {rating}"

# ----------------------------
# SEVERITY SCORE
# ----------------------------

def calculate_severity(symptoms, disease):

    severity_weights = {
        "chest pain": 30,
        "breathlessness": 35,
        "shortness of breath": 35,
        "high fever": 20,
        "fatigue": 10,
        "dizziness": 15,
        "headache": 10,
        "nausea": 10,
        "vomiting": 15,
        "cough": 10,
        "loss of consciousness": 50,
        "severe pain": 30
    }

    score = 0

    for s in symptoms:
        score += severity_weights.get(s.lower(), 5)

    disease = disease.lower()

    if any(x in disease for x in ["heart", "cardio"]):
        score += 40

    elif any(x in disease for x in ["lung", "respiratory", "asthma"]):
        score += 30

    elif any(x in disease for x in ["infection", "fever"]):
        score += 20

    elif any(x in disease for x in ["diabetes"]):
        score += 15

    else:
        score += 10

    return min(score, 100)

# ----------------------------
# RISK LEVEL
# ----------------------------

def get_risk_level(score):

    if score < 30:
        return "🟢 Low Risk"

    elif score < 70:
        return "🟡 Medium Risk"

    else:
        return "🔴 High Risk"

# ----------------------------
# DISTANCE FUNCTION
# ----------------------------

def calculate_distance(lat1, lon1, lat2, lon2):

    R = 6371

    lat1, lon1, lat2, lon2 = map(
        radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

# ----------------------------
# AMBULANCE ETA
# ----------------------------

def estimate_eta(distance_km):

    speed = 40

    return round((distance_km / speed) * 60, 1)

# ----------------------------
# HEADER
# ----------------------------

st.markdown("""
<div class="main-title">
🩺 SwasthyaSathi
</div>

<div class="subtitle">
Your AI Powered Health Assistant
<br>
Predict diseases, analyze severity and find nearby hospitals
</div>
""", unsafe_allow_html=True)

# ----------------------------
# SIDEBAR
# ----------------------------

st.sidebar.header("🧾 Select Symptoms")

if "selected_symptoms_ms" not in st.session_state:
    st.session_state.selected_symptoms_ms = []

if "run_prediction_now" not in st.session_state:
    st.session_state.run_prediction_now = False

if "prediction_symptoms" not in st.session_state:
    st.session_state.prediction_symptoms = []


def run_prediction(symptoms_for_prediction):
    """Runs the full disease prediction + severity + AI summary pipeline
    for a given list of symptoms, and renders the results."""

    if len(symptoms_for_prediction) == 0:
        st.warning("Please select or describe at least one symptom.")
        return

    features = [0] * num_features

    for s in symptoms_for_prediction:
        idx = symptoms_list.index(s)
        features[idx] = 1

    input_vector = csr_matrix([features])

    probs = clf.predict_proba(input_vector)[0]

    top_n = min(3, len(probs))
    top_indices = probs.argsort()[-top_n:][::-1]

    st.subheader("🧠 Possible Diseases")

    cols = st.columns(top_n)

    max_severity = 0
    disease_results = []

    for i, idx in enumerate(top_indices):

        disease = le.inverse_transform([idx])[0]
        confidence = probs[idx] * 100
        doctor = get_specialist(disease)
        severity_score = calculate_severity(symptoms_for_prediction, disease)
        max_severity = max(max_severity, severity_score)
        risk_level = get_risk_level(severity_score)

        disease_results.append({
            "disease": disease,
            "doctor": doctor,
            "severity": severity_score,
            "risk": risk_level,
            "confidence": confidence
        })

        with cols[i]:
            st.markdown(f"""
            <div class="card">
            <h3>🦠 {disease}</h3>
            <p style="color:#666">Recommended Specialist</p>
            <h4 style="color:#2C7BE5">Consult: {doctor}</h4>
            <hr>
            <h4>🎯 Confidence: {confidence:.1f}%</h4>
            <h4>🧠 Severity Score: {severity_score}/100</h4>
            <h4>⚠ {risk_level}</h4>
            </div>
            """, unsafe_allow_html=True)
            st.progress(min(int(confidence), 100))

    if max_severity >= 70:
        st.error("🚨 HIGH RISK DETECTED - Immediate medical attention recommended")
    elif max_severity >= 40:
        st.warning("⚠ Moderate severity detected. Consult doctor soon.")
    else:
        st.success("🟢 Low severity detected.")

    st.subheader("💬 AI Summary")

    if ai_client is None:
        st.info(
            "Add a GROQ_API_KEY in Streamlit secrets to enable "
            "AI-generated plain-language summaries and chat."
        )
    else:
        with st.spinner("Generating a plain-language summary..."):
            ai_summary = ai_generate_summary(disease_results)
        st.info(ai_summary)

    st.session_state.last_prediction_context = (
        f"Symptoms reported: {', '.join(symptoms_for_prediction)}. "
        + "; ".join([
            f"{d['disease']} (severity {d['severity']}/100, {d['risk']}, "
            f"see a {d['doctor']})" for d in disease_results
        ])
    )
    st.session_state.last_disease_results = disease_results


st.sidebar.markdown("**🤖 Describe symptoms in your own words**")
free_text_input = st.sidebar.text_area(
    "e.g. \"I've had a bad headache and feel dizzy for two days\"",
    key="free_text_symptoms",
    height=80
)

if st.sidebar.button("✨ Analyze My Symptoms (AI)"):
    if not free_text_input.strip():
        st.sidebar.warning("Please describe your symptoms first.")
    elif ai_client is None:
        st.sidebar.error("AI features need a GROQ_API_KEY set in Streamlit secrets.")
    else:
        with st.sidebar:
            with st.spinner("Analyzing your description..."):
                matched, err = ai_extract_symptoms(free_text_input, symptoms_list)
        if err:
            st.sidebar.error(err)
        elif not matched:
            st.sidebar.info("No matching symptoms found — try adding more detail, or select manually below.")
        else:
            st.session_state.selected_symptoms_ms = matched
            st.sidebar.success(f"Matched: {', '.join(matched)}")
            # Trigger prediction immediately — no second click needed
            st.session_state.run_prediction_now = True
            st.session_state.prediction_symptoms = matched

st.sidebar.markdown("**Or select manually**")

selected_symptoms = st.sidebar.multiselect(
    "Search symptoms",
    symptoms_list,
    key="selected_symptoms_ms"
)

if st.sidebar.button("🔍 Predict Disease"):
    st.session_state.run_prediction_now = True
    st.session_state.prediction_symptoms = selected_symptoms

# ----------------------------
# RUN PREDICTION (triggered by either button above)
# ----------------------------

if st.session_state.run_prediction_now:
    run_prediction(st.session_state.prediction_symptoms)

# ----------------------------
# BOOK ONLINE CONSULTATION
# ----------------------------

st.divider()
st.subheader("👨‍⚕️ Book Online Consultation")
st.caption("Demo booking flow — doctor profiles and availability shown here are simulated for demonstration purposes.")

if "bookings" not in st.session_state:
    st.session_state.bookings = []

recent_results = st.session_state.get("last_disease_results", [])

if not recent_results:
    st.info("Run a symptom prediction above to see recommended specialists here.")
else:
    recommended_specialists = list(dict.fromkeys([d["doctor"] for d in recent_results]))

    specialist_tabs = st.tabs(recommended_specialists)

    for tab, specialist in zip(specialist_tabs, recommended_specialists):
        with tab:
            doctors = get_doctors_for_specialist(specialist)

            for doc in doctors:
                with st.container():
                    dcol1, dcol2, dcol3 = st.columns([3, 2, 2])

                    with dcol1:
                        st.markdown(f"**{doc['name']}**")
                        st.caption(f"{specialist} · {doc['experience']} yrs experience")

                    with dcol2:
                        st.markdown(f"{star_display(doc['rating'])}")
                        st.caption(f"{doc['reviews']} reviews")

                    with dcol3:
                        st.markdown(f"**₹{doc['fee']}** / consultation")

                    booking_key = f"book_{specialist}_{doc['name']}"

                    with st.expander(f"📅 Book with {doc['name']}"):
                        b_date = st.date_input("Preferred date", key=f"date_{booking_key}")
                        b_time = st.selectbox(
                            "Preferred time slot",
                            ["10:00 AM", "12:00 PM", "3:00 PM", "5:00 PM", "7:00 PM"],
                            key=f"time_{booking_key}"
                        )

                        if st.button(f"Confirm Booking (₹{doc['fee']})", key=f"confirm_{booking_key}"):
                            booking_id = f"SS-{len(st.session_state.bookings) + 1001}"
                            st.session_state.bookings.append({
                                "id": booking_id,
                                "doctor": doc["name"],
                                "specialist": specialist,
                                "date": str(b_date),
                                "time": b_time,
                                "fee": doc["fee"]
                            })
                            st.success(
                                f"✅ Appointment confirmed with {doc['name']} on "
                                f"{b_date} at {b_time}. Booking ID: {booking_id}"
                            )

                    st.markdown("---")

    if st.session_state.bookings:
        with st.expander(f"📋 My Bookings ({len(st.session_state.bookings)})"):
            for b in st.session_state.bookings:
                st.markdown(
                    f"**{b['id']}** — {b['doctor']} ({b['specialist']}) · "
                    f"{b['date']} at {b['time']} · ₹{b['fee']}"
                )

# ----------------------------
# GEN AI: FOLLOW-UP CHAT
# ----------------------------

st.divider()
st.subheader("🤖 Ask Follow-up Questions")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if ai_client is None:
    st.caption("Add a GROQ_API_KEY in Streamlit secrets to enable this chat.")
else:
    st.caption("Ask general questions about your results — e.g. \"What does acute stress reaction mean?\"")

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_question = st.chat_input("Type your question here...")

    if user_question:
        st.session_state.chat_messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        context = st.session_state.get(
            "last_prediction_context",
            "No prediction has been run yet in this session."
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = ai_chat_reply(st.session_state.chat_messages, context)
            st.write(reply)

        st.session_state.chat_messages.append({"role": "assistant", "content": reply})

# ----------------------------
# HOSPITAL FINDER
# ----------------------------

st.divider()

st.subheader("🏥 Nearby Hospitals")

col1, col2 = st.columns(2)

with col1:
    location = st.text_input(
        "Enter City",
        "Pune"
    )

with col2:
    radius = st.slider(
        "Search Radius (meters)",
        1000,
        10000,
        3000
    )

# ----------------------------
# FIND HOSPITALS
# ----------------------------

if st.button("Find Hospitals"):

    try:

        # Nominatim requires a descriptive User-Agent with contact info,
        # or it silently blocks/rate-limits requests from cloud hosts.
        headers = {
            "User-Agent": "SwasthyaSathi-App/1.0 (contact: gayatriadatiya344@gmail.com)"
        }

        # ----------------------------
        # GET LOCATION COORDINATES (cached + retry-on-429)
        # ----------------------------

        geo_data = geocode_location(location)

        if not geo_data:

            st.error("Location not found")
            st.stop()

        user_lat = float(geo_data[0]["lat"])
        user_lon = float(geo_data[0]["lon"])

        # ----------------------------
        # OVERPASS QUERY
        # ----------------------------

        query = f"""
        [out:json];
        node["amenity"="hospital"]
        (around:{radius},{user_lat},{user_lon});
        out;
        """

        hospital_list = []

        try:

            response = requests.post(
                "https://overpass-api.de/api/interpreter",
                data={"data": query},
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:

                data = response.json()
                hospitals = data.get("elements", [])

                for h in hospitals:

                    lat = h.get("lat")
                    lon = h.get("lon")

                    if lat is None or lon is None:
                        continue

                    tags = h.get("tags", {})

                    name = tags.get(
                        "name",
                        "Unknown Hospital"
                    )

                    dist = calculate_distance(
                        user_lat,
                        user_lon,
                        lat,
                        lon
                    )

                    hospital_list.append({
                        "Hospital": name,
                        "Distance (km)": round(dist, 2),
                        "Latitude": lat,
                        "Longitude": lon
                    })

        except requests.exceptions.RequestException:
            st.info("Overpass search unavailable, using fallback search...")

        # ----------------------------
        # FALLBACK SEARCH
        # ----------------------------

        if not hospital_list:

            fallback_data = geocode_hospital_fallback(location)

            for h in fallback_data:

                lat = float(h["lat"])
                lon = float(h["lon"])

                dist = calculate_distance(
                    user_lat,
                    user_lon,
                    lat,
                    lon
                )

                hospital_list.append({
                    "Hospital": h.get(
                        "display_name",
                        "Hospital"
                    ),
                    "Distance (km)": round(dist, 2),
                    "Latitude": lat,
                    "Longitude": lon
                })

        # ----------------------------
        # NO HOSPITALS
        # ----------------------------

        if not hospital_list:

            st.error("No hospitals found")
            st.stop()

        hospital_list = sorted(
            hospital_list,
            key=lambda x: x["Distance (km)"]
        )

        st.success(
            f"{len(hospital_list)} hospitals found"
        )

        # ----------------------------
        # TABLE
        # ----------------------------

        df = pd.DataFrame(hospital_list)

        st.dataframe(
            df[["Hospital", "Distance (km)"]],
            use_container_width=True
        )

        # ----------------------------
        # MAP
        # ----------------------------

        st.subheader("🗺 Nearby Hospitals Map")

        map_df = pd.DataFrame({
            "lat": [h["Latitude"] for h in hospital_list],
            "lon": [h["Longitude"] for h in hospital_list]
        })

        st.map(map_df)

        # ----------------------------
        # HOSPITAL CARDS
        # ----------------------------

        st.subheader("🚗 Directions")

        for h in hospital_list[:5]:

            directions = (
                f"https://www.google.com/maps/dir/?api=1"
                f"&destination={h['Latitude']},{h['Longitude']}"
            )

            st.markdown(f"""
            <div class="hospital-card">

            🏥 <b>{h['Hospital']}</b><br>

            📍 Distance:
            {h['Distance (km)']} km<br><br>

            <a href="{directions}" target="_blank">
            🧭 Open in Google Maps
            </a>

            </div>
            """, unsafe_allow_html=True)

        # ----------------------------
        # EMERGENCY SYSTEM
        # ----------------------------

        st.divider()

        st.subheader("🚑 Emergency Ambulance System")

        if st.button("🚨 Call Ambulance"):

            nearest = hospital_list[0]

            eta = estimate_eta(
                nearest["Distance (km)"]
            )

            st.error("🚑 Ambulance Dispatched")

            st.write(
                f"🏥 Hospital: {nearest['Hospital']}"
            )

            st.write(
                f"📍 Distance: {nearest['Distance (km)']} km"
            )

            st.write(
                f"⏱ ETA: {eta} minutes"
            )

            progress = st.progress(0)

            for i in range(100):

                time.sleep(0.02)

                progress.progress(i + 1)

            st.success(
                "🚑 Ambulance Arrived!"
            )

            route_url = (
                f"https://www.google.com/maps/dir/?api=1"
                f"&origin={user_lat},{user_lon}"
                f"&destination={nearest['Latitude']},{nearest['Longitude']}"
            )

            st.markdown(
                f"[🧭 View Route on Google Maps]({route_url})"
            )

    except requests.exceptions.HTTPError as e:

        if e.response is not None and e.response.status_code == 429:
            st.error(
                "🚦 The hospital/location lookup service is rate-limiting requests right now "
                "(too many searches in a short time). Please wait about 30-60 seconds and try again."
            )
        else:
            st.error("Error fetching hospitals — the location/hospital service returned an error.")
            st.write(str(e))

    except requests.exceptions.RequestException as e:

        st.error("Error fetching hospitals — the location/hospital service may be temporarily unavailable.")
        st.write(str(e))

    except Exception as e:

        st.error("Unexpected error while fetching hospitals")
        st.write(str(e))

# ----------------------------
# DISCLAIMER
# ----------------------------

st.divider()

st.warning(
    "⚠ This AI system is for educational purposes only. Always consult a doctor."
)
