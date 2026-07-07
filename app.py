import streamlit as st
import joblib
import requests
import pandas as pd
from scipy.sparse import csr_matrix
from math import radians, sin, cos, sqrt, atan2
import time

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

selected_symptoms = st.sidebar.multiselect(
    "Search symptoms",
    symptoms_list
)

# ----------------------------
# DISEASE PREDICTION
# ----------------------------

if st.sidebar.button("🔍 Predict Disease"):

    if len(selected_symptoms) == 0:

        st.warning("Please select symptoms")

    else:

        features = [0] * num_features

        for s in selected_symptoms:
            idx = symptoms_list.index(s)
            features[idx] = 1

        input_vector = csr_matrix([features])

        probs = clf.predict_proba(input_vector)[0]

        top_n = min(3, len(probs))
        top_indices = probs.argsort()[-top_n:][::-1]

        st.subheader("🧠 Possible Diseases")

        cols = st.columns(top_n)

        max_severity = 0

        for i, idx in enumerate(top_indices):

            disease = le.inverse_transform([idx])[0]

            doctor = get_specialist(disease)

            severity_score = calculate_severity(
                selected_symptoms,
                disease
            )

            max_severity = max(max_severity, severity_score)

            risk_level = get_risk_level(severity_score)

            with cols[i]:

                st.markdown(f"""
                <div class="card">

                <h3>🦠 {disease}</h3>

                <p style="color:#666">
                Recommended Specialist
                </p>

                <h4 style="color:#2C7BE5">
                Consult: {doctor}
                </h4>

                <hr>

                <h4>
                🧠 Severity Score: {severity_score}/100
                </h4>

                <h4>
                ⚠ {risk_level}
                </h4>

                </div>
                """, unsafe_allow_html=True)

        # ----------------------------
        # ALERTS (based on highest severity across shown diseases)
        # ----------------------------

        if max_severity >= 70:

            st.error(
                "🚨 HIGH RISK DETECTED - Immediate medical attention recommended"
            )

        elif max_severity >= 40:

            st.warning(
                "⚠ Moderate severity detected. Consult doctor soon."
            )

        else:

            st.success(
                "🟢 Low severity detected."
            )

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
