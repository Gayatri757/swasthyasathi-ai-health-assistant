import streamlit as st
import joblib
import requests
import pandas as pd
from scipy.sparse import csr_matrix
from math import radians, cos, sin, asin, sqrt

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="SwasthyaSathi",
    page_icon="🩺",
    layout="wide"
)

# ----------------------------
# UI STYLE
# ----------------------------
st.markdown("""
<style>
.stApp{
background: linear-gradient(to right,#eef5ff,#f8fbff);
}

.main-title{
font-size:48px;
font-weight:800;
text-align:center;
color:#1f4db8;
margin-bottom:10px;
}

.subtitle{
text-align:center;
font-size:20px;
color:#555;
margin-bottom:30px;
}

.card{
background:white;
padding:25px;
border-radius:15px;
box-shadow:0px 6px 18px rgba(0,0,0,0.12);
text-align:center;
height:200px;
display:flex;
flex-direction:column;
justify-content:center;
align-items:center;
}

.hospital-card{
background:white;
padding:18px;
border-radius:12px;
box-shadow:0px 3px 10px rgba(0,0,0,0.08);
margin-bottom:12px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL
# ----------------------------
clf = joblib.load("disease_model.pkl")
le = joblib.load("label_encoder.pkl")
symptoms_list = joblib.load("symptoms_list.pkl")
num_features = len(symptoms_list)

# ----------------------------
# SPECIALIST FUNCTION
# ----------------------------
def get_specialist(disease):
    d = disease.lower()

    if any(x in d for x in ["stress","anxiety","panic","depression"]):
        return "Psychiatrist"
    elif any(x in d for x in ["heart","cardio","hypertension"]):
        return "Cardiologist"
    elif any(x in d for x in ["lung","asthma","pneumonia","respiratory"]):
        return "Pulmonologist"
    elif any(x in d for x in ["brain","migraine","epilepsy","neuro"]):
        return "Neurologist"
    elif any(x in d for x in ["skin","acne","eczema","dermat"]):
        return "Dermatologist"
    elif any(x in d for x in ["kidney","urinary","bladder"]):
        return "Urologist"
    elif any(x in d for x in ["pregnancy","uterus","menstrual"]):
        return "Gynecologist"
    elif any(x in d for x in ["diabetes","thyroid","hormone"]):
        return "Endocrinologist"
    else:
        return "General Physician"

# ----------------------------
# DISTANCE FUNCTION
# ----------------------------
def calculate_distance(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return c * 6371

# ----------------------------
# HEADER
# ----------------------------
st.markdown("""
<div class="main-title">🩺 SwasthyaSathi</div>
<div class="subtitle">AI Health Assistant - Disease Prediction + Hospital Finder</div>
""", unsafe_allow_html=True)

# ----------------------------
# SIDEBAR - DISEASE PREDICTION
# ----------------------------
st.sidebar.header("🧾 Select Symptoms")

selected_symptoms = st.sidebar.multiselect("Search symptoms", symptoms_list)

if st.sidebar.button("🔍 Predict Disease"):

    if not selected_symptoms:
        st.warning("Please select symptoms")
    else:
        features = [0] * num_features

        for s in selected_symptoms:
            features[symptoms_list.index(s)] = 1

        input_vector = csr_matrix([features])

        probs = clf.predict_proba(input_vector)[0]
        top3 = probs.argsort()[-3:][::-1]

        st.subheader("🧠 Possible Diseases")

        cols = st.columns(3)

        for i, idx in enumerate(top3):
            disease = le.inverse_transform([idx])[0]
            doctor = get_specialist(disease)

            with cols[i]:
                st.markdown(f"""
                <div class="card">
                <h3>🦠 {disease}</h3>
                <p>Consult: <b>{doctor}</b></p>
                </div>
                """, unsafe_allow_html=True)

# ----------------------------
# HOSPITAL INPUTS (IMPORTANT FIX)
# ----------------------------
st.divider()
st.subheader("🏥 Nearby Hospitals")

col1, col2 = st.columns(2)

with col1:
    location = st.text_input("Enter City", "Pune")

with col2:
    radius = st.slider("Search Radius (meters)", 1000, 10000, 3000)

# ----------------------------
# HOSPITAL SEARCH
# ----------------------------
if st.button("Find Hospitals"):

    if not location:
        st.error("Please enter location")
        st.stop()

    try:
        # ---------------- GEO ----------------
        geo_url = "https://nominatim.openstreetmap.org/search"
        headers = {"User-Agent": "SwasthyaSathi"}

        geo = requests.get(
            geo_url,
            params={"q": location, "format": "json"},
            headers=headers,
            timeout=20
        )

        geo_data = geo.json()

        if not geo_data:
            st.error("Location not found")
            st.stop()

        lat = float(geo_data[0]["lat"])
        lon = float(geo_data[0]["lon"])

        # ---------------- OVERPASS ----------------
        overpass_query = f"""
        [out:json];
        node["amenity"="hospital"](around:{radius},{lat},{lon});
        out;
        """

        response = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": overpass_query},
            timeout=30
        )

        hospital_list = []

        if response.status_code == 200:
            try:
                data = response.json()
                hospitals = data.get("elements", [])

                for h in hospitals:
                    tags = h.get("tags", {})
                    name = tags.get("name", "Hospital")

                    h_lat = h.get("lat")
                    h_lon = h.get("lon")

                    if h_lat and h_lon:
                        distance = calculate_distance(lat, lon, h_lat, h_lon)

                        hospital_list.append({
                            "Hospital": name,
                            "Distance (km)": round(distance, 2),
                            "Latitude": h_lat,
                            "Longitude": h_lon
                        })

            except:
                pass

        # ---------------- FALLBACK ----------------
        if not hospital_list:
            fallback = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": "hospital " + location,
                    "format": "json",
                    "limit": 10
                },
                headers=headers,
                timeout=20
            )

            for h in fallback.json():
                h_lat = float(h["lat"])
                h_lon = float(h["lon"])

                distance = calculate_distance(lat, lon, h_lat, h_lon)

                hospital_list.append({
                    "Hospital": h.get("display_name", "Hospital"),
                    "Distance (km)": round(distance, 2),
                    "Latitude": h_lat,
                    "Longitude": h_lon
                })

        # ---------------- OUTPUT ----------------
        if not hospital_list:
            st.error("No hospitals found")
            st.stop()

        hospital_list.sort(key=lambda x: x["Distance (km)"])

        st.success(f"{len(hospital_list)} hospitals found")

        df = pd.DataFrame(hospital_list)

        st.dataframe(df[["Hospital", "Distance (km)"]], use_container_width=True)

        st.subheader("🚗 Directions")

        for h in hospital_list[:5]:
            link = f"https://www.google.com/maps/dir/?api=1&destination={h['Latitude']},{h['Longitude']}"

            st.markdown(f"""
            🏥 **{h['Hospital']}**  
            📍 {h['Distance (km)']} km  
            👉 [Open in Google Maps]({link})
            ---
            """)

    except Exception as e:
        st.error("Error fetching hospitals")
        st.write(str(e))

# ----------------------------
# DISCLAIMER
# ----------------------------
st.divider()
st.warning("⚠ Educational use only. Consult a doctor for medical advice.")
