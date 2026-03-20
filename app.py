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
transition:0.3s;
}

.card:hover{
transform:scale(1.03);
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
# LOAD MODEL
# ----------------------------

clf = joblib.load("disease_model.pkl")
le = joblib.load("label_encoder.pkl")
symptoms_list = joblib.load("symptoms_list.pkl")

num_features = len(symptoms_list)

# ----------------------------
# SPECIALIST DETECTION
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

    elif any(x in d for x in ["pregnancy","vaginal","uterus","menstrual"]):
        return "Gynecologist"

    elif any(x in d for x in ["diabetes","thyroid","hormone"]):
        return "Endocrinologist"

    else:
        return "General Physician"

# ----------------------------
# DISTANCE FUNCTION
# ----------------------------

def calculate_distance(lat1, lon1, lat2, lon2):

    lon1, lat1, lon2, lat2 = map(radians,[lon1,lat1,lon2,lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    r = 6371
    return c*r

# ----------------------------
# HEADER
# ----------------------------

st.markdown("""
<div class="main-title">
🩺 SwasthyaSathi
</div>

<div class="subtitle">
Your AI Powered Health Assistant<br>
Predict diseases from symptoms and find nearby hospitals instantly
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

        features = [0]*num_features

        for s in selected_symptoms:
            idx = symptoms_list.index(s)
            features[idx] = 1

        input_vector = csr_matrix([features])

        probs = clf.predict_proba(input_vector)[0]

        top3 = probs.argsort()[-3:][::-1]

        st.subheader("🧠 Possible Diseases")

        col1,col2,col3 = st.columns(3)
        cols = [col1,col2,col3]

        for i,idx in enumerate(top3):

            disease = le.inverse_transform([idx])[0]
            doctor = get_specialist(disease)

            with cols[i]:

                st.markdown(f"""
                <div class="card">
                <h3>🦠 {disease}</h3>
                <p style="color:#666">Recommended Specialist</p>
                <h4 style="color:#2C7BE5">Consult: {doctor}</h4>
                </div>
                """, unsafe_allow_html=True)

# ----------------------------
# HOSPITAL FINDER
# ----------------------------

st.divider()

st.subheader("🏥 Nearby Hospitals")

col1,col2 = st.columns(2)

with col1:
    location = st.text_input("Enter City","Pune")

with col2:
    radius = st.slider("Search Radius (meters)",1000,10000,3000)

# ----------------------------
# SEARCH HOSPITALS
# ----------------------------

if st.button("Find Hospitals"):

    try:

        geo_url="https://nominatim.openstreetmap.org/search"

        params={"q":location,"format":"json"}

        headers={"User-Agent":"SwasthyaSathi"}

        geo=requests.get(geo_url,params=params,headers=headers).json()

        if not geo:
            st.error("Location not found")
            st.stop()

        lat=float(geo[0]["lat"])
        lon=float(geo[0]["lon"])

        overpass_query=f"""
        [out:json];
        node["amenity"="hospital"](around:{radius},{lat},{lon});
        out;
        """

        response=requests.get(
        "https://overpass-api.de/api/interpreter",
        params={"data":overpass_query}
        )

        data=response.json()
        hospitals=data.get("elements",[])

        hospital_list=[]

        for h in hospitals:

            tags=h.get("tags",{})
            name=tags.get("name","Hospital")

            h_lat=h["lat"]
            h_lon=h["lon"]

            distance=calculate_distance(lat,lon,h_lat,h_lon)

            hospital_list.append({
            "Hospital":name,
            "Distance (km)":round(distance,2),
            "Latitude":h_lat,
            "Longitude":h_lon
            })

        hospital_list=sorted(
        hospital_list,
        key=lambda x:x["Distance (km)"]
        )

        st.success(f"{len(hospital_list)} hospitals found")

        df=pd.DataFrame(hospital_list)

        st.dataframe(df[["Hospital","Distance (km)"]],width="stretch")

        st.subheader("🚗 Directions")

        for h in hospital_list[:5]:

            directions=f"https://www.google.com/maps/dir/?api=1&destination={h['Latitude']},{h['Longitude']}"

            st.markdown(f"""
            <div class="hospital-card">
            🏥 <b>{h['Hospital']}</b><br>
            📍 Distance: {h['Distance (km)']} km<br>
            <a href="{directions}" target="_blank">🧭 Open in Google Maps</a>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:

        st.error("Error fetching hospitals")
        st.write(e)

# ----------------------------
# DISCLAIMER
# ----------------------------

st.divider()

st.warning(
"⚠ This AI system is for educational purposes only. Always consult a doctor."
)