# swasthyasathi-ai-health-assistant
AI-powered health assistant that predicts possible diseases from user symptoms and helps users find nearby hospitals

# 🩺 SwasthyaSathi – AI Powered Health Assistant

SwasthyaSathi is a Machine Learning based healthcare assistant that predicts possible diseases based on user-selected symptoms and helps users locate nearby hospitals.

The system is designed to provide **early health awareness and guidance**, especially for people in rural or remote areas where immediate medical consultation may not be available.

---

## 🚀 Features

- Symptom-based disease prediction using Machine Learning
- Displays **Top 3 possible diseases**
- Recommends the **appropriate medical specialist**
- Finds **nearby hospitals based on user location**
- Provides **Google Maps navigation links**
- Simple and user-friendly interface built with Streamlit

---

## 🧠 How It Works

1. The user selects symptoms from the interface.
2. Symptoms are converted into a binary feature vector.
3. A trained **SGDClassifier machine learning model** predicts possible diseases.
4. The system displays the **Top 3 predicted diseases**.
5. Each disease is mapped to the **appropriate medical specialist**.
6. Using **OpenStreetMap APIs**, nearby hospitals are located within a selected radius.

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- SciPy
- OpenStreetMap API
- Overpass API
- Machine Learning (SGDClassifier)

---

## 📊 Dataset

The model was trained on a dataset containing:

- **246,945 rows**
- **378 columns**
- **377 symptoms**
- **1 disease label**

The dataset represents symptom-disease relationships used to train the prediction model.

---

## 🏥 Hospital Finder

The system integrates **Overpass API** to locate nearby hospitals based on the user’s entered location and selected search radius.

Users can directly open navigation using **Google Maps directions**.

---

## 📂 Project Structure
swasthyasathi-ai-health-assistant
│
├── app.py
├── disease_model.pkl
├── label_encoder.pkl
├── symptoms_list.pkl
├── requirements.txt
└── README.md

## ⚙️ Installation

Clone the repository:


git clone https://github.com/Gayatri757/swasthyasathi-ai-health-assistant.git

Install dependencies:


pip install -r requirements.txt


Run the application:


streamlit run app.py


---

## 🌍 Future Scope

- Integration with real healthcare databases
- Doctor appointment booking system
- Real-time telemedicine consultation
- Mobile application version
- Integration with wearable health devices

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.  
It is not intended to replace professional medical advice, diagnosis, or treatment.

Always consult a qualified healthcare professional for medical concerns.

---

## 👩‍💻 Author

**Gayatri Adatiya**

AI & Data Science Student

