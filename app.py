import streamlit as st
import numpy as np
import joblib

# Load your saved model and scaler
model = joblib.load("diabetes_logreg.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Diabetes Prediction App (Logistic Regression)")

st.write("""
Enter patient data below to predict diabetes:
""")

# List the features
features = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

user_data = []
for feature in features:
    val = st.number_input(feature, value=0.0)
    user_data.append(val)

if st.button("Predict"):
    # Prepare data
    data_array = np.array(user_data).reshape(1, -1)
    # Scale input!
    data_scaled = scaler.transform(data_array)
    # Predict
    prediction = model.predict(data_scaled)[0]
    if prediction == 1:
        st.error("⚠️ The person is likely diabetic.")
    else:
        st.success("✅ The person is NOT diabetic.")
