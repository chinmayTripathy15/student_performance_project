import streamlit as st
import pickle
import numpy as np

# Load model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("🎓 Student Performance Predictor")

# Input sliders
study_hours = st.slider("📚 Study Hours per Day", 0, 10, 5)
sleep_hours = st.slider("🛏️ Sleep Hours per Day", 0, 10, 7)
attendance = st.slider("📅 Attendance (%)", 0, 100, 75)

if st.button("Predict Result"):
    input_data = np.array([[study_hours, sleep_hours, attendance]])
    result = model.predict(input_data)
    if result[0] == 1:
        st.success("✅ Student is likely to PASS!")
    else:
        st.error("❌ Student is likely to FAIL.")
