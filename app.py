import streamlit as st
import pickle
import numpy as np

# Load trained ML model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🎓 Exam Score Predictor (ML Regression)")

# User Input Sliders
hours = st.slider("📚 Study Hours", 0, 12, 5)
sleep = st.slider("😴 Sleep Hours", 0, 12, 7)
attendance = st.slider("📅 Attendance (%)", 0, 100, 75)
previous = st.slider("📘 Previous Score (%)", 0, 100, 50)

if st.button("Predict Result"):

    # Prepare input
    input_data = np.array([[hours, sleep, attendance, previous]])

    # Predict score
    predicted_score = model.predict(input_data)[0]

    # Display predicted score
    st.info(f"📊 Predicted Exam Score: {predicted_score:.2f}")

    # PASS / FAIL based on the score
    if predicted_score >= 40:
        st.success("✅ Result: PASS")
    else:
        st.error("❌ Result: FAIL")
