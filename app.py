import streamlit as st
import pickle
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main { background-color: #f9f9fb; }
    .stButton>button {
        background-color: #4F46E5;
        color: white;
        width: 100%;
        border-radius: 8px;
        border: none;
        padding: 10px;
        font-weight: bold;
    }
    .stButton>button:hover { background-color: #4338CA; color: white; }
    .prediction-box {
        padding: 20px;
        background-color: #EEF2F6;
        border-left: 5px solid #4F46E5;
        border-radius: 4px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load the model securely
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model.pkl: {e}")
    st.stop()

# Title and App Description
st.title("🎓 Student Performance Predictor")
st.markdown("Estimate a student's final score using behavioral and attendance metrics.")
st.divider()  # Fixed: Changed from st.hr() to st.divider()

# Layout splits: Sidebar for inputs, Main panel for results
st.sidebar.header("🎯 Input Features")

hours_studied = st.sidebar.slider(
    "Hours Studied (per week)", 
    min_value=0.0, max_value=50.0, value=10.0, step=0.5
)

sleep_hours = st.sidebar.slider(
    "Sleep Hours (per night)", 
    min_value=0.0, max_value=12.0, value=7.0, step=0.5
)

attendance_percent = st.sidebar.slider(
    "Attendance Percentage", 
    min_value=0.0, max_value=100.0, value=85.0, step=1.0
)

previous_scores = st.sidebar.slider(
    "Previous Exam Score", 
    min_value=0.0, max_value=100.0, value=75.0, step=1.0
)

# Main Dashboard View
st.subheader("📊 Session Summary")
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Study Routine", value=f"{hours_studied} hrs/wk")
    st.metric(label="Rest Schedule", value=f"{sleep_hours} hrs/night")
with col2:
    st.metric(label="Class Presence", value=f"{attendance_percent}%")
    st.metric(label="Baseline Score", value=f"{previous_scores}/100")

# Prediction Pipeline
if st.button("Predict Target Score"):
    features = np.array([[hours_studied, sleep_hours, attendance_percent, previous_scores]])
    prediction = model.predict(features)[0]
    
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    st.markdown(f"### 📈 Predicted Output Score: **{prediction:.2f}**")
    st.markdown('</div>', unsafe_allow_html=True)
