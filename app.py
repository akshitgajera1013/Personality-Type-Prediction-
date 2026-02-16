import streamlit as st
import numpy as np
import pickle

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Personality Intelligence System",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------
# Premium CSS
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    font-family: 'Segoe UI', sans-serif;
    color: white;
}

.header {
    text-align: center;
    margin-bottom: 20px;
}

.title {
    font-size: 42px;
    font-weight: 700;
}

.subtitle {
    color: #94a3b8;
    font-size: 18px;
}

.card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 35px rgba(0,0,0,0.4);
}

.stButton>button {
    width: 100%;
    height: 55px;
    font-size: 18px;
    font-weight: bold;
    border-radius: 14px;
    background: linear-gradient(90deg,#22d3ee,#3b82f6);
    color: black;
    border: none;
}

.result-box {
    margin-top: 30px;
    padding: 35px;
    border-radius: 18px;
    text-align: center;
}

.footer {
    text-align: center;
    margin-top: 50px;
    font-size: 14px;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
with open("personality_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scalar.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("""
<div class="header">
    <div class="title">🧠 Personality Intelligence System</div>
    <div class="subtitle">Logistic Regression Model | Multi-Class Classification</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --------------------------------------------------
# Input Section
# --------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Adjust Your Personality Traits (0 = Low, 10 = High)")

col1, col2, col3 = st.columns(3)

with col1:
    social_energy = st.slider("Social Energy", 0, 10, 5)
    alone_time_preference = st.slider("Alone Time Preference", 0, 10, 5)
    talkativeness = st.slider("Talkativeness", 0, 10, 5)
    deep_reflection = st.slider("Deep Reflection", 0, 10, 5)
    group_comfort = st.slider("Group Comfort", 0, 10, 5)
    party_liking = st.slider("Party Liking", 0, 10, 5)
    listening_skill = st.slider("Listening Skill", 0, 10, 5)
    empathy = st.slider("Empathy", 0, 10, 5)
    organization = st.slider("Organization", 0, 10, 5)

with col2:
    leadership = st.slider("Leadership", 0, 10, 5)
    risk_taking = st.slider("Risk Taking", 0, 10, 5)
    public_speaking_comfort = st.slider("Public Speaking Comfort", 0, 10, 5)
    curiosity = st.slider("Curiosity", 0, 10, 5)
    routine_preference = st.slider("Routine Preference", 0, 10, 5)
    excitement_seeking = st.slider("Excitement Seeking", 0, 10, 5)
    friendliness = st.slider("Friendliness", 0, 10, 5)
    planning = st.slider("Planning", 0, 10, 5)
    spontaneity = st.slider("Spontaneity", 0, 10, 5)

with col3:
    adventurousness = st.slider("Adventurousness", 0, 10, 5)
    reading_habit = st.slider("Reading Habit", 0, 10, 5)
    sports_interest = st.slider("Sports Interest", 0, 10, 5)
    online_social_usage = st.slider("Online Social Usage", 0, 10, 5)
    travel_desire = st.slider("Travel Desire", 0, 10, 5)
    gadget_usage = st.slider("Gadget Usage", 0, 10, 5)
    work_style_collaborative = st.slider("Collaborative Work Style", 0, 10, 5)
    decision_speed = st.slider("Decision Speed", 0, 10, 5)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Prepare Features
# --------------------------------------------------
features = np.array([[ 
    social_energy, alone_time_preference, talkativeness,
    deep_reflection, group_comfort, party_liking,
    listening_skill, empathy, organization,
    leadership, risk_taking, public_speaking_comfort,
    curiosity, routine_preference, excitement_seeking,
    friendliness, planning, spontaneity,
    adventurousness, reading_habit, sports_interest,
    online_social_usage, travel_desire, gadget_usage,
    work_style_collaborative, decision_speed
]])

# --------------------------------------------------
# Prediction
# --------------------------------------------------
st.markdown("---")

if st.button("🔍 Analyze Personality"):

    scaled_input = scaler.transform(features)

    prediction_numeric = model.predict(scaled_input)
    prediction_text = label_encoder.inverse_transform(prediction_numeric)

    probability = model.predict_proba(scaled_input)
    confidence = round(np.max(probability) * 100, 2)

    personality = prediction_text[0]

    # Add short description
    personality_description = {
        "Extrovert": "Energetic, social, and thrives in group settings.",
        "Introvert": "Reflective, independent, and comfortable with solitude.",
        "Ambivert": "Balanced personality with both introverted and extroverted traits."
    }

    description = personality_description.get(
        personality,
        "Unique personality pattern detected."
    )

    st.markdown(f"""
    <div class="result-box" style="background:#1f2937;">
        <h2>🎯 Predicted Personality: {personality}</h2>
        <p>{description}</p>
        <br>
        <strong>Confidence Level: {confidence}%</strong>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("""
<div class="footer">
Built with ❤️ by Akshit Gajera | Machine Learning Portfolio
</div>
""", unsafe_allow_html=True)
