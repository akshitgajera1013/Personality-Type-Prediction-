import streamlit as st
import numpy as np
import pickle

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Personality Type Prediction",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS (Modern Look)
# --------------------------------------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}
.stButton>button {
    width: 100%;
    height: 3em;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
}
.result-box {
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    font-size: 24px;
    margin-top: 20px;
}
.footer {
    text-align: center;
    margin-top: 40px;
    font-size: 14px;
    color: #cccccc;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model, Scaler, Encoder
# --------------------------------------------------
with open("personality_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scalar.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# --------------------------------------------------
# Header Section
# --------------------------------------------------
st.markdown("<h1 style='text-align:center;'>🧠 Personality Type Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Logistic Regression Model | Accuracy: 99%</p>", unsafe_allow_html=True)
st.markdown("---")

# --------------------------------------------------
# Input Section
# --------------------------------------------------
st.subheader("Adjust Personality Traits")

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
    work_style_collaborative = st.slider("Work Style Collaborative", 0, 10, 5)
    decision_speed = st.slider("Decision Speed", 0, 10, 5)

# --------------------------------------------------
# Prepare Features (Order MUST Match Training)
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
# Prediction Section
# --------------------------------------------------
st.markdown("---")

if st.button("🔍 Predict Personality Type"):

    scaled_input = scaler.transform(features)

    prediction_numeric = model.predict(scaled_input)
    prediction_text = label_encoder.inverse_transform(prediction_numeric)

    probability = model.predict_proba(scaled_input)
    confidence = round(np.max(probability) * 100, 2)

    st.markdown(f"""
    <div class="result-box" style="background-color:#1e90ff;">
        🎉 Predicted Personality Type: <b>{prediction_text[0]}</b><br><br>
        Confidence Score: <b>{confidence}%</b>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("""
<div class="footer">
Built with ❤️ by Akshit Gajera | Machine Learning Portfolio Project
</div>
""", unsafe_allow_html=True)
