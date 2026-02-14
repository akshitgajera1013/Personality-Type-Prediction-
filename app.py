import streamlit as st
import numpy as np
import pickle

# ---- Load Model, Scaler, Label Encoder ----
with open("personality_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scalar.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

st.set_page_config(page_title="Personality Predictor", layout="wide")

st.title("🧠 Personality Type Prediction App")
st.write("Adjust the sliders and predict personality type")

# ---- Input Section ----
def user_input():

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

    return features


input_data = user_input()

# ---- Prediction ----
if st.button("🔍 Predict Personality"):

    scaled_input = scaler.transform(input_data)
    prediction_numeric = model.predict(scaled_input)

    # Convert numeric back to original label
    prediction_text = label_encoder.inverse_transform(prediction_numeric)

    st.success(f"Predicted Personality Type: {prediction_text[0]}")
