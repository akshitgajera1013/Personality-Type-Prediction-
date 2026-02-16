import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

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
    margin-bottom: 25px;
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
    <div class="subtitle">Logistic Regression | Multi-Class Personality Prediction</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --------------------------------------------------
# Input Section
# --------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Adjust Your Personality Traits (0 = Low, 10 = High)")

traits = {}

trait_names = [
    "Social Energy", "Alone Time Preference", "Talkativeness",
    "Deep Reflection", "Group Comfort", "Party Liking",
    "Listening Skill", "Empathy", "Organization",
    "Leadership", "Risk Taking", "Public Speaking Comfort",
    "Curiosity", "Routine Preference", "Excitement Seeking",
    "Friendliness", "Planning", "Spontaneity",
    "Adventurousness", "Reading Habit", "Sports Interest",
    "Online Social Usage", "Travel Desire", "Gadget Usage",
    "Collaborative Work Style", "Decision Speed"
]

cols = st.columns(3)
for i, trait in enumerate(trait_names):
    with cols[i % 3]:
        traits[trait] = st.slider(trait, 0, 10, 5)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Prepare Features
# --------------------------------------------------
features = np.array([list(traits.values())])

# --------------------------------------------------
# Prediction Section
# --------------------------------------------------
st.markdown("---")

if st.button("🔍 Analyze Personality"):

    scaled_input = scaler.transform(features)
    prediction_numeric = model.predict(scaled_input)
    prediction_text = label_encoder.inverse_transform(prediction_numeric)

    probability = model.predict_proba(scaled_input)
    confidence = round(np.max(probability) * 100, 2)

    personality = prediction_text[0]

    personality_description = {
        "Extrovert": "Energetic, social, expressive, and thrives in group environments.",
        "Introvert": "Reflective, independent, thoughtful, and comfortable with solitude.",
        "Ambivert": "Balanced personality combining both introverted and extroverted strengths."
    }

    description = personality_description.get(
        personality,
        "Unique personality pattern detected."
    )

    # ----------------------------------------------
    # Display Result
    # ----------------------------------------------
    st.markdown(f"""
    <div class="card">
        <h2>🎯 Predicted Personality: {personality}</h2>
        <p>{description}</p>
        <br>
        <strong>Confidence Level:</strong>
    </div>
    """, unsafe_allow_html=True)

    st.progress(confidence / 100)

    st.write(f"Confidence Score: {confidence}%")

    # ----------------------------------------------
    # Radar Chart (Top 5 Dominant Traits)
    # ----------------------------------------------
    st.markdown("<br>")
    st.subheader("📊 Personality Trait Profile")

    # Select top 5 strongest traits
    sorted_traits = sorted(traits.items(), key=lambda x: x[1], reverse=True)[:5]

    labels = [t[0] for t in sorted_traits]
    values = [t[1] for t in sorted_traits]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    ax.set_ylim(0, 10)

    st.pyplot(fig)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("""
<div class="footer">
Built with ❤️ by Akshit Gajera | Machine Learning Portfolio
</div>
""", unsafe_allow_html=True)
