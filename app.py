import streamlit as st
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Personality Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# DARK FUTURISTIC THEME (KEPT SAME)
# =========================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f172a,#1e293b);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
}
.sub-header {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 2rem;
}
.metric-card {
    background: rgba(255,255,255,0.05);
    padding: 1.2rem;
    border-radius: 12px;
    text-align:center;
}
.prediction-box {
    background: linear-gradient(135deg,#2563eb,#3b82f6);
    padding: 2rem;
    border-radius: 18px;
    text-align:center;
    font-size:1.4rem;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
model = pickle.load(open("personality_model.pkl", "rb"))
scaler = pickle.load(open("scalar.pkl", "rb"))
label_encoder = pickle.load(open("encoder.pkl", "rb"))

trait_names = [
    "Social Energy","Alone Time Preference","Talkativeness",
    "Deep Reflection","Group Comfort","Party Liking",
    "Listening Skill","Empathy","Organization",
    "Leadership","Risk Taking","Public Speaking Comfort",
    "Curiosity","Routine Preference","Excitement Seeking",
    "Friendliness","Planning","Spontaneity",
    "Adventurousness","Reading Habit","Sports Interest",
    "Online Social Usage","Travel Desire","Gadget Usage",
    "Collaborative Work Style","Decision Speed"
]

# =========================================================
# SIDEBAR (MOVIE STYLE STRUCTURE)
# =========================================================
with st.sidebar:
    st.markdown("## 🧠 Project Overview")
    st.info("""
    AI-based Personality Classification System  
    Algorithm: Logistic Regression  
    Multi-Class Classification
    """)

    st.markdown("---")
    st.markdown("## 📊 Model Performance")
    st.metric("Accuracy", "99%")
    st.metric("Precision", "0.99")
    st.metric("Recall", "0.99")
    st.metric("F1 Score", "0.99")

    st.markdown("---")
    st.markdown("## 📚 Dataset Info")
    st.write("• Features: 26 Personality Traits")
    st.write("• Output: Multi-Class Personality Type")
    st.write("• Scaling: StandardScaler")

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="main-header">🧠 Personality Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced AI Personality Classification Dashboard</div>', unsafe_allow_html=True)

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Prediction",
    "Trait Analysis",
    "Model Insights",
    "Personality Report"
])

# =========================================================
# TAB 1 - PREDICTION
# =========================================================
with tab1:

    st.markdown("### Enter Personality Traits")

    cols = st.columns(3)
    traits = {}

    for i, trait in enumerate(trait_names):
        with cols[i % 3]:
            traits[trait] = st.slider(trait, 0, 10, 5)

    features = np.array([list(traits.values())])
    scaled = scaler.transform(features)

    if st.button("Run AI Prediction", use_container_width=True):

        prediction = model.predict(scaled)
        pred_text = label_encoder.inverse_transform(prediction)[0]
        probs = model.predict_proba(scaled)[0]
        confidence = round(np.max(probs)*100,2)

        st.session_state["traits"] = traits
        st.session_state["prediction"] = pred_text
        st.session_state["probabilities"] = probs

        st.markdown(f"""
        <div class="prediction-box">
        🎯 Predicted Personality: <b>{pred_text}</b><br><br>
        Confidence Score: {confidence}%
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 2 - TRAIT ANALYSIS
# =========================================================
with tab2:

    if "prediction" in st.session_state:

        st.markdown("### Probability Distribution")

        probs = st.session_state["probabilities"]
        labels = label_encoder.classes_

        fig = px.bar(
            x=labels,
            y=probs*100,
            labels={"x":"Personality Type","y":"Probability (%)"},
            color=labels
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Top Influential Traits")

        class_index = list(label_encoder.classes_).index(
            st.session_state["prediction"]
        )

        coefs = model.coef_[class_index]

        sorted_coef = sorted(
            zip(trait_names, np.abs(coefs)),
            key=lambda x: x[1],
            reverse=True
        )[:8]

        labels_coef = [x[0] for x in sorted_coef]
        values_coef = [x[1] for x in sorted_coef]

        fig2 = px.bar(
            x=values_coef,
            y=labels_coef,
            orientation='h'
        )
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("Run prediction first.")

# =========================================================
# TAB 3 - MODEL INSIGHTS
# =========================================================
with tab3:

    st.markdown("### Model Explanation")

    st.success("""
    Logistic Regression was selected because:

    • High interpretability  
    • Stable performance  
    • Strong multi-class generalization  
    • Probability-based outputs  
    """)

# =========================================================
# TAB 4 - REPORT
# =========================================================
with tab4:

    if "prediction" in st.session_state:

        st.markdown("### Personality Report")

        st.write("Predicted Type:", st.session_state["prediction"])

        sorted_traits = sorted(
            st.session_state["traits"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        st.write("Top Personality Traits:")
        for trait, score in sorted_traits:
            st.write(f"• {trait}: {score}")

    else:
        st.info("Run prediction to generate report.")
