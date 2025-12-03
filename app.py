import streamlit as st
import joblib
import os
from collections import Counter
import pandas as pd
import re

# --------------------------
# File paths
# --------------------------
MODEL_PATH = "model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"

# --------------------------
# Load trained components
# --------------------------
@st.cache_resource
def load_components():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        st.error(f"""
        **Error:** Model files not found!  
        Please run `Train.py` first to generate '{MODEL_PATH}' and '{VECTORIZER_PATH}'.
        """)
        return None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_components()

# --------------------------
# Email classification
# --------------------------
def classify_email(email_text, model, vectorizer):
    X_vec = vectorizer.transform([email_text])
    pred = model.predict(X_vec)[0]
    prob = model.predict_proba(X_vec)[0][pred] * 100
    label = "SPAM" if pred == 1 else "HAM"
    return label, prob

# --------------------------
# Session state for history
# --------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------
# Page config
# --------------------------
st.set_page_config(
    page_title="Email Spam Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# Theme-aware styling
# --------------------------
st.markdown("""
<style>
.stApp {background-color: transparent;}
.main-header {color: #1f77b4; text-align: center; font-weight: bold; margin-bottom: 20px;}
.result-box-ham {
    background-color: #e6ffed; 
    border-left: 6px solid #4CAF50; 
    padding: 20px; 
    border-radius: 10px;
    color: #006400;
}
.result-box-spam {
    background-color: #ffe6e6; 
    border-left: 6px solid #F44336; 
    padding: 20px; 
    border-radius: 10px;
    color: #8B0000;
}
.stButton>button {
    width: 100%; 
    height: 60px; 
    font-size: 1.2em; 
    font-weight: bold; 
    background-color: #1f77b4; 
    color: white; 
    border-radius: 12px;
}
.stButton>button:hover {background-color: #165b8c;}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Sidebar Overview + Analytics
# --------------------------
st.sidebar.header("📌 Project Overview")
st.sidebar.markdown("""
Classifies emails as **Spam** or **Ham** using a **pre-trained Multinomial Naive Bayes model**.

**Components Used:**
- Dataset: Kaggle `spam.csv`
- Feature Extraction: TF-IDF
- Algorithm: Naive Bayes
- Deployment: Streamlit

**Steps:**
1. Enter the email text.
2. Click **Classify Email**.
3. See confidence, suggestions, history, and analytics.

**Author:** Reshmitha Kollati
""")
st.sidebar.markdown("---")
st.sidebar.markdown("💡 Works with **light**, **dark**, and **system** themes.")

# --------------------------
# Analytics Dashboard
# --------------------------
st.sidebar.header("📊 Analytics Dashboard")
if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    total_checked = len(history_df)
    spam_count = len(history_df[history_df['Label'] == "SPAM"])
    spam_percentage = (spam_count / total_checked) * 100

    spam_texts = " ".join(history_df[history_df['Label'] == "SPAM"]['Text'].tolist())
    words = re.findall(r'\b\w+\b', spam_texts.lower())
    top_words = Counter(words).most_common(5)

    st.metric("Total Emails Checked", total_checked)
    st.metric("Spam Emails Detected", f"{spam_count} ({spam_percentage:.2f}%)")
    st.markdown("**Top 5 Spam Words:**")
    for word, count in top_words:
        st.write(f"- {word} ({count} times)")
else:
    st.info("No emails classified yet. Start classifying to see analytics here.")

# --------------------------
# Main UI
# --------------------------
st.markdown('<h1 class="main-header">📧 Interactive Email Spam Classifier</h1>', unsafe_allow_html=True)

if model and vectorizer:
    email_text = st.text_area(
        "Enter the email message here:",
        height=250,
        help="Paste the full email text for classification."
    )

    if st.button("🚀 Classify Email"):
        if email_text.strip():
            with st.spinner("Analyzing..."):
                label, confidence = classify_email(email_text, model, vectorizer)

            # Display result with spam threat meter
            box_class = "result-box-spam" if label == "SPAM" else "result-box-ham"
            icon = "🚨 SPAM Detected!" if label == "SPAM" else "✅ HAM (Safe)"
            color = "#F44336" if label == "SPAM" else "#4CAF50"

            st.markdown("---")
            st.markdown(f"""
                <div class="{box_class}">
                    <h2 style="color: {color}; margin-top: 0px;">{icon}</h2>
                    <p style="font-size: 1.5em; font-weight: bold;">Confidence: {confidence:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)

            # Spam threat meter
            st.progress(int(confidence))

            # Real-time suggestions
            if label == "SPAM":
                st.info("💡 Suggested Actions: Mark as Spam | Do not click links | Report sender")
            else:
                st.success("💡 Suggested Actions: Safe to read | No action required")

            # Save to history
            st.session_state.history.append({
                "Text": email_text,
                "Label": label,
                "Confidence": f"{confidence:.2f}%"
            })
        else:
            st.warning("Please enter an email message to classify.")
