"""
Streamlit application for the Message Intelligence System.

Run:
    streamlit run app.py
"""
import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "message_intelligence_pipeline.pkl"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

st.set_page_config(
    page_title="Message Intelligence System",
    page_icon="📩",
    layout="centered",
)


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r") as file:
        return json.load(file)


if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.error("Model files not found. Keep the `model/` folder next to app.py.")
    st.stop()

model = load_model()
metrics = load_metrics()
labels = metrics["labels"]

st.title("📩 Message Intelligence System")
st.caption("Classify a message as Spam or Legitimate using the trained model.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        message_length = st.number_input("Message Length (characters)", min_value=1, max_value=500, value=75, step=1)
        word_count = st.number_input("Word Count", min_value=1, max_value=100, value=11, step=1)
        num_urls = st.number_input("Number of URLs", min_value=0, max_value=10, value=0, step=1)
        num_digits = st.number_input("Number of Digits", min_value=0, max_value=100, value=0, step=1)
        num_special_chars = st.number_input("Number of Special Characters", min_value=0, max_value=50, value=0, step=1)
        spam_keyword_score = st.number_input("Spam Keyword Score", min_value=0, max_value=10, value=0, step=1)

    with col2:
        legit_keyword_score = st.number_input("Legitimate Keyword Score", min_value=0, max_value=10, value=1, step=1)
        sender_activity_score = st.number_input("Sender Activity Score (0-100)", min_value=0.0, max_value=100.0, value=55.0, step=1.0)
        sender_account_age_days = st.number_input("Sender Account Age (days)", min_value=0, max_value=5000, value=275, step=1)
        messages_sent_last_24h = st.number_input("Messages Sent in Last 24 Hours", min_value=0, max_value=500, value=7, step=1)
        hour_of_day = st.number_input("Hour of Day (0-23)", min_value=0, max_value=23, value=12, step=1)
        day_name = st.selectbox("Day of Week", DAYS, index=3)

    submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

if submitted:
    values = {
        "message_length": message_length,
        "word_count": word_count,
        "num_urls": num_urls,
        "num_digits": num_digits,
        "num_special_chars": num_special_chars,
        "spam_keyword_score": spam_keyword_score,
        "legit_keyword_score": legit_keyword_score,
        "sender_activity_score": sender_activity_score,
        "sender_account_age_days": sender_account_age_days,
        "messages_sent_last_24h": messages_sent_last_24h,
        "hour_of_day": hour_of_day,
        "day_of_week": DAYS.index(day_name),
    }
    input_df = pd.DataFrame([values])[metrics["features"]]

    prediction = int(model.predict(input_df)[0])
    spam_probability = float(model.predict_proba(input_df)[0][1])

    st.subheader("📋 Your Input")
    shown = {
        "Message Length": message_length,
        "Word Count": word_count,
        "Number of URLs": num_urls,
        "Number of Digits": num_digits,
        "Number of Special Characters": num_special_chars,
        "Spam Keyword Score": spam_keyword_score,
        "Legitimate Keyword Score": legit_keyword_score,
        "Sender Activity Score": sender_activity_score,
        "Sender Account Age (days)": sender_account_age_days,
        "Messages Sent in Last 24 Hours": messages_sent_last_24h,
        "Hour of Day": hour_of_day,
        "Day of Week": day_name,
    }
    st.table(pd.DataFrame({"Feature": list(shown.keys()), "Value": [str(v) for v in shown.values()]}))

    st.subheader("🎯 Prediction")
    if prediction == 1:
        st.error(f"🚨 {labels['1']} Message")
    else:
        st.success(f"✅ {labels['0']} Message")
    st.metric("Spam Probability", f"{spam_probability:.1%}")
