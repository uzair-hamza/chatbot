"""Streamlit chat demo for the UET Peshawar FAQ bot.

Run: streamlit run app.py
Note: streamlit was NOT installed in the training venv at build time,
so install it first: pip install streamlit
"""
import os
import sys

import joblib
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import predict  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model", "chatbot_model.joblib")

st.title("UET Peshawar FAQ Bot")
st.write("Ask about admissions, courses, exams, fees, hostel, library, "
         "scholarships, NCBC, or NAVTTC courses.")

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Train it first: python src/chatbot.py")
    st.stop()

if "bundle" not in st.session_state:
    st.session_state.bundle = joblib.load(MODEL_PATH)
if "history" not in st.session_state:
    st.session_state.history = []

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.write(text)

user = st.chat_input("Ask a question...")
if user:
    intent, conf, reply = predict(st.session_state.bundle, user)
    st.session_state.history.append(("user", user))
    st.session_state.history.append(
        ("assistant", f"{reply}\n\n_(intent: {intent}, confidence: {conf:.2f})_"))
    st.rerun()
