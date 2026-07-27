import os
import streamlit as st
import cricket
import diabetes
import email_spam

# -----------------------------------
# Page Config
# -----------------------------------
st.set_page_config(
    page_title="Machine Learning Applications",
    page_icon="🤖",
    layout="wide"
)

# Store selected project in session state to show ONLY one project at a time
if "active_project" not in st.session_state:
    st.session_state.active_project = None

# If no project is selected yet, present a clean launch selection
if st.session_state.active_project is None:
    st.title("🤖 Select Machine Learning Program")
    st.write("Click on any program below to open and run **only that project** on the page.")
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🏏 Cricket Score Predictor")
        st.write("Predict estimated final scores of cricket matches using match statistics.")
        if st.button("Run Cricket Program 🏏", key="btn_cricket", type="primary", use_container_width=True):
            st.session_state.active_project = "cricket"
            st.rerun()

    with col2:
        st.markdown("### 🩺 Diabetes Risk & Chatbot")
        st.write("Interactive AI health assistant and clinical diabetes risk assessment tool.")
        if st.button("Run Diabetes Program 🩺", key="btn_diabetes", type="primary", use_container_width=True):
            st.session_state.active_project = "diabetes"
            st.rerun()

    with col3:
        st.markdown("### 📧 Email Spam Detector")
        st.write("Classify email text as Spam or Legitimate using NLP Machine Learning.")
        if st.button("Run Email Spam Program 📧", key="btn_email", type="primary", use_container_width=True):
            st.session_state.active_project = "email"
            st.rerun()

# -----------------------------------
# RENDER ONLY THE SINGLE SELECTED PROJECT
# -----------------------------------
else:
    # Top bar back button to exit project
    top_col1, top_col2 = st.columns([1, 8])
    with top_col1:
        if st.button("← Switch Program", key="btn_back"):
            st.session_state.active_project = None
            st.rerun()

    st.divider()

    if st.session_state.active_project == "cricket":
        cricket.run_streamlit()
    elif st.session_state.active_project == "diabetes":
        diabetes.run_streamlit()
    elif st.session_state.active_project == "email":
        email_spam.run_streamlit()
