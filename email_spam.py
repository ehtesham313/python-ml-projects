import os
import joblib
import pandas as pd
import streamlit as st

# -----------------------------------
# Load Model & Vectorizer
# -----------------------------------
def load_spam_assets():
    BASE_DIR = os.path.dirname(__file__)
    
    paths_to_check = [
        (os.path.join(BASE_DIR, "spam_model.pkl"), os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")),
        (os.path.join(BASE_DIR, "email_spam_detection", "spam_model.pkl"), os.path.join(BASE_DIR, "email_spam_detection", "tfidf_vectorizer.pkl"))
    ]
    
    for m_path, v_path in paths_to_check:
        if os.path.exists(m_path) and os.path.exists(v_path):
            model = joblib.load(m_path)
            vectorizer = joblib.load(v_path)
            return model, vectorizer
            
    raise FileNotFoundError("Spam model or vectorizer files missing.")

# -----------------------------------
# Predict Function
# -----------------------------------
def predict_email(text, model, vectorizer):
    X_vec = vectorizer.transform([text])
    prediction = model.predict(X_vec)[0]
    probabilities = model.predict_proba(X_vec)[0]
    classes = list(model.classes_)
    
    spam_idx = classes.index("spam") if "spam" in classes else 1
    ham_idx = classes.index("ham") if "ham" in classes else 0
    
    spam_prob = probabilities[spam_idx] * 100
    ham_prob = probabilities[ham_idx] * 100
    
    return prediction, spam_prob, ham_prob

# -----------------------------------
# CLI Mode (python email_spam.py)
# -----------------------------------
def run_cli():
    print("=" * 55)
    print(" EMAIL SPAM DETECTION - CONSOLE MODE")
    print("=" * 55)
    try:
        model, vectorizer = load_spam_assets()
    except Exception as e:
        print(f"X Error loading model artifacts: {e}")
        return

    sample_email = "WINNER!! You have won a $1000 cash prize! Claim your reward now!"
    prediction, spam_prob, ham_prob = predict_email(sample_email, model, vectorizer)

    print(f"\n  Sample Email Text:")
    print(f'  "{sample_email}"')
    print("\n" + "=" * 55)
    print(f" CLASSIFICATION RESULT : {prediction.upper()}")
    print(f" Spam Probability     : {spam_prob:.2f}%")
    print(f" Legitimate Probability: {ham_prob:.2f}%")
    print("=" * 55 + "\n")

# -----------------------------------
# Streamlit Web App Mode
# -----------------------------------
def run_streamlit():
    st.set_page_config(
        page_title="Email Spam Detection",
        page_icon="📧",
        layout="centered"
    )

    st.title("📧 Email Spam Detector")
    st.write("Enter an email subject or message body to classify it as **Spam** or **Legitimate (Ham)** using NLP Machine Learning.")
    st.divider()

    try:
        model, vectorizer = load_spam_assets()
    except Exception as e:
        st.error(f"❌ {e}")
        st.stop()

    st.subheader("📝 Input Email Message")

    sample_choice = st.selectbox(
        "Load a sample email (optional):",
        [
            "Custom Input",
            "Sample Spam 1: Cash Prize Winner",
            "Sample Spam 2: Account Locked Verification",
            "Sample Ham 1: Project Sync Reminder",
            "Sample Ham 2: Meeting Presentation Slides"
        ]
    )

    default_text = ""
    if sample_choice == "Sample Spam 1: Cash Prize Winner":
        default_text = "WINNER!! As a valued customer you have been selected for a $1000 cash prize! Text CLAIM to 87121."
    elif sample_choice == "Sample Spam 2: Account Locked Verification":
        default_text = "URGENT: Your bank account has been locked. Verify your credentials immediately at http://fakebank-login.com"
    elif sample_choice == "Sample Ham 1: Project Sync Reminder":
        default_text = "Hi team, just a quick reminder about our project sync scheduled for 3 PM today."
    elif sample_choice == "Sample Ham 2: Meeting Presentation Slides":
        default_text = "Can you send me the updated slides for the presentation on Monday? Thanks!"

    email_text = st.text_area(
        "Email Content:",
        value=default_text,
        height=140,
        placeholder="Paste email text here..."
    )

    if st.button("Analyze Email 🔍", type="primary"):
        if not email_text.strip():
            st.warning("⚠️ Please enter or select an email message to analyze.")
        else:
            prediction, spam_prob, ham_prob = predict_email(email_text, model, vectorizer)

            st.divider()
            if prediction == "spam":
                st.error(f"🚨 **CLASSIFICATION RESULT: SPAM DETECTED**")
                st.metric(label="Spam Risk Level", value=f"{spam_prob:.1f}%", delta="HIGH RISK", delta_color="inverse")
            else:
                st.success(f"✅ **CLASSIFICATION RESULT: LEGITIMATE EMAIL (HAM)**")
                st.metric(label="Legitimate Confidence", value=f"{ham_prob:.1f}%", delta="SAFE", delta_color="normal")

            st.write("---")
            st.write("**Probability Breakdown:**")
            col1, col2 = st.columns(2)
            col1.progress(spam_prob / 100, text=f"Spam: {spam_prob:.1f}%")
            col2.progress(ham_prob / 100, text=f"Legitimate: {ham_prob:.1f}%")

if __name__ == "__main__":
    if st.runtime.exists():
        run_streamlit()
    else:
        run_cli()
