import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# -----------------------------------
# Model Loader
# -----------------------------------
def load_diabetes_assets():
    BASE_DIR = os.path.dirname(__file__)
    
    paths_to_check = [
        (os.path.join(BASE_DIR, "diabetes_model.pkl"), os.path.join(BASE_DIR, "scaler.pkl")),
        (os.path.join(BASE_DIR, "diabetes_awareness_chatbot", "diabetes_model.pkl"), os.path.join(BASE_DIR, "diabetes_awareness_chatbot", "scaler.pkl"))
    ]
    
    for m_path, s_path in paths_to_check:
        if os.path.exists(m_path) and os.path.exists(s_path):
            model = joblib.load(m_path)
            scaler = joblib.load(s_path)
            return model, scaler
            
    raise FileNotFoundError("Diabetes model or scaler files missing.")

# -----------------------------------
# Risk Prediction Function
# -----------------------------------
def predict_diabetes_risk(glucose, bmi, age, pregnancies, blood_pressure, insulin, skin_thickness, dpf, model, scaler):
    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }])

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1] * 100
    return prediction, probability

# -----------------------------------
# Chatbot Knowledge Engine
# -----------------------------------
FAQ_KNOWLEDGE_BASE = {
    "symptoms": (
        "🩺 **Common Diabetes Symptoms Include:**\n"
        "- Frequent urination (polyuria)\n"
        "- Excessive thirst (polydipsia)\n"
        "- Extreme hunger or fatigue\n"
        "- Unexplained weight loss\n"
        "- Blurry vision & slow-healing sores"
    ),
    "prevention": (
        "🛡️ **Diabetes Prevention & Management Tips:**\n"
        "- Maintain a balanced, low-sugar diet rich in fiber and whole grains.\n"
        "- Engage in at least 30 minutes of moderate exercise 5 days a week.\n"
        "- Monitor blood glucose levels regularly.\n"
        "- Stay hydrated and maintain a healthy BMI.\n"
        "- Avoid smoking and limit alcohol consumption."
    ),
    "types": (
        "📘 **Types of Diabetes:**\n"
        "- **Type 1**: Autoimmune condition where the pancreas produces little or no insulin.\n"
        "- **Type 2**: Chronic condition affecting how your body processes blood sugar (most common).\n"
        "- **Gestational**: High blood sugar developing during pregnancy."
    ),
    "diet": (
        "🥗 **Recommended Diabetes Diet:**\n"
        "- **Eat More**: Leafy greens, berries, oats, beans, lean poultry, fish, and nuts.\n"
        "- **Limit**: Sugary drinks, refined carbs (white bread, pasta), processed snacks, and trans fats."
    )
}

def get_chatbot_response(prompt):
    text = prompt.lower()
    if any(k in text for k in ["symptom", "sign", "feel", "warning"]):
        return FAQ_KNOWLEDGE_BASE["symptoms"]
    elif any(k in text for k in ["prevent", "avoid", "manage", "lifestyle"]):
        return FAQ_KNOWLEDGE_BASE["prevention"]
    elif any(k in text for k in ["type", "difference", "type 1", "type 2"]):
        return FAQ_KNOWLEDGE_BASE["types"]
    elif any(k in text for k in ["diet", "food", "eat", "nutrition", "meal"]):
        return FAQ_KNOWLEDGE_BASE["diet"]
    elif any(k in text for k in ["hello", "hi", "hey"]):
        return "Hello! I am your **Diabetes Awareness Assistant**. How can I help you today? You can ask about symptoms, prevention, diet, or types of diabetes."
    else:
        return (
            "I'm here to provide diabetes awareness information! You can ask me about:\n"
            "• Diabetes symptoms and early warning signs\n"
            "• Prevention tips and lifestyle habits\n"
            "• Recommended diets and nutrition guidance\n"
            "• Differences between Type 1 and Type 2 diabetes\n"
            "\n*Note: For medical emergencies or diagnosis, always consult a licensed doctor.*"
        )

# -----------------------------------
# CLI Mode (python diabetes.py)
# -----------------------------------
def run_cli():
    print("=" * 60)
    print(" DIABETES AWARENESS & RISK PREDICTOR - CONSOLE MODE")
    print("=" * 60)
    try:
        model, scaler = load_diabetes_assets()
    except Exception as e:
        print(f"X Error loading model: {e}")
        return

    glucose, bmi, age = 145, 31.5, 48
    prediction, probability = predict_diabetes_risk(
        glucose=glucose, bmi=bmi, age=age, pregnancies=2,
        blood_pressure=75, insulin=100, skin_thickness=25, dpf=0.5,
        model=model, scaler=scaler
    )

    print("\n  Sample Health Metrics:")
    print(f"   * Fasting Glucose : {glucose} mg/dL")
    print(f"   * Body Mass Index : {bmi}")
    print(f"   * Age             : {age} years")
    print("\n" + "=" * 60)
    print(f" RISK EVALUATION : {'HIGH RISK OF DIABETES' if prediction == 1 else 'LOW RISK / NORMAL'}")
    print(f" Risk Probability: {probability:.1f}%")
    print("=" * 60 + "\n")

# -----------------------------------
# Streamlit Web App Mode
# -----------------------------------
def run_streamlit():
    st.set_page_config(
        page_title="Diabetes Awareness Chatbot & Risk Predictor",
        page_icon="🩺",
        layout="centered"
    )

    st.title("🩺 Diabetes Awareness & Health Hub")
    st.write("An interactive AI assistant and Machine Learning risk assessment tool for diabetes awareness and prevention.")
    st.divider()

    try:
        model, scaler = load_diabetes_assets()
    except Exception as e:
        st.error(f"❌ {e}")
        st.stop()

    tab1, tab2 = st.tabs(["💬 AI Awareness Chatbot", "📊 ML Diabetes Risk Predictor"])

    with tab1:
        st.subheader("💬 Ask the Diabetes Assistant")
        st.caption("Ask questions about symptoms, diet, prevention tips, and diabetes management.")

        if "diabetes_messages" not in st.session_state:
            st.session_state.diabetes_messages = [
                {"role": "assistant", "content": "Hello! I am your **Diabetes Awareness Assistant**. How can I help you today?"}
            ]

        for msg in st.session_state.diabetes_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_prompt := st.chat_input("Type your question here (e.g. What are the symptoms of diabetes?)..."):
            st.session_state.diabetes_messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            response = get_chatbot_response(user_prompt)
            st.session_state.diabetes_messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

    with tab2:
        st.subheader("📊 Diabetes Risk Assessment Tool")
        st.write("Enter clinical parameters to estimate diabetes risk probability using Machine Learning.")

        col1, col2 = st.columns(2)

        with col1:
            glucose = st.number_input("Fasting Glucose Level (mg/dL)", min_value=50, max_value=300, value=120)
            bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=28.5, step=0.1)
            age = st.number_input("Age (Years)", min_value=1, max_value=120, value=45)
            pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)

        with col2:
            blood_pressure = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=30, max_value=140, value=75)
            insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=800, value=85)
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=99, value=20)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.05, max_value=3.0, value=0.47, step=0.01)

        st.divider()

        if st.button("Evaluate Diabetes Risk 🩺", type="primary"):
            prediction, probability = predict_diabetes_risk(
                glucose, bmi, age, pregnancies, blood_pressure, insulin, skin_thickness, dpf, model, scaler
            )

            if prediction == 1:
                st.error(f"⚠️ **HIGH RISK DETECTED**: Estimated Diabetes Probability is **{probability:.1f}%**")
                st.info("💡 **Recommendation**: Please consult a medical professional for formal lab tests and evaluation.")
            else:
                st.success(f"✅ **LOW RISK / NORMAL**: Estimated Diabetes Probability is **{prob:.1f}%**" if 'prob' in locals() else f"✅ **LOW RISK / NORMAL**: Estimated Diabetes Probability is **{probability:.1f}%**")
                st.info("💡 **Tip**: Maintain healthy eating habits and regular physical exercise.")

if __name__ == "__main__":
    if st.runtime.exists():
        run_streamlit()
    else:
        run_cli()
