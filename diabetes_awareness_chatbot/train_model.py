import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def train_and_save_model():
    print("Training Diabetes Risk Assessment Model...")

    # Generate synthetic diabetes dataset following Pima Indian Diabetes distribution
    np.random.seed(42)
    n_samples = 500

    pregnancies = np.random.randint(0, 12, n_samples)
    glucose = np.random.normal(120, 30, n_samples).clip(60, 200)
    blood_pressure = np.random.normal(70, 12, n_samples).clip(40, 120)
    skin_thickness = np.random.normal(20, 10, n_samples).clip(0, 50)
    insulin = np.random.normal(80, 50, n_samples).clip(0, 300)
    bmi = np.random.normal(32, 6, n_samples).clip(18, 50)
    dpf = np.random.normal(0.47, 0.3, n_samples).clip(0.08, 2.4)
    age = np.random.randint(21, 80, n_samples)

    # Risk score logic for synthetic target labels
    risk_score = (
        0.03 * (glucose - 100) +
        0.04 * (bmi - 25) +
        0.02 * (age - 30) +
        0.01 * (pregnancies * 5) +
        0.5 * dpf
    )
    prob = 1 / (1 + np.exp(-risk_score))
    outcome = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
        "Outcome": outcome
    })

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    # Save trained model and scaler
    joblib.dump(model, "diabetes_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("SUCCESS: Model (diabetes_model.pkl) and Scaler (scaler.pkl) trained and saved successfully!")

if __name__ == "__main__":
    train_and_save_model()
