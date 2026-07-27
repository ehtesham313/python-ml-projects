import os
import joblib
import pandas as pd
import streamlit as st

# -----------------------------------
# Core Prediction Logic
# -----------------------------------
def get_model():
    BASE_DIR = os.path.dirname(__file__)
    model_path = os.path.join(BASE_DIR, "cricket_score_model.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(BASE_DIR, "..", "cricket_score_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("cricket_score_model.pkl not found.")
    return joblib.load(model_path)

def predict(overs_played, wickets_lost, run_rate, opponent_strength, venue, pitch, weather):
    model = get_model()
    raw_input = pd.DataFrame([{
        "Overs Played": overs_played,
        "Wickets Lost": wickets_lost,
        "Run Rate": run_rate,
        "Opponent Strength": opponent_strength,
        "Home/Away": venue,
        "Pitch Condition": pitch,
        "Weather": weather
    }])

    encoded_input = pd.get_dummies(raw_input)
    expected_columns = getattr(model, "feature_names_in_", None)
    
    if expected_columns is not None:
        final_input = encoded_input.reindex(columns=expected_columns, fill_value=0)
    else:
        final_input = encoded_input

    prediction = model.predict(final_input)[0]
    return max(0, int(round(prediction)))

# -----------------------------------
# CLI Mode (python cricket.py)
# -----------------------------------
def run_cli():
    print("=" * 60)
    print(" CRICKET SCORE PREDICTION - EXECUTION RESULT")
    print("=" * 60)
    
    overs_played = 10
    wickets_lost = 2
    run_rate = 6.5
    opponent_strength = 5
    venue = "Home"
    pitch = "Batting"
    weather = "Sunny"

    try:
        score = predict(overs_played, wickets_lost, run_rate, opponent_strength, venue, pitch, weather)
        print("\n  Input Match Parameters:")
        print(f"   * Overs Played     : {overs_played}")
        print(f"   * Wickets Lost     : {wickets_lost}")
        print(f"   * Current Run Rate : {run_rate}")
        print(f"   * Opponent Strength: {opponent_strength}/10")
        print(f"   * Venue            : {venue}")
        print(f"   * Pitch Condition  : {pitch}")
        print(f"   * Weather          : {weather}")
        print("\n" + "=" * 60)
        print(f" PREDICTED FINAL SCORE: {score} RUNS")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"Execution Error: {e}")

# -----------------------------------
# Streamlit Mode (streamlit run cricket.py)
# -----------------------------------
def run_streamlit():
    st.set_page_config(
        page_title="Cricket Score Predictor",
        page_icon="🏏",
        layout="centered"
    )

    st.title("🏏 Cricket Score Prediction App")
    st.write("Predict the estimated final score of a cricket match based on current match conditions.")
    st.divider()

    try:
        model = get_model()
    except FileNotFoundError:
        st.error("❌ Model file `cricket_score_model.pkl` not found.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        overs_played = st.number_input("Overs Played", min_value=1, max_value=50, value=10)
        wickets_lost = st.number_input("Wickets Lost", min_value=0, max_value=10, value=2)
        run_rate = st.number_input("Current Run Rate", min_value=0.0, max_value=20.0, value=6.5, step=0.1)
        opponent_strength = st.slider("Opponent Strength (1-10)", 1, 10, 5)

    with col2:
        venue = st.selectbox("Venue Location", ["Home", "Away"])
        pitch = st.selectbox("Pitch Condition", ["Batting", "Balanced", "Bowling"])
        weather = st.selectbox("Weather Condition", ["Sunny", "Cloudy", "Overcast"])

    st.divider()

    if st.button("Predict Score 🏏", type="primary"):
        try:
            score = predict(overs_played, wickets_lost, run_rate, opponent_strength, venue, pitch, weather)
            st.success(f"🏏 **Predicted Final Score: {score} Runs**")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

if __name__ == "__main__":
    if st.runtime.exists():
        run_streamlit()
    else:
        run_cli()
