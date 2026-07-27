import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def train_and_save_cricket_model():
    print("Training new Cricket Score Prediction Model...")

    np.random.seed(42)
    n_samples = 1000

    overs_played = np.random.randint(1, 51, n_samples)
    wickets_lost = np.random.randint(0, 11, n_samples)
    run_rate = np.random.uniform(3.0, 12.0, n_samples)
    opponent_strength = np.random.randint(1, 11, n_samples)
    home_away = np.random.choice(["Home", "Away"], n_samples)
    pitch_condition = np.random.choice(["Batting", "Balanced", "Bowling"], n_samples)
    weather = np.random.choice(["Sunny", "Cloudy", "Overcast"], n_samples)

    # Calculate expected final score based on match dynamics
    base_runs = overs_played * run_rate
    remaining_overs = 50 - overs_played
    resource_factor = (10 - wickets_lost) / 10.0
    projected_remaining_runs = remaining_overs * run_rate * resource_factor

    pitch_mult = np.where(pitch_condition == "Batting", 1.1, np.where(pitch_condition == "Bowling", 0.9, 1.0))
    weather_mult = np.where(weather == "Sunny", 1.05, np.where(weather == "Overcast", 0.95, 1.0))
    venue_mult = np.where(home_away == "Home", 1.03, 0.97)
    opp_mult = 1.0 - (opponent_strength - 5) * 0.02

    final_score = (base_runs + projected_remaining_runs) * pitch_mult * weather_mult * venue_mult * opp_mult
    final_score = np.clip(final_score + np.random.normal(0, 10, n_samples), 50, 450).astype(int)

    df = pd.DataFrame({
        "Overs Played": overs_played,
        "Wickets Lost": wickets_lost,
        "Run Rate": run_rate,
        "Opponent Strength": opponent_strength,
        "Home/Away": home_away,
        "Pitch Condition": pitch_condition,
        "Weather": weather,
        "Final Score": final_score
    })

    X = df.drop("Final Score", axis=1)
    y = df["Final Score"]

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_encoded, y)

    # Save trained model to current directory
    model_path = os.path.join(os.path.dirname(__file__), "cricket_score_model.pkl")
    joblib.dump(model, model_path)

    # Also save to root workspace for convenience
    root_model_path = os.path.join(os.path.dirname(__file__), "..", "cricket_score_model.pkl")
    joblib.dump(model, root_model_path)

    print("SUCCESS: Trained and saved cricket_score_model.pkl successfully!")

if __name__ == "__main__":
    train_and_save_cricket_model()
