import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def generate_synthetic_data(n=50000):
    barangays = ["Santa Cruz", "Calamba", "Los Baños", "San Pablo", "Biñan", "Pagsanjan"]

    rng = np.random.default_rng(42)

    data = {
        "Age": rng.integers(18, 80, n),
        "Gender": rng.choice(["Male", "Female"], n),
        "Barangay": rng.choice(barangays, n),
        "BMI": np.round(rng.normal(24, 4, n), 1),
        "BloodPressure": rng.choice(["Normal", "Elevated", "Hypertension"], n, p=[0.5, 0.3, 0.2]),
        "ExerciseFreq": rng.choice(["Rarely", "Weekly", "Daily"], n, p=[0.3, 0.5, 0.2]),
        "DietScore": rng.integers(1, 11, n),
    }

    df = pd.DataFrame(data)

    # Stronger correlation for DiabetesRisk
    risk_score = (
        (df["Age"] > 50).astype(int)
        + (df["BMI"] > 27).astype(int)
        + (df["BloodPressure"] == "Hypertension").astype(int)
        + (df["ExerciseFreq"] == "Rarely").astype(int)
        + (df["DietScore"] < 4).astype(int)
    )

    prob = np.clip(0.05 + 0.25 * risk_score, 0, 0.95)
    df["DiabetesRisk"] = np.where(rng.random(n) < prob, "Yes", "No")

    return df

def train_model(df):
    df_enc = df.copy()
    for col in ["Gender", "Barangay", "BloodPressure", "ExerciseFreq", "DiabetesRisk"]:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col])

    X = df_enc.drop("DiabetesRisk", axis=1)
    y = df_enc["DiabetesRisk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)
    return model, report

def main():
    print("Generating synthetic Laguna health dataset...")
    df = generate_synthetic_data(n=5000)
    df.to_csv("laguna_health_data.csv", index=False)
    print("Saved dataset to laguna_health_data.csv")

    print("Training Random Forest model...")
    model, report = train_model(df)

    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("Classification Report:\n")
    print(report)

if __name__ == "__main__":
    main()