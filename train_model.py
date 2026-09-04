from pathlib import Path
import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "Ecommerce Customers.csv"
MODEL_PATH = PROJECT_DIR / "customer_spend_model.pkl"

FEATURES = [
    "Avg. Session Length",
    "Time on App",
    "Time on Website",
    "Length of Membership",
]
TARGET = "Yearly Amount Spent"


def train_and_save_model() -> None:
    data = pd.read_csv(DATA_PATH)
    required_columns = FEATURES + [TARGET]
    missing_columns = sorted(set(required_columns) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    training_data = data[required_columns].dropna()
    if training_data.empty:
        raise ValueError("The dataset has no complete rows for training.")

    model = LinearRegression()
    model.fit(training_data[FEATURES], training_data[TARGET])

    artifact = {
        "model": model,
        "features": FEATURES,
        "target": TARGET,
    }
    with MODEL_PATH.open("wb") as file:
        pickle.dump(artifact, file)

    print(f"Saved {MODEL_PATH.name} using {len(training_data)} rows.")


if __name__ == "__main__":
    train_and_save_model()