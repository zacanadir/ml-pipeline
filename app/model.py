# trainer/task.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import gcsfs

# ---- Hard-coded paths ----
DATA_PATH = "gs://taxi_model028/data/processed2/n_20_trips-00002-of-00030.jsonl"
MODEL_PATH = "gs://taxi_model028/output/rf_taxi_model22.joblib"

def train():
    # ---- Load data from GCS ----
    print(f"📂 Loading data from {DATA_PATH} ...")
    fs = gcsfs.GCSFileSystem()
    with fs.open(DATA_PATH, 'r') as f:
        df = pd.read_json(f, lines=True)

    # ---- Features and target ----
    feature_cols = [
        'pickup_longitude', 'pickup_latitude',
        'dropoff_latitude', 'dropoff_longitude',
        'trip_seconds', 'trip_miles',
        'day_of_week', 'hour_of_day'
    ]
    target_col = 'trip_total'

    X = df[feature_cols]
    y = df[target_col]

    # ---- Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---- Train ----
    print("🚀 Training RandomForestRegressor ...")
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # ---- Save directly to GCS ----
    print(f"💾 Saving model to {MODEL_PATH} ...")
    with gcsfs.GCSFileSystem().open(MODEL_PATH, 'wb') as f:
        joblib.dump(model, f)
    print("✅ Model saved successfully.")

if __name__ == "__main__":
    train()
