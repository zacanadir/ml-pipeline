# trainer/task.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import gcsfs

def train(data_path: str):
    # ---- Load data from GCS ----
    print(f"Loading data from {data_path} ...")
    fs = gcsfs.GCSFileSystem()
    with fs.open(data_path, "r") as f:
        df = pd.read_json(f, lines=True)

    # ---- Features and target ----
    feature_cols = [
        "pickup_longitude", "pickup_latitude",
        "dropoff_latitude", "dropoff_longitude",
        "trip_seconds", "trip_miles",
        "day_of_week", "hour_of_day"
    ]
    target_col = "trip_total"

    X = df[feature_cols]
    y = df[target_col]

    # ---- Train/test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---- Train model ----
    print("Training RandomForestRegressor ...")
    model = RandomForestRegressor(random_state=42)  # reproducible
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)  # R² score
    return model, score
