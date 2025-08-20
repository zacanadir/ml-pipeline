from kfp import dsl
import os

IMAGE_URI = os.environ.get(
    "PIPELINE_IMAGE_URI",
    "us-central1-docker.pkg.dev/modified-wonder-468716-e8/myrepo/ml-pipeline:latest"
)

@dsl.component(base_image=IMAGE_URI)
def train_op(model_path: dsl.OutputPath(str), commit_id: str = "unknown"):
    import numpy as np
    import joblib
    from sklearn.linear_model import LinearRegression
    import os

    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])
    model = LinearRegression().fit(X, y)

    # Ensure folder exists
    os.makedirs(model_path, exist_ok=True)

    # Full file path
    out_file = os.path.join(model_path, f"model_{commit_id}.joblib")
    joblib.dump(model, out_file)

    print(f"✅ Model saved to {out_file}")

@dsl.pipeline(name="hello-pipeline")
def pipeline(commit_id: str = "unknown"):
    train_op(commit_id=commit_id)
