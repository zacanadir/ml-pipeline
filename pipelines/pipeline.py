from kfp import dsl
import os

IMAGE_URI = os.environ.get("PIPELINE_IMAGE_URI", "us-central1-docker.pkg.dev/$PROJECT_ID/myrepo/ml-pipeline:latest")

@dsl.component(base_image=IMAGE_URI)
def train_op():
    import numpy as np
    import joblib
    from sklearn.linear_model import LinearRegression

    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])
    model = LinearRegression().fit(X, y)

    joblib.dump(model, "model.joblib")
    print("✅ Model trained and saved.")

@dsl.pipeline(name="hello-pipeline")
def pipeline():
    train_op()
