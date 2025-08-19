
from kfp import dsl
import os

# Make sure the environment variable is passed correctly
IMAGE_URI = os.environ.get(
    "PIPELINE_IMAGE_URI",
    "us-central1-docker.pkg.dev/modified-wonder-468716-e8/myrepo/ml-pipeline:latest"
)

@dsl.component(base_image=IMAGE_URI)
def train_op():
    import numpy as np
    import joblib
    from sklearn.linear_model import LinearRegression
    import os

    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])
    model = LinearRegression().fit(X, y)

    model_path = os.path.join(os.environ["AIP_PIPELINE_ROOT"], "model.joblib")
    joblib.dump(model, model_path)
    print(f"✅ Model trained and saved at {model_path}")

@dsl.pipeline(name="hello-pipeline")
def pipeline():
    train_op()
