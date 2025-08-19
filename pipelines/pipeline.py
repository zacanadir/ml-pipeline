from kfp import dsl
import os

# Use the image you built/pushed (passed from Cloud Build)
IMAGE_URI = os.environ.get(
    "PIPELINE_IMAGE_URI",
    "us-central1-docker.pkg.dev/modified-wonder-468716-e8/myrepo/ml-pipeline:latest"
)

@dsl.component(base_image=IMAGE_URI)
def train_op(model_path: dsl.OutputPath(str)):
    import numpy as np
    import joblib
    from sklearn.linear_model import LinearRegression

    # Dummy training data
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])

    # Train model
    model = LinearRegression().fit(X, y)

    # Save artifact to provided path (Vertex AI injects a GCS location)
    joblib.dump(model, model_path)
    print(f"✅ Model trained and saved at {model_path}")

@dsl.pipeline(name="hello-pipeline")
def pipeline():
    train_op()
