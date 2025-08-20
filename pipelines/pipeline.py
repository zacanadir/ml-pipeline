from kfp import dsl
import os

IMAGE_URI = os.environ.get(
    "PIPELINE_IMAGE_URI",
    "us-central1-docker.pkg.dev/modified-wonder-468716-e8/myrepo/ml-pipeline:latest"
)

@dsl.component(base_image=IMAGE_URI)
def train_op(model_path: dsl.OutputPath(str),
             data_path: str,
             commit_id: str = "unknown",
             metrics: dsl.Output[dsl.Metrics] = None):
    import joblib, os
    import trainer.task as my_model

    # ---- Train the model ----
    model, score = my_model.train(data_path=data_path)

    # ---- Ensure folder exists ----
    os.makedirs(model_path, exist_ok=True)

    # ---- Save model with commit ID ----
    out_file = os.path.join(model_path, f"model_{commit_id}.joblib")
    joblib.dump(model, out_file)

    print(f"✅ Model saved to {out_file}, R² = {score:.4f}")

    # ---- Log metrics to pipeline system ----
    if metrics is not None:
        metrics.log_metric("r2_score", score)

@dsl.pipeline(name="hello-pipeline")
def pipeline(data_path: str = "gs://taxi_model028/data/processed2/n_20_trips-00003-of-00030.jsonl",
             commit_id: str = "unknown"):
    train_op(data_path=data_path, commit_id=commit_id)
