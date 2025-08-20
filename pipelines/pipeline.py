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

    # ---- Log metrics ----
    if metrics is not None:
        metrics.log_metric("r2_score", score)

    return score  # return the score explicitly

@dsl.component(base_image=IMAGE_URI)
def evaluate_op(eval_score: float, threshold: float = 0.75) -> bool:
    passed = eval_score >= threshold
    print(f"📊 Model score = {eval_score}, threshold = {threshold}, passed = {passed}")
    return passed

@dsl.component(base_image=IMAGE_URI)
def deploy_op(model_path: str, commit_id: str = "unknown"):
    from google.cloud import aiplatform

    aiplatform.init(project="modified-wonder-468716-e8", location="us-central1")
    model = aiplatform.Model.upload(
        display_name=f"taxi-model-{commit_id}",
        artifact_uri=model_path,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
    )
    endpoint = model.deploy(
        deployed_model_display_name=f"taxi-endpoint-{commit_id}",
        machine_type="n1-standard-2"
    )
    print(f"🚀 Model deployed at endpoint {endpoint.resource_name}")


@dsl.pipeline(name="conditional-deploy-pipeline")
def pipeline(data_path: str = "gs://taxi_model028/data/processed2/n_20_trips-00003-of-00030.jsonl",
             commit_id: str = "unknown",
             threshold: float = 0.75):

    # Train the model
    train_task = train_op(data_path=data_path, commit_id=commit_id)

    # Evaluate score and conditionally deploy
    eval_task = evaluate_op(eval_score=train_task.output, threshold=threshold)
    with dsl.If(eval_task.output):
        deploy_op(
            model_path=train_task.outputs["model_path"],
            commit_id=commit_id
        )
