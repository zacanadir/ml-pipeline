from kfp import dsl
import os

# Use the image you built/pushed
IMAGE_URI = os.environ.get(
    "PIPELINE_IMAGE_URI",
    "us-central1-docker.pkg.dev/modified-wonder-468716-e8/myrepo/ml-pipeline:latest"
)

@dsl.component(base_image=IMAGE_URI)
def train_op(
    model_path: dsl.OutputPath(str),
    data_path: str,
    commit_id: str = "unknown",
    score: dsl.Output[float] = None,
    metrics: dsl.Output[dsl.Metrics] = None
):
    """Train a model, save it, and output the score."""
    import joblib, os
    import trainer.task as my_model

    # Train the model
    model, model_score = my_model.train(data_path=data_path)

    # Ensure folder exists
    os.makedirs(model_path, exist_ok=True)

    # Save model with commit ID
    out_file = os.path.join(model_path, f"model_{commit_id}.joblib")
    joblib.dump(model, out_file)
    print(f"✅ Model saved to {out_file}, R² = {model_score:.4f}")

    # Write score to output
    if score is not None:
        score.write(str(model_score))

    # Log metrics
    if metrics is not None:
        metrics.log_metric("r2_score", model_score)


@dsl.component(base_image=IMAGE_URI)
def evaluate_op(score: float, threshold: float = 0.75) -> bool:
    """Check if the model meets the quality bar."""
    passed = score >= threshold
    print(f"📊 Model score = {score}, threshold = {threshold}, passed = {passed}")
    return passed


@dsl.component(base_image=IMAGE_URI)  # Could also use IMAGE_URI
def deploy_op(model_path: str, commit_id: str = "unknown"):
    """Deploy the model to Vertex AI."""
    from google.cloud import aiplatform

    aiplatform.init(project="modified-wonder-468716-e8", location="us-central1")

    model = aiplatform.Model.upload(
        display_name=f"taxi-model-{commit_id}",
        artifact_uri=model_path,  # folder containing model
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
    )

    endpoint = model.deploy(
        deployed_model_display_name=f"taxi-endpoint-{commit_id}",
        machine_type="n1-standard-2"
    )
    print(f"🚀 Model deployed at endpoint {endpoint.resource_name}")


@dsl.pipeline(name="conditional-deploy-pipeline")
def pipeline(
    data_path: str = "gs://taxi_model028/data/processed2/n_20_trips-00003-of-00030.jsonl",
    commit_id: str = "unknown",
    threshold: float = 0.75
):
    # Train the model
    train_task = train_op(data_path=data_path, commit_id=commit_id)

    # Evaluate the score
    score_val = train_task.outputs["score"]
    passed = evaluate_op(score=score_val, threshold=threshold).output

    # Conditional deployment
    with dsl.If(passed):
        deploy_op(
            model_path=train_task.outputs["model_path"],
            commit_id=commit_id
        )
