from kfp import dsl
import os

IMAGE_URI = os.environ.get(
    "PIPELINE_IMAGE_URI",
    "us-central1-docker.pkg.dev/modified-wonder-468716-e8/myrepo/ml-pipeline:latest"
)

# ---- Train Component ----
@dsl.component(base_image=IMAGE_URI)
def train_op(
    model_path: dsl.OutputPath(str),
    score: dsl.Output[float],
    data_path: str,
    commit_id: str = "unknown"
) ->float:
    import joblib, os
    import trainer.task as my_model

    # ---- Train the model ----
    model, r2_score = my_model.train(data_path=data_path)

    # ---- Save model ----
    os.makedirs(model_path, exist_ok=True)
    out_file = os.path.join(model_path, f"model_{commit_id}.joblib")
    joblib.dump(model, out_file)
    print(f"Model saved to {out_file}, R² = {r2_score:.4f}")

    return r2_score

# ---- Deploy Component ----
@dsl.component(base_image=IMAGE_URI)
def deploy_op(model_path: str, commit_id: str = "unknown"):
    from google.cloud import aiplatform

    project = "modified-wonder-468716-e8"
    location = "us-central1"
    endpoint_display_name = "taxi-endpoint"  # fixed endpoint name

    aiplatform.init(project=project, location=location)

    # Upload the new model
    model = aiplatform.Model.upload(
        display_name=f"taxi-model-{commit_id}",
        artifact_uri=model_path,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
    )

    # Try to find existing endpoint
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"',
        location=location
    )
    if endpoints:
        endpoint = endpoints[0]
        print(f"Reusing existing endpoint: {endpoint.resource_name}")
    else:
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)
        print(f"Created new endpoint: {endpoint.resource_name}")

    # (Optional) undeploy older models
    if endpoint.traffic_split:  # means some models already deployed
        for model_id in endpoint.traffic_split.keys():
            endpoint.undeploy(model_id=model_id)
            print(f"Undeployed previous model: {model_id}")

    # Deploy new model to the endpoint
    endpoint.deploy(
        model=model,
        deployed_model_display_name=f"taxi-model-{commit_id}",
        machine_type="n1-standard-2",
    )
    print(f"✅ Model {commit_id} deployed at endpoint {endpoint.resource_name}")


# ---- Pipeline ----
@dsl.pipeline(name="conditional-deploy-pipeline")
def pipeline(
    data_path: str = "gs://taxi_model028/data/processed2/n_20_trips-00003-of-00030.jsonl",
    commit_id: str = "unknown",
    threshold: float = 0.75
):
    # Train the model
    train_task = train_op(data_path=data_path, commit_id=commit_id)

    with dsl.If(train_task.output>=threshold):
        deploy_op(
            model_path=train_task.outputs["model_path"],
            commit_id=commit_id
        )
