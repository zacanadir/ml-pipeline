import argparse
import re
import os
from google.cloud import aiplatform
from kfp import compiler
import pipelines.pipeline as my_pipeline


def clean_string(value: str, max_len: int = 128) -> str:
    """
    Sanitize a string for Vertex AI (display names, commit IDs, etc.)
    - Keep only letters, numbers, dash, underscore, dot
    - Replace invalid characters with '_'
    - Trim to max_len
    """
    if not value:
        return "unknown"
    safe = re.sub(r'[^a-zA-Z0-9._-]', '_', value)
    return safe[:max_len]


def main(args):
    aiplatform.init(project=args.project, location=args.region)

    # --- Sanitize inputs ---
    safe_commit_id = clean_string(args.commit_id, max_len=64)
    safe_display_name = clean_string(args.display_name, max_len=128)

    # --- Compile pipeline JSON ---
    json_path = f"pipeline_{safe_commit_id}.json"
    compiler.Compiler().compile(
        pipeline_func=my_pipeline.pipeline,
        package_path=json_path,
    )

    # --- Submit pipeline job ---
    print(f"🚀 Submitting pipeline with display_name='{safe_display_name}', commit_id='{safe_commit_id}'")
    job = aiplatform.PipelineJob(
        display_name=safe_display_name,
        template_path=json_path,
        pipeline_root=args.pipeline_root,
        parameter_values={
            "commit_id": safe_commit_id
        }
    )
    job.run(sync=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--pipeline_root", required=True)
    parser.add_argument("--display_name", required=True)
    parser.add_argument("--commit_id", required=True)
    args = parser.parse_args()
    main(args)
