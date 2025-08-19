import argparse
from google.cloud import aiplatform
from kfp import compiler
import pipelines.pipeline as my_pipeline
import os

def main(args):
    aiplatform.init(project=args.project, location=args.region)

    # Compile pipeline JSON with commit tag
    json_path = f"pipeline_{args.commit_id}.json"
    compiler.Compiler().compile(
        pipeline_func=my_pipeline.pipeline,
        package_path=json_path,
    )

    job = aiplatform.PipelineJob(
        display_name=args.display_name,
        template_path=json_path,
        pipeline_root=args.pipeline_root,
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
