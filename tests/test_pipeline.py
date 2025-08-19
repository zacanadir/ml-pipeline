import os
import tempfile
from kfp import compiler
import pipelines.pipeline as my_pipeline

def test_pipeline_compiles():
    """Ensure pipeline compiles into JSON successfully."""
    os.environ["PIPELINE_IMAGE_URI"] = "gcr.io/test/fake-image:latest"

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = tmpdir + "/test_pipeline.json"

        compiler.Compiler().compile(
            pipeline_func=my_pipeline.pipeline,
            package_path=json_path,
        )
        assert os.path.exists(json_path)
