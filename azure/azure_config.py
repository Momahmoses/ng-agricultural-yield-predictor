"""Azure config for Agricultural Yield Predictor."""
import os

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_CONTAINER_FARMS = "agri-farm-records"
AZURE_CONTAINER_MODELS = "agri-models"
AZURE_ML_WORKSPACE = os.getenv("AZURE_ML_WORKSPACE", "")
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")


def upload_to_blob(local_path: str, blob_name: str, container: str) -> None:
    try:
        from azure.storage.blob import BlobServiceClient
        client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        with open(local_path, "rb") as f:
            client.get_blob_client(container=container, blob=blob_name).upload_blob(f, overwrite=True)
    except ImportError:
        print("Install azure-storage-blob")


def get_databricks_job(script="pipeline/spark_pipeline.py") -> dict:
    return {
        "run_name": "AgriYieldPipeline",
        "spark_python_task": {"python_file": f"dbfs:/FileStore/{script}"},
    }
