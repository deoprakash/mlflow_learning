import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient


TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "iris-classifier-model"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()


def get_dataset_for_model(model_version):

    version = client.get_model_version(
        MODEL_NAME,
        model_version
    )

    run = client.get_run(version.run_id)

    dataset_hash = run.data.params.get("dataset_hash")
    dataset_path = run.data.params.get("versioned_data")

    print(f"Model version: {version.version}")
    print(f"Run ID: {version.run_id}")
    print(f"Dataset hash: {dataset_hash}")
    print(f"Dataset path: {dataset_path}")

    if dataset_path:
        path = Path(dataset_path)

        if path.exists():
            print()
            print("✅ Exact dataset found.")
            print(f"File: {path}")
        else:
            print()
            print("❌ Dataset file not found locally.")

    return dataset_path


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage:")
        print("python training/get_dataset.py <model_version>")
        sys.exit(1)

    model_version = sys.argv[1]

    get_dataset_for_model(model_version)