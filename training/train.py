import mlflow
import mlflow.sklearn
import mlflow.data
import hashlib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def calculate_file_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def train():

    # TRACKING_URI = "http://host.docker.internal:5000"
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("iris-classifier")

    data_path = Path("data/iris.csv")

    dataset_hash = calculate_file_hash(data_path)

    print(f"Dataset hash: {dataset_hash}")

    versioned_dataset_path = Path("data/versions/") / f"{dataset_hash}.csv"

    if not versioned_dataset_path.exists():
        versioned_dataset_path.write_bytes(data_path.read_bytes())
        print("Saved dataset version: ", versioned_dataset_path)

    else:
        print("Dataset version already exists: ", versioned_dataset_path)

    df = pd.read_csv(data_path)

    x = df[
        [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]       
    ]

    y = df["target"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

    n_estimators = 100
    max_depth = 5

    print("Tracking URI:", mlflow.get_tracking_uri())

    with mlflow.start_run():

        mlflow.log_param("dataset_file", str(data_path))

        mlflow.log_param("dataset_hash", dataset_hash)

        mlflow.log_param("versioned_data", versioned_dataset_path)

        dataset = mlflow.data.from_numpy(
            features=x.to_numpy(),
            targets=y.to_numpy(),
            name="iris-dataset"
        )

        mlflow.log_input(
            dataset,
            context="training"
        )

        # algorithm = "RandomForestClassifier"
        # model = RandomForestClassifier(
        #     n_estimators = n_estimators,
        #     max_depth = max_depth,
        #     random_state = 42
        # )    

        # algorithm = "AdaBoostClassifier"
        # model = AdaBoostClassifier(
        #     n_estimators = n_estimators,
        #     random_state = 42
        # )    

        algorithm = "GradientBoostingClassifier"
        model = GradientBoostingClassifier(
            n_estimators = n_estimators,
            max_depth = max_depth,
            random_state = 42
        )

        model.fit(x_train, y_train)

        predictions = model.predict(x_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="weighted")
        recall = recall_score(y_test, predictions, average="weighted")
        f1 = f1_score(y_test, predictions, average="weighted")

        mlflow.log_param(
            "algorithm",
            algorithm
        )

        mlflow.log_param(
            "n_estimators",
            n_estimators
        )

        mlflow.log_param(
            "max_depth",
            max_depth
        )

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        model_info = mlflow.sklearn.log_model(
            model, 
            name="iris_model",
            registered_model_name="iris-classifier-model",
        )

        print(f"accuracy: {accuracy:.4f}")
        print(f"precision: {precision:.4f}")
        print(f"recall: {recall:.4f}")
        print(f"f1: {f1:.4f}")

        candidate_version = model_info.registered_model_version

        print(f"Registered Model Version: {candidate_version}")

        return candidate_version

if __name__ == "__main__":
    train()