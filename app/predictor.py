'''This file answers:

How do I interact with my ML model?'''

import mlflow
import mlflow.sklearn
import os


class IrisPredictor:

    def __init__(self):
        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5000"
        )
        mlflow.set_tracking_uri(tracking_uri)

        self.model = mlflow.sklearn.load_model(
            "models:/iris-classifier-model@champion"
        )

    def predict(self, features: list[float]) -> int:
        prediction = self.model.predict([features])

        return int(prediction[0])