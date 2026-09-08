'''This file answers:

How do I interact with my ML model?'''

import mlflow
import mlflow.sklearn


class IrisPredictor:

    def __init__(self):
        # mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_tracking_uri("http://host.docker.internal:5000")

        self.model = mlflow.sklearn.load_model(
            "models:/iris-classifier-model@champion"
        )

    def predict(self, features: list[float]) -> int:
        prediction = self.model.predict([features])

        return int(prediction[0])