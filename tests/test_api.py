from fastapi.testclient import TestClient
from unittest.mock import patch


class FakeModel:

    def predict(self, features):
        return [0]


with patch("mlflow.sklearn.load_model", return_value=FakeModel()):
    from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Iris prediction API is running"


def test_predict():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    )

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert result["prediction"] in [0, 1, 2]

def test_predict_invalid_input():
    response = client.post(
        "/predict",
        json={
            "sepal_length": "hello",
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    )

    assert response.status_code == 422