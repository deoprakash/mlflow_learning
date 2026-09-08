from fastapi import FastAPI

from app.schemas import IrisInput
from app.predictor import IrisPredictor


app = FastAPI()

predictor = IrisPredictor()


@app.get("/")
def home():
    return {
        "message": "Iris prediction API is running"
    }


@app.post("/predict")
def predict(data: IrisInput):

    features = [
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]

    prediction = predictor.predict(features)

    return {
        "prediction": prediction
    }