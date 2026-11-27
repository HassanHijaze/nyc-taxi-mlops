import os
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_URI = "models:/green-taxi-ride-duration@production"


class Trip(BaseModel):
    PULocationID: int
    DOLocationID: int
    trip_distance: float


class Prediction(BaseModel):
    duration: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    app.state.model = mlflow.sklearn.load_model(MODEL_URI)
    yield


app = FastAPI(title="Green taxi trip duration", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(trip: Trip):
    # Location IDs are categories, same as in training
    features = {
        "PULocationID": str(trip.PULocationID),
        "DOLocationID": str(trip.DOLocationID),
        "trip_distance": trip.trip_distance,
    }
    duration = app.state.model.predict([features])[0]
    return Prediction(duration=float(duration))
