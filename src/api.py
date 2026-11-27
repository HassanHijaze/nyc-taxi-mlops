import os
from contextlib import asynccontextmanager
from time import perf_counter

import mlflow
import requests
from fastapi import BackgroundTasks, FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

MODEL_URI = "models:/green-taxi-ride-duration@production"
EVIDENTLY_URL = os.getenv("EVIDENTLY_URL", "http://localhost:8001")

PREDICTION_REQUESTS = Counter(
    "api_prediction_requests_total", "Total number of prediction requests."
)
PREDICTION_LATENCY = Histogram(
    "api_prediction_latency_seconds", "Latency of prediction requests in seconds."
)
PREDICTED_DURATION = Histogram(
    "api_predicted_duration_minutes",
    "Predicted trip duration in minutes.",
    buckets=[5, 10, 15, 20, 30, 45, 60],
)


class Trip(BaseModel):
    # NYC TLC taxi zones are numbered 1 to 265
    PULocationID: int = Field(ge=1, le=265)
    DOLocationID: int = Field(ge=1, le=265)
    # Same distance range that train.py keeps
    trip_distance: float = Field(ge=0.1, le=100)


class Prediction(BaseModel):
    duration: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    app.state.model = mlflow.sklearn.load_model(MODEL_URI)
    yield


app = FastAPI(title="Green taxi trip duration", lifespan=lifespan)


def send_to_evidently(event: dict) -> None:
    # Monitoring must never break predictions, so errors are ignored
    try:
        requests.post(f"{EVIDENTLY_URL}/events", json=event, timeout=2)
    except requests.RequestException:
        pass


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=Prediction)
def predict(trip: Trip, background_tasks: BackgroundTasks):
    start = perf_counter()

    # Location IDs are categories, same as in training
    features = {
        "PULocationID": str(trip.PULocationID),
        "DOLocationID": str(trip.DOLocationID),
        "trip_distance": trip.trip_distance,
    }
    duration = float(app.state.model.predict([features])[0])

    PREDICTION_REQUESTS.inc()
    PREDICTION_LATENCY.observe(perf_counter() - start)
    PREDICTED_DURATION.observe(duration)
    background_tasks.add_task(send_to_evidently, {**features, "prediction": duration})

    return Prediction(duration=duration)
