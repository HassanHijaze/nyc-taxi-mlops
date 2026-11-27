import os
import threading
from collections import deque
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.metrics import DriftedColumnsCount, ValueDrift
from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest
from pydantic import BaseModel
from starlette.responses import Response

REFERENCE_PATH = os.getenv("REFERENCE_PATH", "data/reference.parquet")
REPORT_PATH = Path(os.getenv("REPORT_PATH", "reports/drift_report.html"))
WINDOW_SIZE = 200  # number of recent predictions compared with the reference
RUN_EVERY = 50  # recompute drift after this many new events

COLUMNS = ["PULocationID", "DOLocationID", "trip_distance", "prediction"]
# Zero-padded labels so Grafana sorts the buckets in the right order
BINS = [0, 5, 10, 15, 20, 30, 45, 60, float("inf")]
BUCKETS = ["00-05", "05-10", "10-15", "15-20", "20-30", "30-45", "45-60", "60+"]

DRIFT_SHARE = Gauge("evidently_drift_share", "Share of drifted columns.")
PREDICTION_DRIFT_SCORE = Gauge(
    "evidently_prediction_drift_score",
    "Wasserstein distance for predictions (above 0.1 means drift).",
)
COLUMN_DRIFT_SCORE = Gauge(
    "evidently_column_drift_score", "Drift score per column.", ["column"]
)
DURATION_SHARE = Gauge(
    "evidently_duration_share",
    "Share of predictions per duration bucket.",
    ["dataset", "bucket"],
)
WINDOW_ROWS = Gauge("evidently_window_rows", "Rows in the current window.")


class Event(BaseModel):
    PULocationID: str
    DOLocationID: str
    trip_distance: float
    prediction: float


app = FastAPI(title="Evidently drift monitor")
reference = pd.read_parquet(REFERENCE_PATH)
window = deque(maxlen=WINDOW_SIZE)
lock = threading.Lock()
event_count = 0


def set_duration_shares(dataset: str, predictions: pd.Series) -> None:
    buckets = pd.cut(predictions, bins=BINS, labels=BUCKETS, right=False)
    shares = buckets.value_counts(normalize=True)
    for bucket in BUCKETS:
        DURATION_SHARE.labels(dataset=dataset, bucket=bucket).set(shares[bucket])


set_duration_shares("training", reference["prediction"])


def compute_drift(current: pd.DataFrame) -> None:
    report = Report(
        [DriftedColumnsCount()] + [ValueDrift(column=column) for column in COLUMNS]
    )
    snapshot = report.run(reference_data=reference, current_data=current)
    results = snapshot.dict()["metrics"]

    DRIFT_SHARE.set(results[0]["value"]["share"])
    for column, result in zip(COLUMNS, results[1:]):
        COLUMN_DRIFT_SCORE.labels(column=column).set(result["value"])
    PREDICTION_DRIFT_SCORE.set(results[COLUMNS.index("prediction") + 1]["value"])
    set_duration_shares("current", current["prediction"])

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(REPORT_PATH))


@app.post("/events")
def add_event(event: Event):
    global event_count
    with lock:
        window.append(event.model_dump())
        event_count += 1
        WINDOW_ROWS.set(len(window))
        if event_count % RUN_EVERY == 0:
            compute_drift(pd.DataFrame(window))
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
