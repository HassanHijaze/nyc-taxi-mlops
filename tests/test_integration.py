import os

import httpx
import pytest

pytestmark = pytest.mark.integration

API_URL = os.getenv("API_URL", "http://localhost:8000")
TRIP = {"PULocationID": 236, "DOLocationID": 239, "trip_distance": 1.98}


def test_health():
    response = httpx.get(f"{API_URL}/health")
    assert response.status_code == 200


def test_predict_returns_reasonable_duration():
    response = httpx.post(f"{API_URL}/predict", json=TRIP)
    assert response.status_code == 200
    assert 0 < response.json()["duration"] < 60
