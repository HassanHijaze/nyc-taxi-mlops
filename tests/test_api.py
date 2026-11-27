from fastapi.testclient import TestClient

from src.api import app


class FakeModel:
    def predict(self, X):
        self.received = X
        return [12.5]


def make_client():
    # No "with" block, so the startup code (loading from MLflow) does not run
    app.state.model = FakeModel()
    return TestClient(app)


TRIP = {"PULocationID": 236, "DOLocationID": 239, "trip_distance": 1.98}


def test_health():
    response = make_client().get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_duration():
    response = make_client().post("/predict", json=TRIP)
    assert response.status_code == 200
    assert response.json() == {"duration": 12.5}


def test_predict_sends_location_ids_as_strings():
    make_client().post("/predict", json=TRIP)
    sent = app.state.model.received[0]
    assert sent["PULocationID"] == "236"
    assert sent["DOLocationID"] == "239"


def test_predict_rejects_invalid_input():
    response = make_client().post("/predict", json={"PULocationID": 236})
    assert response.status_code == 422


def test_predict_rejects_invalid_location_id():
    response = make_client().post("/predict", json={**TRIP, "PULocationID": 0})
    assert response.status_code == 422


def test_predict_rejects_invalid_distance():
    response = make_client().post("/predict", json={**TRIP, "trip_distance": 0.05})
    assert response.status_code == 422


def test_metrics_counts_predictions():
    client = make_client()
    client.post("/predict", json=TRIP)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_prediction_requests_total" in response.text
