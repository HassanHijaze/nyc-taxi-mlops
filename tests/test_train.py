import pandas as pd

from src.train import FEATURES, TARGET, preprocess, should_promote


def make_trips():
    pickup = pd.Timestamp("2024-01-01 10:00")
    return pd.DataFrame(
        {
            "lpep_pickup_datetime": [pickup] * 4,
            "lpep_dropoff_datetime": [
                pickup + pd.Timedelta(minutes=10),  # normal trip
                pickup + pd.Timedelta(seconds=30),  # too short
                pickup + pd.Timedelta(minutes=90),  # too long
                pickup + pd.Timedelta(minutes=10),  # distance too small
            ],
            "PULocationID": [74, 74, 74, 74],
            "DOLocationID": [75, 75, 75, 75],
            "trip_distance": [2.0, 2.0, 2.0, 0.05],
            "fare_amount": [10.0, 5.0, 50.0, 10.0],
        }
    )


def test_preprocess_removes_outliers():
    df = preprocess(make_trips())
    assert len(df) == 1


def test_preprocess_computes_duration_in_minutes():
    df = preprocess(make_trips())
    assert df[TARGET].iloc[0] == 10


def test_preprocess_location_ids_are_strings():
    df = preprocess(make_trips())
    assert df["PULocationID"].iloc[0] == "74"
    assert df["DOLocationID"].iloc[0] == "75"


def test_preprocess_keeps_only_features_and_target():
    df = preprocess(make_trips())
    assert list(df.columns) == FEATURES + [TARGET]


def test_should_promote_without_production_model():
    assert should_promote(5.0, None)


def test_should_promote_better_model():
    assert should_promote(4.9, 5.0)


def test_should_promote_equal_model():
    assert should_promote(5.0, 5.0)


def test_should_not_promote_worse_model():
    assert not should_promote(5.1, 5.0)
