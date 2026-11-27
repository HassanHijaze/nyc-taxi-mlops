import logging
import os
from pathlib import Path
from urllib.request import urlretrieve

import mlflow
import pandas as pd
import skops.io as sio
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# Reduce utility log noise without hiding warnings about model packaging.
logging.getLogger("mlflow.utils").setLevel(logging.WARNING)

YEAR = 2024
MONTH = 1
COLOR = "green"

FEATURES = ["PULocationID", "DOLocationID", "trip_distance"]
TARGET = "duration"

EXPERIMENT_NAME = "green-taxi-monitoring"
MODEL_NAME = "green-taxi-ride-duration"


def download_data(year, month, color):
    data_path = Path(f"./data/{color}_tripdata_{year}-{month:02d}.parquet")
    data_path.parent.mkdir(exist_ok=True)

    if not data_path.exists():
        urlretrieve(
            f"https://d37ci6vzurychx.cloudfront.net/trip-data/{color}_tripdata_{year}-{month:02d}.parquet",
            data_path,
        )
    return data_path


def load_data(data_path):
    return pd.read_parquet(data_path)


# Calculate duration, remove outliers, and treat the location IDs as categories
def preprocess(df):
    df["duration"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]
    df = df[df["trip_distance"].between(0.1, 100)]
    df[["PULocationID", "DOLocationID"]] = df[["PULocationID", "DOLocationID"]].astype(
        str
    )
    df = df[FEATURES + [TARGET]]
    return df


def split(df_processed):
    y = df_processed["duration"]
    X = df_processed.drop(columns=["duration"])

    return train_test_split(X, y, random_state=42, test_size=0.2)


def train(X_train, y_train):
    pipeline = make_pipeline(DictVectorizer(), LinearRegression())
    pipeline.fit(X_train.to_dict(orient="records"), y_train)
    return pipeline


def evaluate(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test.to_dict(orient="records"))
    return root_mean_squared_error(y_test, y_pred)


def main():
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    data_path = download_data(YEAR, MONTH, COLOR)
    df = load_data(data_path)
    df_processed = preprocess(df)
    X_train, X_test, y_train, y_test = split(df_processed)

    with mlflow.start_run() as run:
        mlflow.set_tags(
            {
                "model": "linear regression",
                "developer": "Hassan",
                "dataset": f"{COLOR}-taxi",
                "year": YEAR,
                "month": MONTH,
                "features": FEATURES,
                "target": TARGET,
            }
        )

        pipeline = train(X_train, y_train)
        rmse = evaluate(pipeline, X_test, y_test)
        mlflow.log_metric("rmse", rmse)

        model_path = Path("models/model.skops")
        model_path.parent.mkdir(exist_ok=True)
        sio.dump(pipeline, model_path)

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline, name="mlflow-model", serialization_format="skops"
        )

        client = MlflowClient()
        try:
            client.create_registered_model(
                name=MODEL_NAME,
                tags={"framework": "scikit-learn"},
                description="Model to predict ride duration in NYC",
            )
        except MlflowException as exc:
            # Reuse the model only if it already exists; re-raise any other MLflow error.
            if exc.error_code != "RESOURCE_ALREADY_EXISTS":
                raise

        mv = client.create_model_version(
            name=MODEL_NAME,
            source=model_info.model_uri,
            model_id=model_info.model_id,
            run_id=run.info.run_id,
        )
        client.set_registered_model_alias(
            name=MODEL_NAME, version=mv.version, alias="production"
        )

    print(f"RMSE: {rmse}")
    print(f"Registered model: {MODEL_NAME}, version {mv.version}, alias: production")
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
