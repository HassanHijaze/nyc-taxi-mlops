import argparse
import time

import requests

from src.train import COLOR, YEAR, download_data, load_data, preprocess

API_URL = "http://localhost:8000/predict"


def main():
    parser = argparse.ArgumentParser(description="Replay taxi trips to the API.")
    parser.add_argument("--month", type=int, default=2)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument(
        "--sleep", type=float, default=0.0, help="seconds between requests"
    )
    args = parser.parse_args()

    df = preprocess(load_data(download_data(YEAR, args.month, COLOR)))
    trips = df.drop(columns=["duration"]).head(args.n).to_dict(orient="records")

    for i, trip in enumerate(trips, start=1):
        response = requests.post(API_URL, json=trip, timeout=5)
        response.raise_for_status()
        time.sleep(args.sleep)
        if i % 100 == 0:
            print(f"Sent {i} trips")

    print(f"Done: sent {len(trips)} trips from {YEAR}-{args.month:02d}")


if __name__ == "__main__":
    main()
