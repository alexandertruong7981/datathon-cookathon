"""Predict new hourly covariates with a saved train_pm25 model bundle."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from train_pm25 import TARGET, features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("outputs/pm25_model.joblib"))
    parser.add_argument("--input", type=Path, default=Path("test(1).csv"))
    parser.add_argument("--history", type=Path, default=Path("train.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.csv"))
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    future = pd.read_csv(args.input, parse_dates=["observation_timestamp"])
    history = pd.read_csv(args.history, parse_dates=["observation_timestamp"])
    assert future.id.is_unique and history.id.is_unique, "IDs must be unique"
    assert not set(future.id) & set(history.id), "History and input IDs overlap"
    start = future.observation_timestamp.min()
    assert history.observation_timestamp.max() < start, "History must precede input"
    # 24-hour average lagged by 3 hours needs 26 prior hours, plus fill context.
    history = history.loc[history.observation_timestamp >= start - pd.Timedelta(hours=29)]
    frame = pd.concat([history, future], ignore_index=True).drop(columns=TARGET, errors="ignore")
    X = features(frame).reindex(columns=bundle["feature_columns"], fill_value=0).loc[future.id]
    prediction = (np.full(len(future), bundle["mean"]) if bundle["model"] is None
                  else np.maximum(0, bundle["model"].predict(X)))
    assert np.isfinite(prediction).all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": future.id, TARGET: prediction}).to_csv(args.output, index=False)
    print(f"Saved {len(future)} predictions to {args.output}")


if __name__ == "__main__":
    main()
