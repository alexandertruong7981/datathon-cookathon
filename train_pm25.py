"""PDF-inspired, causal station-level one-hour PM2.5 regression."""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

TARGET = "PM2_5_next_hour"
SENSORS = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]


def features(frame):
    """Use only covariates available at t; reindex gaps to actual clock hours."""
    parts = []
    for station, rows in frame.groupby("station", sort=True):
        rows = rows.sort_values("observation_timestamp").set_index("observation_timestamp")
        if rows.index.has_duplicates:
            raise ValueError(f"Duplicate station timestamps: {station}")
        grid = rows.reindex(pd.date_range(rows.index.min(), rows.index.max(), freq="h"))
        raw = grid[SENSORS].astype(float)
        # Short gaps only; longer gaps remain missing for train-fitted imputation.
        values = raw.ffill(limit=3)
        cols = {name: values[name] for name in SENSORS}
        for name in SENSORS:
            cols[name + "_missing"] = raw[name].isna().astype(float)
            for lag in (1, 2, 3):
                cols[f"{name}_lag{lag}"] = values[name].shift(lag)
            for window in (6, 12, 24):
                avg = values[name].rolling(window, min_periods=1).mean()
                cols[f"{name}_mean{window}"] = avg
                for lag in (1, 2, 3):
                    cols[f"{name}_mean{window}_lag{lag}"] = avg.shift(lag)
        out = pd.DataFrame(cols, index=grid.index).loc[rows.index]
        out["station"] = station
        out["wd"] = rows.wd.fillna("unknown")
        out["year"] = out.index.year
        out["month"] = out.index.month
        out["day"] = out.index.day
        out["hour"] = out.index.hour
        out["weekday"] = out.index.dayofweek
        out["timestamp_days"] = (out.index - pd.Timestamp("2013-01-01")).total_seconds() / 86400
        for name, period in (("hour", 24), ("month", 12), ("weekday", 7)):
            out[name + "_sin"] = np.sin(2 * np.pi * out[name] / period)
            out[name + "_cos"] = np.cos(2 * np.pi * out[name] / period)
        out.index = rows.id
        parts.append(out)
    result = pd.get_dummies(pd.concat(parts), columns=["station", "wd"], dtype=float)
    return result.reindex(frame.id).astype("float32")


def scores(y, prediction):
    return {"RMSE": float(np.sqrt(mean_squared_error(y, prediction))),
            "MAE": float(mean_absolute_error(y, prediction)),
            "R2": float(r2_score(y, prediction))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.data_dir / "train.csv", parse_dates=["observation_timestamp"])
    test = pd.read_csv(args.data_dir / "test(1).csv", parse_dates=["observation_timestamp"])
    assert train.id.is_unique and test.id.is_unique
    assert not set(train.id) & set(test.id)
    assert train.observation_timestamp.max() < test.observation_timestamp.min()
    # Explicitly discard labels before constructing any feature.
    combined = pd.concat([train.drop(columns=TARGET), test], ignore_index=True)
    print("Building causal features", flush=True)
    X = features(combined)
    xtrain, xtest = X.loc[train.id], X.loc[test.id]
    y = train[TARGET].to_numpy()
    end = train.observation_timestamp.max() + pd.Timedelta(hours=1)
    holdout_start = end - pd.Timedelta(days=60)
    validation_start = holdout_start - pd.Timedelta(days=60)
    t = train.observation_timestamp
    # Purge one hour so training target timestamps precede evaluation origins.
    fit = (t < validation_start - pd.Timedelta(hours=1)).to_numpy()
    valid = ((t >= validation_start) & (t < holdout_start - pd.Timedelta(hours=1))).to_numpy()
    holdout = (t >= holdout_start).to_numpy()
    models = {
        "linear_regression": make_pipeline(SimpleImputer(add_indicator=True), MinMaxScaler(), LinearRegression()),
        "random_forest": make_pipeline(SimpleImputer(add_indicator=True), RandomForestRegressor(
            n_estimators=100, max_depth=18, min_samples_leaf=10, max_features=0.7,
            max_samples=0.5, n_jobs=4, random_state=42)),
        "gradient_boosting": LGBMRegressor(n_estimators=1800, learning_rate=0.035,
            num_leaves=31, max_depth=-1, min_child_samples=100, colsample_bytree=0.85,
            reg_lambda=5, n_jobs=4, random_state=42, verbosity=-1),
    }
    results = {"training_mean": scores(y[valid], np.full(valid.sum(), y[fit].mean()))}
    for name, model in models.items():
        print(f"Training {name}", flush=True)
        if name == "gradient_boosting":
            model.fit(xtrain.iloc[fit], y[fit], eval_set=[(xtrain.iloc[valid], y[valid])],
                      eval_metric="rmse", callbacks=[early_stopping(100), log_evaluation(200)])
        else:
            model.fit(xtrain.iloc[fit], y[fit])
        results[name] = scores(y[valid], np.maximum(0, model.predict(xtrain.iloc[valid])))
        print(name, results[name], flush=True)
    winner = min(results, key=lambda key: results[key]["RMSE"])
    preholdout = (t < holdout_start - pd.Timedelta(hours=1)).to_numpy()
    selected = models.get(winner)
    if selected is not None:
        if winner == "gradient_boosting":
            selected.set_params(n_estimators=selected.best_iteration_)
        selected.fit(xtrain.iloc[preholdout], y[preholdout])
        predictions = np.maximum(0, selected.predict(xtrain.iloc[holdout]))
    else:
        predictions = np.full(holdout.sum(), y[preholdout].mean())
    report = {"validation_start": str(validation_start), "holdout_start": str(holdout_start),
              "rows": {"fit": int(fit.sum()), "validation": int(valid.sum()), "holdout": int(holdout.sum())},
              "feature_count": X.shape[1], "validation": results, "selected_model": winner,
              "holdout": scores(y[holdout], predictions),
              "holdout_mean_baseline": scores(y[holdout], np.full(holdout.sum(), y[preholdout].mean()))}
    print(json.dumps(report, indent=2), flush=True)
    pd.DataFrame({"id": train.loc[holdout, "id"], "actual": y[holdout], "prediction": predictions}).to_csv(args.output_dir / "holdout_predictions.csv", index=False)
    station_scores = []
    for station in sorted(train.station.unique()):
        mask = train.loc[holdout, "station"].to_numpy() == station
        station_scores.append({"station": station, "rows": int(mask.sum()),
                               **scores(y[holdout][mask], predictions[mask])})
    pd.DataFrame(station_scores).to_csv(args.output_dir / "holdout_by_station.csv", index=False)
    if selected is not None:
        selected.fit(xtrain, y)
        prediction = np.maximum(0, selected.predict(xtest))
    else:
        prediction = np.full(len(test), y.mean())
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    assert set(sample.id) == set(test.id) and sample.id.is_unique
    submission = pd.DataFrame({"id": test.id, TARGET: prediction}).set_index("id").loc[sample.id].reset_index()
    assert np.isfinite(submission[TARGET]).all()
    submission.to_csv(args.output_dir / "submission.csv", index=False)
    joblib.dump({"model": selected, "mean": float(y.mean()), "feature_columns": X.columns.tolist(),
                 "sensor_columns": SENSORS}, args.output_dir / "pm25_model.joblib", compress=3)
    if winner == "gradient_boosting":
        pd.Series(selected.booster_.feature_importance(importance_type="gain"), index=X.columns,
                  name="gain").sort_values(ascending=False).to_csv(args.output_dir / "feature_importance.csv")
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    print("Saved model, metrics and submission to", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
