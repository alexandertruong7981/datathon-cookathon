"""Refit the selected XGBoost configuration and predict every competition test row."""
import ast
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from model_comparison import MODEL_CODE


def main():
    directory = Path(__file__).resolve().parent
    notebook = json.loads((directory / "AT_Attempt.ipynb").read_text(encoding="utf-8"))
    train = pd.read_csv(directory / "train.csv")
    test = pd.read_csv(directory / "test(1).csv")
    sample = pd.read_csv(directory / "sample_submission.csv")
    target = "PM2_5_next_hour"
    features = train.columns.drop(["id", "observation_timestamp", target]).tolist()
    namespace = {"pd": pd, "np": np, "X_train": train[features]}
    preparation = next(c for c in notebook["cells"] if c.get("id") == "seasonal-log-imputation")
    # Refit preprocessing on all labelled rows, using the notebook's exact logic.
    source = "".join(preparation["source"]).split("X_train_preprocessed =")[0]
    exec(source, namespace)
    prepared_train = namespace["preprocess_features"](train[features])
    prepared_test = namespace["preprocess_features"](test[features])
    missing_train_direction = prepared_train["wd"].isna()
    # Use an observed category only to construct the feature schema, then mask
    # every direction-derived feature for these rows before fitting.
    observed_directions = sorted(prepared_train["wd"].dropna().unique().tolist())
    if not observed_directions:
        raise ValueError("Training data must contain an observed wind direction.")
    prepared_train.loc[missing_train_direction, "wd"] = observed_directions[0]
    labels = train.loc[prepared_train.index, target]
    interaction_cell = next(c for c in notebook["cells"] if c.get("id") == "centered-interactions")
    tree = ast.parse("".join(interaction_cell["source"]))
    definitions = ast.Module(body=[n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))], type_ignores=[])
    exec(compile(definitions, "interactions", "exec"), namespace)
    numerical = prepared_train.select_dtypes(include="number").columns.difference(
        ["year", "month", "day", "hour"], sort=False).tolist()
    main_train, products_train, settings = namespace["make_interactions"](
        prepared_train, continuous_columns=numerical, categorical_columns=["station", "wd"])
    model_train = pd.concat([main_train, products_train], axis=1)
    # Preserve every test row. XGBoost supports missing numerical feature values.
    # A temporary category allows encoding; ALL direction-derived columns are
    # subsequently masked for missing/unseen directions, so no direction is imputed.
    unknown_direction = ~prepared_test["wd"].isin(settings["levels"]["wd"])
    prepared_test.loc[unknown_direction, "wd"] = settings["levels"]["wd"][0]
    main_test, products_test, _ = namespace["make_interactions"](prepared_test, fitted=settings)
    model_test = pd.concat([main_test, products_test], axis=1)
    wd_columns = [c for c in model_test.columns if c.startswith("wd_") or "_x_wd_" in c]
    model_train[wd_columns] = model_train[wd_columns].astype(float)
    model_train.loc[missing_train_direction, wd_columns] = np.nan
    model_test[wd_columns] = model_test[wd_columns].astype(float)
    model_test.loc[unknown_direction, wd_columns] = np.nan
    assert model_train.columns.equals(model_test.columns)
    assert model_train.index.equals(train.index), "Final fit must retain every training row"
    assert labels.index.equals(train.index) and np.isfinite(labels).all()
    print(f"Fitting selected XGBoost on {len(model_train):,} rows and {model_train.shape[1]} features", flush=True)
    # Reuse precisely the configuration evaluated in the model comparison.
    model_node = next(n for n in ast.parse(MODEL_CODE).body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "xgb_model" for t in n.targets))
    model_ns = {"XGBRegressor": XGBRegressor}
    exec(compile(ast.Module(body=[model_node], type_ignores=[]), "selected_model", "exec"), model_ns)
    model = model_ns["xgb_model"]
    model.fit(model_train, labels)
    prediction = model.predict(model_test)
    submission = pd.DataFrame({"id": test["id"], target: prediction})
    assert submission.columns.equals(sample.columns)
    assert submission["id"].equals(sample["id"])
    assert submission["id"].is_unique and len(submission) == len(test)
    assert np.isfinite(prediction).all()
    output = directory / "best_predictions.csv"
    submission.to_csv(output, index=False)
    artifacts = directory / "AT_Outputs"
    artifacts.mkdir(exist_ok=True)
    model.save_model(artifacts / "final_xgboost.ubj")
    joblib.dump({"interaction_settings": settings, "feature_columns": features,
                 "seasonal_means": namespace["seasonal_means"],
                 "season_map": namespace["season_map"],
                 "log_columns": namespace["log_columns"],
                 "log_column_names": namespace["log_column_names"],
                 "log_means": namespace["log_means"]}, artifacts / "final_preprocessing.joblib")
    metadata = {"training_rows": len(model_train), "source_training_rows": len(train),
                "all_training_rows_used": model_train.index.equals(train.index),
                "training_missing_wd": int(missing_train_direction.sum()),
                "test_rows": len(test),
                "test_missing_or_unseen_wd": int(unknown_direction.sum()),
                "missing_wd_policy": "Mask all wd-derived features as NaN; use XGBoost missing-value routing",
                "training_policy": "Refit selected configuration on every training row, including validation rows and missing wd",
                "prediction_min": float(prediction.min()), "prediction_max": float(prediction.max()),
                "model_parameters": model.get_params()}
    (artifacts / "best_predictions_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(submission):,} predictions to {output}", flush=True)
    print(f"Missing/unseen test wd handled natively: {unknown_direction.sum():,}", flush=True)


if __name__ == "__main__":
    main()
