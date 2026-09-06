"""Run the notebook's preparation and model cells, saving their actual outputs."""
import contextlib
import io
import json
import os
from pathlib import Path

# Bound BLAS parallelism before importing numerical libraries.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd


MODEL_CODE = '''from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from pathlib import Path
import xgboost

# Use the same retained observations and original target units for every model.
X_model_train = X_train_interactions
X_model_val = X_val_interactions
y_model_train = y_train_preprocessed
y_model_val = y_val_preprocessed
assert X_model_train.index.equals(y_model_train.index)
assert X_model_val.index.equals(y_model_val.index)
assert X_model_train.columns.equals(X_model_val.columns)
if not np.isfinite(X_model_train.to_numpy()).all() or not np.isfinite(X_model_val.to_numpy()).all():
    raise ValueError("Model features must be finite.")

# Drop constant columns using training data only; keep all remaining main effects.
feature_sd = X_model_train.std(ddof=0)
usable = feature_sd.index[feature_sd.gt(0)].tolist()
main_columns = [c for c in X_train_main_effects.columns if c in usable]
interaction_columns = [c for c in train_interactions.columns if c in usable]
scaler = StandardScaler()
Z_train = pd.DataFrame(scaler.fit_transform(X_model_train[usable]),
                       index=X_model_train.index, columns=usable)
Z_val = pd.DataFrame(scaler.transform(X_model_val[usable]),
                     index=X_model_val.index, columns=usable)

# Matrix multiplication computes all Pearson correlations on the training rows.
z = Z_train.to_numpy()
correlations = pd.DataFrame((z.T @ z) / len(z), index=usable, columns=usable)
target_correlations = Z_train.corrwith(y_model_train).abs()
upper = np.triu(np.ones(correlations.shape, dtype=bool), k=1)
high_correlations = correlations.where(upper).stack()
high_correlations = high_correlations[high_correlations.abs().ge(0.90)]
high_correlations = high_correlations.rename("pearson_r").rename_axis(
    ["feature_1", "feature_2"]).reset_index()

def screened_columns(threshold):
    # Retain main effects for interaction hierarchy. Prefer interactions with
    # stronger training target correlation when screening redundant products.
    kept = main_columns.copy()
    for column in target_correlations.reindex(interaction_columns).sort_values(ascending=False).index:
        if not correlations.loc[column, kept].abs().ge(threshold).any():
            kept.append(column)
    return kept

candidate_columns = {
    "OLS_main_effects": main_columns,
    "OLS_corr_0.90": screened_columns(0.90),
    "OLS_corr_0.98": screened_columns(0.98),
    "OLS_all_interactions": usable,
}
linear_models = {}
records = []
predictions = pd.DataFrame({"actual": y_model_val})

def prediction_metrics(y, prediction):
    return dict(RMSE=float(np.sqrt(mean_squared_error(y, prediction))),
                MAE=float(mean_absolute_error(y, prediction)),
                R2=float(r2_score(y, prediction)))

for name, columns in candidate_columns.items():
    print(f"Fitting {name}: {len(columns)} features", flush=True)
    model = LinearRegression(n_jobs=4).fit(Z_train[columns], y_model_train)
    train_prediction = model.predict(Z_train[columns])
    residual_ss = float(np.sum((y_model_train.to_numpy() - train_prediction) ** 2))
    n = len(y_model_train)
    # Gaussian maximum likelihood; effective design rank plus intercept and variance.
    k = int(model.rank_) + 2
    minus_two_log_likelihood = n * (np.log(2 * np.pi) + 1 + np.log(max(residual_ss / n, np.finfo(float).tiny)))
    records.append(dict(model=name, features=len(columns), rank=int(model.rank_),
                        AIC=minus_two_log_likelihood + 2 * k,
                        BIC=minus_two_log_likelihood + np.log(n) * k,
                        train_RMSE=np.sqrt(residual_ss / n)))
    linear_models[name] = model

# Select by training information criteria before looking at validation scores.
linear_selection = pd.DataFrame(records).set_index("model")
aic_choice = linear_selection["AIC"].idxmin()
bic_choice = linear_selection["BIC"].idxmin()
for record in records:
    name = record["model"]
    prediction = linear_models[name].predict(Z_val[candidate_columns[name]])
    predictions[name] = prediction
    record.update({f"validation_{key}": value for key, value in
                   prediction_metrics(y_model_val, prediction).items()})

# Fixed settings avoid tuning or early stopping against the reporting holdout.
# Trees can ignore features, but irrelevant features are not guaranteed to vanish.
print("Fitting XGBoost with all supplied features and interactions", flush=True)
xgb_model = XGBRegressor(
    objective="reg:squarederror", n_estimators=400, learning_rate=0.05,
    max_depth=6, min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=10.0, reg_alpha=0.1, tree_method="hist", n_jobs=4,
    random_state=42, eval_metric="rmse",
)
xgb_model.fit(X_model_train, y_model_train)
xgb_prediction = xgb_model.predict(X_model_val)
predictions["XGBoost"] = xgb_prediction
xgb_record = dict(model="XGBoost", features=X_model_train.shape[1], rank=np.nan,
                  AIC=np.nan, BIC=np.nan,
                  train_RMSE=prediction_metrics(y_model_train, xgb_model.predict(X_model_train))["RMSE"])
xgb_record.update({f"validation_{key}": value for key, value in
                   prediction_metrics(y_model_val, xgb_prediction).items()})
records.append(xgb_record)

# Fit target scaling on training data only and report in original PM2.5 units.
# The internal random early-stopping subset comes exclusively from training;
# the chronological reporting holdout is never used to stop or tune the MLP.
print("Fitting neural network: hidden layers (64, 32)", flush=True)
neural_network = TransformedTargetRegressor(
    regressor=MLPRegressor(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        alpha=0.01, batch_size=1024, learning_rate_init=0.001,
        max_iter=100, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=12, tol=0.0001, random_state=42,
    ),
    transformer=StandardScaler(),
)
neural_network.fit(Z_train.to_numpy(dtype=np.float32), y_model_train)
nn_prediction = neural_network.predict(Z_val.to_numpy(dtype=np.float32))
if not np.isfinite(nn_prediction).all():
    raise ValueError("Neural network predictions must be finite.")
predictions["Neural_network"] = nn_prediction
nn_record = dict(model="Neural_network", features=len(usable), rank=np.nan,
                 AIC=np.nan, BIC=np.nan,
                 train_RMSE=prediction_metrics(y_model_train, neural_network.predict(
                     Z_train.to_numpy(dtype=np.float32)))["RMSE"])
nn_record.update({f"validation_{key}": value for key, value in
                 prediction_metrics(y_model_val, nn_prediction).items()})
records.append(nn_record)
print(f"Neural network epochs: {neural_network.regressor_.n_iter_}; "
      f"internal early-stopping R2: {neural_network.regressor_.best_validation_score_:.4f}")
model_comparison = pd.DataFrame(records).set_index("model")
model_comparison["selected_by_AIC"] = model_comparison.index == aic_choice
model_comparison["selected_by_BIC"] = model_comparison.index == bic_choice
print(f"Training AIC choice: {aic_choice}; training BIC choice: {bic_choice}")
print(f"XGBoost version: {xgboost.__version__}")
display(model_comparison.round(4))

# OLS coefficients returned to the original demeaned-feature units.
selected_name = bic_choice
selected_columns = candidate_columns[selected_name]
selected_model = linear_models[selected_name]
scales = pd.Series(scaler.scale_, index=usable)
offsets = pd.Series(scaler.mean_, index=usable)
ols_coefficients = pd.Series(selected_model.coef_, index=selected_columns) / scales[selected_columns]
ols_intercept = float(selected_model.intercept_ - (ols_coefficients * offsets[selected_columns]).sum())
xgb_importance = pd.Series(xgb_model.feature_importances_, index=X_model_train.columns,
                          name="gain_importance").sort_values(ascending=False)
output_dir = (Path("AT_attempt") if Path("AT_attempt/train.csv").exists() else Path(".")) / "AT_Outputs"
output_dir.mkdir(exist_ok=True)
model_comparison.to_csv(output_dir / "model_comparison.csv")
predictions.to_csv(output_dir / "validation_predictions.csv", index_label="original_row")
high_correlations.to_csv(output_dir / "high_training_correlations.csv", index=False)
pd.concat([pd.Series({"intercept": ols_intercept}), ols_coefficients]).rename("coefficient").to_csv(
    output_dir / "bic_selected_ols_coefficients.csv")
xgb_importance.to_csv(output_dir / "xgboost_importance.csv")
print(f"Saved results to {output_dir}")
'''

NOTES = '''## 3. Linear regression, XGBoost and neural network comparison

All models use the same chronological train/validation split, retained wind-direction rows, original target units, encoded main effects and interactions. OLS compares main effects, two training-only correlation screens (absolute Pearson thresholds 0.90 and 0.98), and all interactions. Screens retain main effects so retained products have their constituent terms. Correlation screening is heuristic and does not eliminate all multicollinearity. Constants are removed and scaling is fitted only on training rows for OLS numerical stability.

AIC and BIC are Gaussian training-fit criteria, not backtesting scores. Lower is better among these OLS candidates on the same target and rows. Parameter counts use effective design rank plus intercept and residual variance. Temporal dependence and repeated stations weaken the independent-error interpretation of these criteria. AIC/BIC choices are recorded before validation scoring; RMSE, MAE and R-squared on the later holdout provide the chronological backtest. This is one holdout, not rolling cross-validation; earlier full-data exploratory analysis also means it is not a pristine untouched test.

XGBoost receives all features with fixed regularised settings and no tuning/early stopping on this holdout. Trees may leave features unused, but do not guarantee removal of every unnecessary variable. Conventional OLS AIC/BIC are not reported for boosted trees. Validation covers only rows with observed wind direction. Results, predictions, high-correlation pairs, BIC-selected OLS coefficients, and XGBoost gain importances are saved under `AT_Outputs`. Gain importance is not a causal effect.

The neural network is a scikit-learn MLP with ReLU hidden layers of 64 and 32 units, Adam optimisation, L2 regularisation and a maximum of 100 epochs. It receives all nonconstant main effects and interactions. Input and target standardisation are fitted on training rows only; predictions and metrics are returned to original target units. Early stopping uses an internal random 10% subset of training rows (not a temporal backtest), with patience of 12 epochs. This subset shares preprocessing fitted on the outer training set; only the later chronological holdout provides the reported comparison. Settings are fixed rather than tuned on the reporting holdout. OLS AIC/BIC do not apply to this network.
'''


def main():
    notebook_path = Path(__file__).with_name("AT_Attempt.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell_id, kind, source in [("model-comparison-notes", "markdown", NOTES),
                                  ("model-comparison", "code", MODEL_CODE)]:
        cell = next((c for c in notebook["cells"] if c.get("id") == cell_id), None)
        if cell is None:
            cell = {"cell_type": kind, "id": cell_id, "metadata": {}}
            notebook["cells"].append(cell)
        cell["source"] = source.splitlines(keepends=True)
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
    notebook_path.write_text(json.dumps(notebook, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    namespace = {"np": np, "pd": pd}
    namespace["df"] = pd.read_csv(notebook_path.with_name("train.csv"))
    namespace["var_df"] = namespace["df"].drop(columns=["id", "observation_timestamp"])
    split = next(c for c in notebook["cells"] if "validation_start = timestamp_counts.index" in "".join(c["source"]))
    cells = [split] + [next(c for c in notebook["cells"] if c.get("id") == cell_id)
                      for cell_id in ["seasonal-log-imputation", "1b21ff5b", "centered-interactions", "model-comparison"]]
    for count, cell in enumerate(cells, 1):
        print(f"Executing {cell.get('id')}", flush=True)
        outputs = []
        stream = io.StringIO()
        def display(value):
            outputs.append({"output_type": "display_data", "metadata": {},
                            "data": {"text/plain": [str(value)], **({"text/html": [value.to_html()]} if hasattr(value, "to_html") else {})}})
        namespace["display"] = display
        # Interaction code imports IPython only in earlier cells; these selected cells use display above.
        with contextlib.redirect_stdout(stream):
            exec(compile("".join(cell["source"]), f"notebook:{cell.get('id')}", "exec"), namespace)
        if stream.getvalue():
            outputs.insert(0, {"output_type": "stream", "name": "stdout", "text": stream.getvalue().splitlines(keepends=True)})
            print(stream.getvalue(), flush=True)
        cell.update(execution_count=count, outputs=outputs)
        notebook_path.write_text(json.dumps(notebook, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    print(namespace["model_comparison"].to_string(), flush=True)


if __name__ == "__main__":
    main()
