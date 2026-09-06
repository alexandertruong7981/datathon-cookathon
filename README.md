# PM2.5 next-hour prediction

The active workflow is in `AT_attempt/`. It compares linear regression,
XGBoost and a neural network using the same chronological validation split.

Install dependencies and run from the repository root:

```powershell
python -m pip install -r requirements.txt
python AT_attempt/model_comparison.py
python AT_attempt/create_best_predictions.py
```

For interactive exploration, open `AT_attempt/AT_Attempt.ipynb` with
`AT_attempt/` as the working directory. The comparison script also updates
its modelling cells and saves executed results.

Inputs are `AT_attempt/train.csv`, `AT_attempt/test(1).csv` and
`AT_attempt/sample_submission.csv`.

`AT_attempt/AT_Outputs/` contains validation metrics and predictions,
correlation diagnostics, OLS coefficients, XGBoost importances, the final
XGBoost model, preprocessing settings and training metadata.
`AT_attempt/best_predictions.csv` contains the final test predictions.

XGBoost achieved validation RMSE 26.81, compared with 29.47 for the neural
network. The final prediction script uses this selected XGBoost configuration
and refits on all 360,954 training rows, including the validation period.
Missing wind direction is handled by masking direction-derived features as
NaN for XGBoost. Assertions verify that every training row reaches the final
fit; `AT_Outputs/best_predictions_metadata.json` records the row counts.
Validation metrics describe the comparison models before this full-data refit.
