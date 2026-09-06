# PM2.5 next-hour prediction

`train_pm25.py` adapts the regression approach in `COMP9417_Group_Project.pdf`
(methodology pp. 3–4) to the supplied multi-station dataset. The target is
`PM2_5_next_hour`, already aligned to the next hour; it is never shifted or
used as an input feature.

Run from this directory:

```powershell
python -m pip install -r requirements.txt
python train_pm25.py
```

Reuse the trained model without retraining:

```powershell
python predict_pm25.py --input "test(1).csv" --history train.csv
```

History must precede the input period and use the same sensor schema. Only
the last 29 hours of history are needed. The full model comparison and refits can take around 20–30 minutes
on a laptop; prediction is considerably faster.

The script compares a training-mean baseline, ordinary linear regression,
random forest, and LightGBM gradient boosting. It selects by validation RMSE,
evaluates the winner on a separate chronological holdout, then retrains on
all labelled rows and predicts `test(1).csv` in sample-submission order.

Features follow the PDF: calendar variables, 1/2/3-hour sensor lags,
6/12/24-hour moving averages, and 1/2/3-hour lags of those averages.
Additional cyclic calendar encoding and one-hot station/wind direction features
support this dataset. Windows include the current hour, which is available
when forecasting the next hour. Each station is reindexed to an hourly grid
before computing lags, so missing records do not compress time.

Adaptations from the PDF:

- Short missing sensor gaps use forward filling for at most three hours.
  Linear regression and random forest use training-fitted mean imputation
  with missing indicators; LightGBM handles remaining missing values directly.
  Bidirectional splines are omitted because they can use future readings;
  full-data kNN imputation is expensive at this dataset's scale.
- Linear regression uses training-fitted min-max scaling. Trees need no scaling.
- No current PM2.5 measurement is supplied, so the PDF's ground-truth
  persistence baseline is unavailable. Target-derived lags are excluded.
- Pollution spikes are retained. This is a regression task, so classification
  and the PDF's multi-horizon LSTM are outside this implementation.

Validation uses the 60 days before the final 60 days of labelled data.
The final 60 days form the holdout. A one-hour purge before each evaluation
boundary keeps fitted target timestamps earlier than evaluation origins.
All stations use the same date boundaries. Gradient boosting uses early
stopping on validation only; the resulting tree count is fixed for refits.
RMSE is the selection metric; MAE and R² are also reported.

This is rolling one-hour forecasting: observations up to each prediction's
timestamp are available, including earlier validation/test covariates.
It is not a forecast of the entire test period from one fixed origin.
The summer holdout does not establish performance across every test season.

Results from the supplied data (seed 42, 211 features):

| Model | Validation RMSE | Validation MAE |
| --- | ---: | ---: |
| Training mean | 48.14 | 41.17 |
| Linear regression | 23.91 | 16.63 |
| Random forest | **21.03** | **13.06** |
| Gradient boosting | 21.22 | 13.36 |

Random forest was selected. On the untouched holdout beginning
2016-07-02 23:00, it achieved **RMSE 15.98, MAE 10.95, R² 0.852**
over 16,906 rows. The holdout training-mean baseline scored RMSE 46.72.
Validation begins 2016-05-03 23:00. Different seasonal distributions mean
the lower holdout error should not be interpreted as an improvement from
tuning on that period. Test targets are unavailable, so test accuracy is unknown.

Outputs are written to `outputs/`:

- `submission.csv`: IDs and next-hour PM2.5 predictions, ready for submission.
- `metrics.json`: validation comparison and independent holdout metrics.
- `holdout_predictions.csv`: actual and predicted holdout values.
- `holdout_by_station.csv`: station-level holdout errors.
- `pm25_model.joblib`: final fitted estimator and feature schema.
- `feature_importance.csv`: gain importances if gradient boosting wins.

The saved estimator expects the engineered feature schema. For new hourly
data, use `features()` with the preceding 26 hours of station covariates
(plus three more hours for forward-fill context), then align columns with
`feature_columns` in the saved bundle. Retraining with the script is the
supported end-to-end workflow.
