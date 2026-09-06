## Approach

The raw data is first placed onto a complete hourly timeline for each monitoring station. Missing short gaps are interpolated for feature construction while the original values and missingness information are also retained.

The first feature set contains 231 features based on pollutant and weather measurements around the target hour, rolling statistics, city wide conditions, pollutant ratios, calendar information and wind features.

The second feature set expands this to 375 features by using a wider time window, 48 hour history, network minimum and maximum values, pressure trends and PM10, CO and NO2 readings from each individual monitoring station.

Validation is chronological rather than random. The main holdout period is September 2015 to February 2016 so that it matches the same seasonal period as the hidden test set.

## Results

| Model | Holdout RMSE |
|---|---:|
| Mean baseline | 102.04 |
| Scaled PM10 baseline | 47.36 |
| LightGBM v1 | 21.39 |
| LightGBM v2 | 20.95 |
| v1 and v2 blend | 20.90 |
| v1, v2 and Variant W blend | **20.88** |

The final three model blend uses 20% v1, 55% v2 and 25% Variant W. Variant W gives additional training weight to observations with high PM10 and helps diversify the model errors. :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}

## Running the Workflow

Run the notebook from top to bottom:

```text
1. Load and inspect the data
2. Build feature set v1
3. Build feature set v2
4. Run time aware validation
5. Analyse validation errors
6. Test calibration
7. Train the final v1 and v2 models
8. Train Variant W and create the final blend