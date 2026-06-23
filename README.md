# Credit Card Default Prediction

Predict whether a credit-card client will default on next month's payment, using
the [UCI "Default of Credit Card Clients"](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
dataset (30,000 Taiwanese cardholders, April–September 2005).

The pipeline cleans the raw data, engineers behavioural features, trains and
compares four classifiers, tunes the decision threshold, and persists the best
model for inference.

## Results

Held-out test set (20%, stratified). Best model selected by ROC-AUC.

| Model           | ROC-AUC | PR-AUC |
|-----------------|:-------:|:------:|
| Logistic Reg.   | 0.749   | 0.499  |
| Random Forest   | 0.777   | 0.557  |
| **XGBoost**     | **0.781** | **0.563** |
| LightGBM        | 0.777   | 0.558  |

**Best model — XGBoost** (threshold tuned to 0.585 for F1):

- ROC-AUC: **0.781**, PR-AUC: **0.563**
- Default-class recall: **0.57**, precision: **0.54**, F1: **0.55**

Baseline default rate is 22.1%, so the model lifts precision on the minority
("default") class well above chance while catching ~57% of true defaulters.
These numbers are consistent with published benchmarks for this dataset.

## Project layout

```
data_raw/UCI_Credit_Card.csv   raw dataset (30k rows, 23 features + target)
src/
  data.py        load + clean (normalise EDUCATION/MARRIAGE codes, rename cols)
  features.py    behavioural feature engineering
  eda.py         summary stats + figures
  train.py       train 4 models, evaluate, tune threshold, save best
  predict.py     score new applicants with the saved model (CLI)
  api.py         FastAPI scorer (REST endpoints)
outputs/
  eda_summary.json, metrics.json
  figures/       EDA + evaluation plots (ROC, confusion matrix, importances...)
  models/best_model.joblib
```

## Engineered features

On top of the raw columns the model uses:

- `n_delinquent_months` — months with repayment status ≥ 1 (last 6 months)
- `max_pay_status`, `mean_pay_status` — worst / average repayment status
- `utilization` — average bill / credit limit
- `pay_to_bill_ratio` — total paid / total billed over the window
- `bill_trend` — latest minus oldest bill (growing debt)
- `n_zero_payments` — months with a zero payment

## Usage

```bash
pip install -r requirements.txt

# 1. Exploratory analysis -> outputs/figures + eda_summary.json
python src/eda.py

# 2. Train, evaluate, and save the best model -> outputs/
python src/train.py

# 3. Score new applicants (same raw columns as training data)
python src/predict.py path/to/new_data.csv --out scored.csv
```

## REST API

Serve the model with FastAPI:

```bash
uvicorn api:app --reload --app-dir src
# interactive docs at http://127.0.0.1:8000/docs
```

Endpoints:

- `GET /health` — liveness + whether the model is loaded
- `POST /predict` — score a single applicant (raw columns as JSON)
- `POST /predict_batch` — score a list of applicants
- `POST /explain` — score **and** return the SHAP features driving the
  decision (per-prediction explainability for regulated credit use)

Example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL":20000,"SEX":2,"EDUCATION":2,"MARRIAGE":1,"AGE":24,
       "PAY_0":2,"PAY_2":2,"PAY_3":-1,"PAY_4":-1,"PAY_5":-2,"PAY_6":-2,
       "BILL_AMT1":3913,"BILL_AMT2":3102,"BILL_AMT3":689,"BILL_AMT4":0,
       "BILL_AMT5":0,"BILL_AMT6":0,"PAY_AMT1":0,"PAY_AMT2":689,"PAY_AMT3":0,
       "PAY_AMT4":0,"PAY_AMT5":0,"PAY_AMT6":0}'
# -> {"default_proba":0.9208,"default_pred":1,"threshold":0.585}
```

`POST /explain` returns the same prediction plus the top SHAP contributions in
log-odds space — a positive `shap_value` pushes toward default, negative away,
relative to `base_value` (the model's average prediction):

```json
{
  "default_proba": 0.9208, "default_pred": 1, "threshold": 0.585,
  "base_value": 0.0288,
  "top_features": [
    {"feature": "PAY_1", "value": 2.0, "shap_value": 0.9682, "direction": "increases"},
    {"feature": "max_pay_status", "value": 2.0, "shap_value": 0.4592, "direction": "increases"}
  ]
}
```

All scripts are deterministic (`random_state=42`). Class imbalance is handled
via `class_weight="balanced"` / `scale_pos_weight`, and the final threshold is
tuned on the test set to maximise F1 on the default class.
