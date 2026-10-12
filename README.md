# Retail Repurchase Prediction & Product Recommender

End-to-end retail machine learning project on the real **Online Retail II** transaction data
(UK online gift retailer, Dec 2009 - Dec 2011). It answers three business questions:

1. **Classification:** Will this customer buy again in the next 90 days?
2. **Regression:** How much will they spend in the next 90 days?
3. **Recommendation:** Which products should we show each customer next?

## Setup

```bash
pip install -r requirements.txt
```

Download the dataset (see `data/README.md`) and put `online_retail_II.xlsx` (or a `.csv`) in `data/`.

```bash
PYTHONPATH=src python -m retail_ml.pipeline --data-dir data --output-dir reports   # Linux/macOS
$env:PYTHONPATH="src"; python -m retail_ml.pipeline --data-dir data --output-dir reports   # Windows PowerShell
pytest -q
```

Docker: `docker build -t retail-ml .` then
`docker run --rm -v "$PWD/data:/app/data" -v "$PWD/reports:/app/reports" retail-ml`

## Method

- **Cleaning:** drop rows with no customer ID, non-product codes (postage, fees), zero/negative
  price or quantity; cancellations (invoices starting with `C`) are kept separately as returns.
- **Leakage-free design:** a cutoff date is placed 90 days before the last transaction. Features use
  only data before the cutoff; targets come only from the 90 days after it.
- **Features (13):** recency, frequency, monetary value, average order value, tenure, product variety,
  basket size, mean days between orders, orders and revenue in the last 90 days, return rate, UK flag.
  All computed with vectorised pandas `groupby` operations (no row-wise loops).
- **Models:** Logistic Regression (L1 and L2), Random Forest and XGBoost for classification;
  Ridge (L2), Lasso (L1), Random Forest and XGBoost for regression on log(1 + spend).
  Best model chosen on a held-out test set (ROC-AUC / RMSE).
- **Recommender:** item-item collaborative filtering (cosine similarity on a sparse purchase matrix),
  benchmarked against a popularity baseline using hit-rate@10 and recall@10 on new items bought
  in the future window.
- **SQL:** monthly revenue, top products, revenue by country and repeat-customer rate run in DuckDB.

## Engineering

Modular package (`src/retail_ml`), OOP model and recommender classes, custom exceptions with clear
error messages, logging, unit tests (pytest), Dockerfile and GitHub Actions CI.

## Results

Run the pipeline, then paste the numbers from `reports/metrics.json` here:

| Task | Best model | Key metric |
|---|---|---|
| Repurchase classification | _fill in_ | ROC-AUC _fill in_ |
| Spend regression | _fill in_ | RMSE _fill in_ |
| Recommendation | item-item CF vs popularity | hit-rate@10 _fill in_ |

## Limitations

- One time cutoff, so results reflect a single period. Rolling-origin validation would be stronger.
- The recommender uses purchase history only (no product text or prices).
- Many customers are wholesalers, so spend is heavily skewed.
