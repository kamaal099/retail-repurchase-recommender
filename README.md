# Retail Repurchase Prediction & Product Recommender

End-to-end retail machine learning project on the real **Online Retail II** transaction data
(UK online gift retailer, Dec 2009 - Dec 2011). It answers three business questions:

1. **Classification:** Will this customer buy again in the next 90 days?
2. **Regression:** How much will they spend in the next 90 days?
3. **Recommendation:** Which products should we show each customer next?

## The business problem

Online retailers earn most of their revenue from repeat customers, but they can't treat every customer the same way. Marketing budget is limited, and two mistakes are costly:

- **Wasted discounts and emails** sent to customers who would have bought anyway.
- **Missed chances to win back** customers who are about to drift away.

This project helps a retail team answer three practical questions for each customer:

| Question | Output | How the business uses it |
|---|---|---|
| Will they buy again in the next 90 days? | Repurchase probability | Target win-back campaigns at customers who are likely to leave, and skip discounts for those who are likely to return |
| How much will they spend? | Predicted spend (GBP) | Prioritise high-value customers and set the campaign budget |
| What should we show them? | Top-10 product recommendations | Personalise emails, homepage and cross-sell suggestions |

On the Online Retail II data, the recommender found a product the customer then bought for 36.2% of customers, against 29.0% for simply showing the most popular products. The API serves all three predictions, so a CRM or website could call it directly.

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
error messages, logging, unit tests (pytest), Dockerfile, FastAPI service and GitHub Actions CI.

## Results

Evaluated on a held-out 20% of customers (classification and regression) and on the final 90 days of data (recommender). Full numbers are in `reports/metrics.json`.

| Task | Best model | Key metric |
|---|---|---|
| Repurchase classification | XGBoost | ROC-AUC 0.816 (L2 logistic regression 0.805) |
| Spend regression (log scale) | XGBoost | RMSE 2.52, R² 0.38 |
| Recommendation | Item-item CF | hit-rate@10 36.2% vs 29.0% popularity baseline |

Data: 776,577 cleaned sales lines, 5,256 modelled customers, cutoff date 10 Sep 2011.
A simple logistic regression gets within 0.01 AUC of XGBoost, so recency and frequency carry most of the signal.

## Run the API

After running the pipeline once (it saves the trained models to `artifacts/`):

```bash
python -m uvicorn retail_ml.api:app --port 8000
```

Open http://127.0.0.1:8000/docs to try the endpoints:

- `GET /customers/{customer_id}/repurchase`: probability of buying again in 90 days and predicted spend (GBP)
- `GET /customers/{customer_id}/recommendations?k=10`: top-k products with descriptions
- `GET /health`: service status

## Limitations

- One time cutoff, so results reflect a single period. Rolling-origin validation would be stronger.
- The recommender uses purchase history only (no product text or prices).
- Many customers are wholesalers, so spend is heavily skewed.
- Results are measured offline on historical data. A real rollout would need an A/B test to measure the actual lift in sales.s
