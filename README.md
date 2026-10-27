# Retail Repurchase Prediction & Product Recommender

End-to-end retail machine-learning project using the **Online Retail II** transaction dataset (a UK online gift retailer, Dec 2009–Dec 2011). It combines customer analytics, predictive modelling, product recommendations, SQL KPIs, and a FastAPI service.

## Business questions

1. **Repurchase classification:** Is a customer likely to buy in the next 90-day window?
2. **Spend regression:** How much might a customer spend in that window?
3. **Product recommendations:** Which previously unseen products should be surfaced to a customer?

| Use case | Output | Potential business action |
|---|---|---|
| Retention | Repurchase probability | Prioritise customers for appropriate retention campaigns |
| Customer value | Predicted future spend (GBP) | Rank customers for campaign planning |
| Cross-sell | Top-k product recommendations | Personalise product discovery and email campaigns |

These are offline decision-support outputs, not proof of incremental revenue. A real deployment would need monitoring and a controlled experiment.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset following [data/README.md](data/README.md) and place `online_retail_II.xlsx` (or a supported CSV) in `data/`.

Run the pipeline:

```bash
# Linux / macOS
PYTHONPATH=src python -m retail_ml.pipeline --data-dir data --output-dir reports --artifact-dir artifacts

# Windows PowerShell
$env:PYTHONPATH="src"
python -m retail_ml.pipeline --data-dir data --output-dir reports --artifact-dir artifacts

# Tests
pytest -q
```

Run with Docker:

```bash
docker build -t retail-ml .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/reports:/app/reports" retail-ml
```

## Methodology

- **Cleaning:** exclude rows without customer IDs and non-product stock codes; separate cancellation invoices as return lines.
- **Temporal holdout:** the final `horizon_days` calendar dates are the future window; features use only transactions before its cutoff, and targets use transactions in the future window.
- **Customer features:** recency, frequency, monetary value, average order value, tenure, product variety, basket statistics, recent orders/revenue, a return-line share proxy, and UK indicator.
- **Classification candidates:** L1/L2 Logistic Regression, Random Forest, and XGBoost; primary selection metric is validation ROC-AUC.
- **Regression candidates:** Ridge, Lasso, Random Forest, and XGBoost on log(1 + future spend); primary selection metric is validation RMSE.
- **Evaluation protocol:** candidates are trained on a training partition and selected on a separate validation partition. The selected candidate is refit on training + validation data and scored once on an untouched test partition. Validation and final test metrics are reported separately.
- **Recommendations:** item-item collaborative filtering (cosine similarity) compared with a popularity baseline. Reports hit-rate@k, recall@k, and catalog coverage on future items not present in each customer's history.
- **SQL:** DuckDB queries report monthly revenue, top products, revenue by country, and repeat-customer rate.

## Results

Results below were regenerated from the Online Retail II CSV under the validation-based model-selection protocol. The saved report is `reports/metrics.json`. The evaluation used 5,256 modelled customers, 776,577 cleaned sales rows, and a 90-day future window from 11 Sep 2011 through 9 Dec 2011. Customers were split approximately 60% train, 20% validation, and 20% test; the selected models were refit on train + validation before final test scoring.

| Task | Selected model (validation) | Final test metrics |
|---|---|---|
| Repurchase classification | Random Forest | ROC-AUC 0.811; PR-AUC 0.794; F1 0.696 |
| Spend regression | Random Forest | RMSE 2.532; MAE 2.090; R² 0.370 |

Regression metrics are measured on the modelled log(1 + future spend) target, not directly in GBP. The API converts predicted spend back to GBP.

### Recommendation benchmark

Evaluated on 2,204 customers with new products in the future window.

| Metric at k=10 | Popularity baseline | Item-item CF |
|---|---:|---:|
| Hit rate | 29.04% | 36.21% |
| Recall | 1.68% | 3.09% |
| Catalog coverage | 1.80% | 17.65% |

Item-item CF improves historical hit rate by about 7.17 percentage points over popularity. These are offline historical metrics and do not establish incremental revenue or causal business lift. Results may vary across time; rolling-origin validation and a controlled online experiment would provide stronger evidence.

## API

After a successful pipeline run:

```bash
PYTHONPATH=src python -m uvicorn retail_ml.api:app --port 8000
```

Open `http://127.0.0.1:8000/docs` to explore:

- `GET /health` — service and artifact status
- `GET /customers/{customer_id}/repurchase` — repurchase probability and predicted spend
- `GET /customers/{customer_id}/recommendations?k=10` — top-k products

## Engineering

- Modular `src/retail_ml` package with custom exceptions and logging
- Reusable classifier/regressor suites and recommender classes
- Pytest unit tests and GitHub Actions CI
- FastAPI inference service and Docker support
- Machine-readable evaluation and business KPI reports

## Limitations

- A single temporal cutoff does not show how performance varies across time; rolling-origin validation would be stronger.
- The return feature is a **return-line share proxy**, not a matched-item return probability, because cancellation lines cannot reliably be paired with original sale lines.
- Item-item recommendations use purchase co-occurrence only; they do not use product text, price, stock availability, or margin.
- Spend is highly skewed and many customers may be wholesalers.
- Offline ranking metrics do not establish causal business lift; deployment requires monitoring, drift checks, and an A/B test.
