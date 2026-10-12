import pandas as pd
from sklearn.datasets import make_classification, make_regression

from retail_ml.models import ClassifierSuite, RegressorSuite
from retail_ml.sql_reports import run_kpi_queries


def _positive(X):
    return pd.DataFrame(abs(X), columns=[f"f{i}" for i in range(X.shape[1])])


def test_classifier_suite_selects_best_model():
    X, y = make_classification(n_samples=300, n_features=6, random_state=0)
    X = _positive(X)
    suite = ClassifierSuite(0)
    metrics = suite.fit_evaluate(X[:240], y[:240], X[240:], y[240:])
    assert set(metrics) == {"logreg_l1", "logreg_l2", "random_forest", "xgboost"}
    assert suite.best_name in metrics
    assert len(suite.feature_importance(X.columns)) == 6


def test_regressor_suite_runs():
    X, y = make_regression(n_samples=300, n_features=5, noise=5, random_state=0)
    X = _positive(X)
    suite = RegressorSuite(0)
    suite.fit_evaluate(X[:240], y[:240], X[240:], y[240:])
    assert suite.metrics[suite.best_name]["rmse"] > 0


def test_sql_kpis():
    sales = pd.DataFrame({
        "invoice": ["1", "2", "3"], "stock_code": ["11111", "11111", "22222"],
        "description": ["A", "A", "B"], "quantity": [1, 2, 1], "price": [1.0, 1.0, 5.0],
        "invoice_date": pd.to_datetime(["2011-01-05", "2011-01-20", "2011-02-01"]),
        "customer_id": [1, 1, 2], "country": ["UK", "UK", "FR"],
    })
    sales["revenue"] = sales["quantity"] * sales["price"]
    out = run_kpi_queries(sales)
    assert out["monthly_revenue"]["revenue"].tolist() == [3.0, 5.0]
    assert out["repeat_customer_rate"].loc[0, "repeat_customers"] == 1
