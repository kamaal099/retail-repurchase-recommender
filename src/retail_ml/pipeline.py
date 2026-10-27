"""End-to-end run: load -> clean -> features -> models -> recommender -> reports."""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .artifacts import save_artifacts
from .cleaning import clean_transactions
from .config import Config
from .data_loader import RetailDataLoader
from .exceptions import DataValidationError, RetailMLError
from .features import (FEATURE_COLUMNS, add_targets, build_customer_features,
                       log_target, temporal_split)
from .models import ClassifierSuite, RegressorSuite
from .recommender import ItemItemRecommender, PopularityRecommender, evaluate_recommender
from .sql_reports import run_kpi_queries

logger = logging.getLogger("retail_ml")


def run(config: Config) -> dict:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    raw = RetailDataLoader(config.data_dir).load()
    sales, returns = clean_transactions(raw)
    split = temporal_split(sales, config.horizon_days)
    logger.info("First future date (cutoff): %s", split.cutoff.date())

    features = add_targets(
        build_customer_features(split.history, returns, split.cutoff), split.future)
    if features["will_repurchase"].nunique() < 2:
        raise DataValidationError("Repurchase label has a single class; check the data.")

    X = features[FEATURE_COLUMNS]
    y_cls = features["will_repurchase"]
    y_reg = log_target(features["future_spend"])

    # Hold out test data first. Select candidates on validation, then refit the
    # selected model on train + validation before evaluating once on the test set.
    X_train_valid, X_test, ycl_train_valid, ycl_test, yr_train_valid, yr_test = train_test_split(
        X, y_cls, y_reg, test_size=config.test_size,
        random_state=config.random_state, stratify=y_cls)
    X_train, X_valid, ycl_train, ycl_valid, yr_train, yr_valid = train_test_split(
        X_train_valid, ycl_train_valid, yr_train_valid, test_size=0.25,
        random_state=config.random_state, stratify=ycl_train_valid)

    logger.info("Training classifiers; selecting by validation ROC-AUC")
    clf = ClassifierSuite(config.random_state)
    clf.fit_evaluate(X_train, ycl_train, X_valid, ycl_valid, X_test, ycl_test)
    logger.info("Training regressors; selecting by validation RMSE")
    reg = RegressorSuite(config.random_state)
    reg.fit_evaluate(X_train, yr_train, X_valid, yr_valid, X_test, yr_test)

    logger.info("Fitting recommenders")
    popularity = PopularityRecommender().fit(split.history)
    item_item = ItemItemRecommender().fit(split.history)
    rec_metrics = {
        "popularity_baseline": evaluate_recommender(popularity, split.history, split.future, config.top_k),
        "item_item_cf": evaluate_recommender(item_item, split.history, split.future, config.top_k),
    }

    results = {
        "evaluation_protocol": {
            "customer_split": {"train": round(len(X_train) / len(X), 4),
                               "validation": round(len(X_valid) / len(X), 4),
                               "test": round(len(X_test) / len(X), 4)},
            "model_selection": "validation only",
            "final_test": "selected model refit on train + validation, then evaluated once",
            "temporal_horizon_days": config.horizon_days,
            "test_size_config": config.test_size,
        },
        "data": {
            "sales_rows": int(len(sales)),
            "customers_modelled": int(len(features)),
            "cutoff_date": str(split.cutoff.date()),
            "last_future_date": str(pd.to_datetime(split.future["invoice_date"]).max().date()),
            "repurchase_rate": float(y_cls.mean()),
        },
        "classification": {"best_model": clf.best_name,
                            "validation_metrics_by_model": clf.validation_metrics,
                            "final_test_metrics": clf.test_metrics},
        "regression": {"best_model": reg.best_name,
                       "validation_metrics_by_model": reg.validation_metrics,
                       "final_test_metrics": reg.test_metrics},
        "recommendation": rec_metrics,
    }
    out = config.output_dir
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    clf.feature_importance(FEATURE_COLUMNS).to_csv(out / "feature_importance_classifier.csv")
    reg.feature_importance(FEATURE_COLUMNS).to_csv(out / "feature_importance_regressor.csv")
    for name, frame in run_kpi_queries(sales).items():
        frame.to_csv(out / f"kpi_{name}.csv", index=False)
    save_artifacts(config.artifact_dir, clf.best_model, reg.best_model,
                   features[FEATURE_COLUMNS], split.history, sales)
    logger.info("Saved API artifacts to %s", config.artifact_dir.resolve())
    logger.info("Done. Results written to %s", out.resolve())
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--horizon-days", type=int, default=90)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = Config(data_dir=Path(args.data_dir), output_dir=Path(args.output_dir),
                    artifact_dir=Path(args.artifact_dir), horizon_days=args.horizon_days)
    try:
        results = run(config)
    except RetailMLError as exc:
        logger.error("%s", exc)
        return 1
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
