"""Candidate model suites selected on validation data, never on the final test set."""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from .exceptions import ModelTrainingError, NotFittedError


def _linear(estimator):
    # Features are non-negative and right-skewed; log1p then scale.
    return make_pipeline(FunctionTransformer(np.log1p), StandardScaler(), estimator)


def _concat_rows(left, right):
    """Concatenate train/validation rows for pandas or NumPy inputs."""
    if isinstance(left, (pd.Series, pd.DataFrame)) and isinstance(right, type(left)):
        return pd.concat([left, right], axis=0)
    return np.concatenate([np.asarray(left), np.asarray(right)], axis=0)


class BaseModelSuite(ABC):
    primary_metric: str
    higher_is_better: bool

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: dict = {}
        self.metrics: dict[str, dict[str, float]] = {}
        self.validation_metrics: dict[str, dict[str, float]] = {}
        self.test_metrics: dict[str, float] = {}
        self.best_name: str | None = None

    @abstractmethod
    def _candidates(self) -> dict: ...

    @abstractmethod
    def _score(self, model, X, y) -> dict[str, float]: ...

    def fit_evaluate(self, X_train, y_train, X_valid, y_valid, X_test=None, y_test=None):
        """Fit candidates on train; select by validation; optionally test once."""
        self.models = {}
        self.validation_metrics = {}
        self.test_metrics = {}
        for name, model in self._candidates().items():
            try:
                model.fit(X_train, y_train)
                self.models[name] = model
                self.validation_metrics[name] = self._score(model, X_valid, y_valid)
            except Exception as exc:
                raise ModelTrainingError(f"{name} failed to train/evaluate: {exc}") from exc
        pick = max if self.higher_is_better else min
        self.best_name = pick(
            self.validation_metrics,
            key=lambda n: self.validation_metrics[n][self.primary_metric],
        )
        self.metrics = self.validation_metrics
        if X_test is not None or y_test is not None:
            if X_test is None or y_test is None:
                raise ValueError("X_test and y_test must be provided together.")
            selected = self._candidates()[self.best_name]
            try:
                selected.fit(_concat_rows(X_train, X_valid), _concat_rows(y_train, y_valid))
                self.test_metrics = self._score(selected, X_test, y_test)
                self.models[self.best_name] = selected
            except Exception as exc:
                raise ModelTrainingError(f"Selected model {self.best_name} failed final fit/test: {exc}") from exc
        return self.metrics

    @property
    def best_model(self):
        if self.best_name is None:
            raise NotFittedError("Call fit_evaluate() first.")
        return self.models[self.best_name]

    def feature_importance(self, columns) -> pd.Series:
        model = self.best_model
        final = model[-1] if hasattr(model, "steps") else model
        if hasattr(final, "feature_importances_"):
            values = final.feature_importances_
        elif hasattr(final, "coef_"):
            values = np.abs(np.ravel(final.coef_))
        else:
            raise NotFittedError("Best model exposes no importances.")
        return pd.Series(values, index=list(columns)).sort_values(ascending=False)


class ClassifierSuite(BaseModelSuite):
    """Predict whether a customer buys again in the next window."""
    primary_metric = "roc_auc"
    higher_is_better = True

    def _candidates(self):
        rs = self.random_state
        return {
            "logreg_l1": _linear(LogisticRegression(penalty="l1", solver="liblinear",
                                                    class_weight="balanced", random_state=rs)),
            "logreg_l2": _linear(LogisticRegression(penalty="l2", max_iter=1000,
                                                    class_weight="balanced", random_state=rs)),
            "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                    class_weight="balanced", n_jobs=-1,
                                                    random_state=rs),
            "xgboost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8,
                                     eval_metric="logloss", random_state=rs),
        }

    def _score(self, model, X, y):
        proba = model.predict_proba(X)[:, 1]
        return {
            "roc_auc": float(roc_auc_score(y, proba)),
            "pr_auc": float(average_precision_score(y, proba)),
            "f1": float(f1_score(y, (proba >= 0.5).astype(int), zero_division=0)),
        }


class RegressorSuite(BaseModelSuite):
    """Predict log(1 + spend) in the next window."""
    primary_metric = "rmse"
    higher_is_better = False

    def _candidates(self):
        rs = self.random_state
        return {
            "ridge_l2": _linear(Ridge(alpha=1.0)),
            "lasso_l1": _linear(Lasso(alpha=0.01, max_iter=5000)),
            "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                   n_jobs=-1, random_state=rs),
            "xgboost": XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, random_state=rs),
        }

    def _score(self, model, X, y):
        pred = model.predict(X)
        return {
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)),
        }
