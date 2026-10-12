"""Model suites: several candidate models per task, compared on a held-out set."""
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
    # All features are non-negative and right-skewed, so log1p then scale.
    return make_pipeline(FunctionTransformer(np.log1p), StandardScaler(), estimator)


class BaseModelSuite(ABC):
    primary_metric: str
    higher_is_better: bool

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: dict = {}
        self.metrics: dict[str, dict[str, float]] = {}
        self.best_name: str | None = None

    @abstractmethod
    def _candidates(self) -> dict: ...

    @abstractmethod
    def _score(self, model, X, y) -> dict[str, float]: ...

    def fit_evaluate(self, X_train, y_train, X_test, y_test) -> dict[str, dict[str, float]]:
        for name, model in self._candidates().items():
            try:
                model.fit(X_train, y_train)
            except Exception as exc:
                raise ModelTrainingError(f"{name} failed to train: {exc}") from exc
            self.models[name] = model
            self.metrics[name] = self._score(model, X_test, y_test)
        pick = max if self.higher_is_better else min
        self.best_name = pick(self.metrics, key=lambda n: self.metrics[n][self.primary_metric])
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
            "f1": float(f1_score(y, (proba >= 0.5).astype(int))),
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
