"""Item-item collaborative filtering and a popularity baseline (both vectorised)."""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from .exceptions import DataValidationError, NotFittedError


class BaseRecommender(ABC):
    batch_size = 1000

    def __init__(self):
        self.matrix: sparse.csr_matrix | None = None
        self.items: np.ndarray | None = None
        self.user_index: pd.Index | None = None

    def fit(self, history: pd.DataFrame):
        if history.empty:
            raise DataValidationError("Cannot fit a recommender on empty history.")
        users = np.sort(history["customer_id"].unique())
        self.items = np.sort(history["stock_code"].unique())
        self.user_index = pd.Index(users)
        rows = self.user_index.get_indexer(history["customer_id"])
        cols = pd.Index(self.items).get_indexer(history["stock_code"])
        mat = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                                shape=(len(users), len(self.items)))
        mat.data[:] = 1.0  # binary: bought / not bought
        self.matrix = mat
        self._fit_scores()
        return self

    def _fit_scores(self):
        """Hook for subclasses that precompute something after the matrix is built."""

    @abstractmethod
    def _score(self, user_rows: sparse.csr_matrix) -> np.ndarray:
        """Return a (n_users, n_items) score array."""

    def recommend(self, customer_ids, k: int = 10) -> dict[int, list[str]]:
        if self.matrix is None:
            raise NotFittedError("Call fit() before recommend().")
        ids = np.asarray(list(customer_ids))
        rows = self.user_index.get_indexer(ids)
        keep = rows >= 0
        ids, rows = ids[keep], rows[keep]
        k = min(k, len(self.items))
        out: dict[int, list[str]] = {}
        for start in range(0, len(rows), self.batch_size):
            sub = self.matrix[rows[start:start + self.batch_size]]
            scores = np.array(self._score(sub), dtype=np.float32)
            scores[sub.toarray() > 0] = -np.inf  # never recommend what they already bought
            top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            top_scores = np.take_along_axis(scores, top, axis=1)
            order = np.argsort(-top_scores, axis=1)
            top = np.take_along_axis(top, order, axis=1)
            top_scores = np.take_along_axis(top_scores, order, axis=1)
            for cid, idx, sc in zip(ids[start:start + self.batch_size], top, top_scores):
                out[int(cid)] = self.items[idx[np.isfinite(sc)]].tolist()  # drop already-bought
        return out


class PopularityRecommender(BaseRecommender):
    def _fit_scores(self):
        self._popularity = np.asarray(self.matrix.sum(axis=0)).ravel()

    def _score(self, user_rows):
        return np.broadcast_to(self._popularity, (user_rows.shape[0], len(self.items)))


class ItemItemRecommender(BaseRecommender):
    """Score = sum of cosine similarities between a customer's items and each candidate."""

    def _fit_scores(self):
        normed = normalize(self.matrix, axis=0)  # L2-normalise item columns
        sim = (normed.T @ normed).toarray().astype(np.float32)
        np.fill_diagonal(sim, 0.0)
        self._sim = sim

    def _score(self, user_rows):
        return user_rows @ self._sim


def evaluate_recommender(model: BaseRecommender, history: pd.DataFrame,
                         future: pd.DataFrame, k: int = 10) -> dict[str, float]:
    """Hit-rate@k and recall@k on NEW items bought in the future window."""
    seen = history.groupby("customer_id")["stock_code"].agg(set)
    bought = future.groupby("customer_id")["stock_code"].agg(set)
    targets = {}
    for cid in bought.index:
        if cid in seen.index:
            new_items = bought[cid] - seen[cid]
            if new_items:
                targets[int(cid)] = new_items
    if not targets:
        raise DataValidationError("No customers with new items in the future window.")
    recs = model.recommend(list(targets), k)
    hits, recalls = [], []
    for cid, wanted in targets.items():
        got = set(recs.get(cid, [])) & wanted
        hits.append(1.0 if got else 0.0)
        recalls.append(len(got) / len(wanted))
    return {f"hit_rate@{k}": float(np.mean(hits)),
            f"recall@{k}": float(np.mean(recalls)),
            "n_customers_evaluated": len(targets)}
