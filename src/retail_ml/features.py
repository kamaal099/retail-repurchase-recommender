"""Leakage-free customer features built only from data before a cutoff date."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .exceptions import DataValidationError

FEATURE_COLUMNS = [
    "recency_days", "frequency", "monetary", "avg_order_value", "tenure_days",
    "unique_products", "avg_basket_items", "avg_basket_lines", "mean_gap_days",
    "orders_last_90d", "revenue_last_90d", "return_rate", "is_uk",
]


@dataclass(frozen=True)
class TemporalSplit:
    cutoff: pd.Timestamp
    history: pd.DataFrame
    future: pd.DataFrame


def temporal_split(sales: pd.DataFrame, horizon_days: int) -> TemporalSplit:
    """Use exactly horizon_days calendar dates ending on the last transaction date."""
    if sales.empty:
        raise DataValidationError("Cannot split an empty sales table.")
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, (int, np.integer)) or horizon_days < 1:
        raise DataValidationError("horizon_days must be a positive integer.")
    max_date = pd.to_datetime(sales["invoice_date"], errors="coerce").max()
    if pd.isna(max_date):
        raise DataValidationError("Sales must contain valid invoice_date values.")
    cutoff = max_date.normalize() - pd.Timedelta(days=int(horizon_days) - 1)
    dates = pd.to_datetime(sales["invoice_date"], errors="coerce")
    history = sales[dates < cutoff]
    future = sales[dates >= cutoff]
    if history.empty or future.empty:
        raise DataValidationError("Cutoff leaves history or future window empty.")
    return TemporalSplit(cutoff, history, future)


def build_customer_features(history: pd.DataFrame, returns: pd.DataFrame,
                            cutoff: pd.Timestamp) -> pd.DataFrame:
    """One row per customer with RFM-style features. All vectorised groupby operations."""
    inv = (history.groupby(["customer_id", "invoice"])
           .agg(order_date=("invoice_date", "min"),
                order_value=("revenue", "sum"),
                n_items=("quantity", "sum"),
                n_lines=("stock_code", "nunique"))
           .reset_index()
           .sort_values(["customer_id", "order_date"]))
    inv["gap_days"] = inv.groupby("customer_id")["order_date"].diff().dt.total_seconds() / 86400

    cust = inv.groupby("customer_id").agg(
        frequency=("invoice", "count"),
        monetary=("order_value", "sum"),
        avg_order_value=("order_value", "mean"),
        first_order=("order_date", "min"),
        last_order=("order_date", "max"),
        avg_basket_items=("n_items", "mean"),
        avg_basket_lines=("n_lines", "mean"),
        mean_gap_days=("gap_days", "mean"),
    )
    cust["recency_days"] = (cutoff - cust["last_order"]).dt.days
    cust["tenure_days"] = (cutoff - cust["first_order"]).dt.days
    cust["mean_gap_days"] = cust["mean_gap_days"].fillna(0)
    cust["unique_products"] = history.groupby("customer_id")["stock_code"].nunique()

    recent = (inv[inv["order_date"] >= cutoff - pd.Timedelta(days=90)]
              .groupby("customer_id")
              .agg(orders_last_90d=("invoice", "count"),
                   revenue_last_90d=("order_value", "sum")))
    cust = cust.join(recent)
    cust[["orders_last_90d", "revenue_last_90d"]] = (
        cust[["orders_last_90d", "revenue_last_90d"]].fillna(0))

    # This is a return-line share proxy, not a matched-item return probability:
    # cancellation lines cannot reliably be paired to their original sales lines.
    returned = returns[returns["invoice_date"] < cutoff].groupby("customer_id").size()
    sold_lines = history.groupby("customer_id").size()
    total_lines = sold_lines.add(returned, fill_value=0)
    cust["return_rate"] = (
        returned.reindex(cust.index).fillna(0) /
        total_lines.reindex(cust.index).replace(0, np.nan)
    ).fillna(0).clip(0, 1)

    is_uk = history.groupby("customer_id")["country"].last() == "United Kingdom"
    cust["is_uk"] = is_uk.reindex(cust.index).astype(int)
    return cust[FEATURE_COLUMNS]


def add_targets(features: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    """Add future_spend (regression) and will_repurchase (classification) targets."""
    spend = future.groupby("customer_id")["revenue"].sum()
    out = features.copy()
    out["future_spend"] = spend.reindex(out.index).fillna(0.0)
    out["will_repurchase"] = (out["future_spend"] > 0).astype(int)
    return out


def log_target(spend: pd.Series) -> pd.Series:
    return np.log1p(spend)
