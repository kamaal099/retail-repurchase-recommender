import pandas as pd
import pytest

from retail_ml.exceptions import DataValidationError
from retail_ml.features import (FEATURE_COLUMNS, add_targets, build_customer_features,
                                temporal_split)


def _sales():
    rows = [
        # customer 1: two orders in history, one in the future window
        ("A1", "11111", 2, 5.0, "2011-01-10", 1, "United Kingdom"),
        ("A2", "22222", 1, 10.0, "2011-02-10", 1, "United Kingdom"),
        ("A3", "11111", 1, 5.0, "2011-04-20", 1, "United Kingdom"),
        # customer 2: only history
        ("B1", "33333", 4, 2.0, "2011-01-15", 2, "France"),
        # sets the max date
        ("C1", "11111", 1, 5.0, "2011-04-30", 3, "United Kingdom"),
    ]
    df = pd.DataFrame(rows, columns=["invoice", "stock_code", "quantity", "price",
                                     "invoice_date", "customer_id", "country"])
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["revenue"] = df["quantity"] * df["price"]
    return df


def test_features_and_targets_are_leakage_free():
    sales = _sales()
    split = temporal_split(sales, horizon_days=20)
    assert split.cutoff == pd.Timestamp("2011-04-10")
    returns = sales.iloc[0:0]
    feats = add_targets(build_customer_features(split.history, returns, split.cutoff),
                        split.future)
    assert list(feats.columns[:len(FEATURE_COLUMNS)]) == FEATURE_COLUMNS
    c1 = feats.loc[1]
    assert c1["frequency"] == 2
    assert c1["monetary"] == pytest.approx(20.0)
    assert c1["recency_days"] == 59           # Feb 10 -> Apr 10
    assert c1["is_uk"] == 1
    assert c1["will_repurchase"] == 1 and c1["future_spend"] == pytest.approx(5.0)
    assert feats.loc[2, "will_repurchase"] == 0
    assert 3 not in feats.index               # customer 3 has no history


def test_empty_window_raises():
    sales = _sales()
    with pytest.raises(DataValidationError):
        temporal_split(sales, horizon_days=10_000)
