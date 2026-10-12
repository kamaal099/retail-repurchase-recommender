import pandas as pd
import pytest

from retail_ml.cleaning import clean_transactions
from retail_ml.exceptions import DataValidationError


def _frame(rows):
    cols = ["invoice", "stock_code", "description", "quantity", "invoice_date",
            "price", "customer_id", "country"]
    return pd.DataFrame(rows, columns=cols)


def test_splits_sales_and_returns_and_drops_junk():
    d = pd.Timestamp("2011-01-01")
    df = _frame([
        ["10001", "85123A", "MUG", 2, d, 3.0, 17850, "United Kingdom"],   # valid sale
        ["C10002", "85123A", "MUG", -1, d, 3.0, 17850, "United Kingdom"],  # return
        ["10003", "POST", "POSTAGE", 1, d, 18.0, 17850, "United Kingdom"],  # non-product
        ["10004", "21730", "LAMP", 1, d, 5.0, None, "United Kingdom"],      # no customer
        ["10005", "21730", "LAMP", -3, d, 5.0, 12345, "France"],            # negative qty
        ["10006", "21730", "LAMP", 1, d, 0.0, 12345, "France"],             # zero price
    ])
    sales, returns = clean_transactions(df)
    assert len(sales) == 1 and sales.loc[0, "revenue"] == pytest.approx(6.0)
    assert len(returns) == 1 and returns.loc[0, "invoice"] == "C10002"


def test_all_rows_invalid_raises():
    df = _frame([["10003", "POST", "POSTAGE", 1, pd.Timestamp("2011-01-01"), 1.0, 1, "UK"]])
    with pytest.raises(DataValidationError):
        clean_transactions(df)
