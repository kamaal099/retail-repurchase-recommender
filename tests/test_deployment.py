import pandas as pd
import pytest

from scripts.build_demo_service import CANONICAL_COLUMNS, normalize_source_columns


def test_normalize_source_columns_accepts_common_uci_headers():
    raw = pd.DataFrame([{
        "InvoiceNo": "536365",
        "StockCode": "85123A",
        "Description": "WHITE HANGING HEART T-LIGHT HOLDER",
        "Quantity": 6,
        "InvoiceDate": "2010-12-01 08:26:00",
        "Price": 2.55,
        "Customer ID": 17850,
        "Country": "United Kingdom",
    }])

    normalized = normalize_source_columns(raw)

    assert list(normalized.columns) == CANONICAL_COLUMNS
    assert normalized.loc[0, "UnitPrice"] == 2.55
    assert normalized.loc[0, "CustomerID"] == 17850


def test_normalize_source_columns_rejects_unexpected_schema():
    with pytest.raises(RuntimeError, match="do not match the expected schema"):
        normalize_source_columns(pd.DataFrame({"InvoiceNo": ["1"]}))
