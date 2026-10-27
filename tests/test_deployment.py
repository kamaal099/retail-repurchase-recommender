from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_demo_service.py"
SPEC = spec_from_file_location("build_demo_service", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CANONICAL_COLUMNS = MODULE.CANONICAL_COLUMNS
normalize_source_columns = MODULE.normalize_source_columns


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
