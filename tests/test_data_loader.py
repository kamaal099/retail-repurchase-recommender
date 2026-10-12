import pandas as pd
import pytest

from retail_ml.data_loader import RetailDataLoader
from retail_ml.exceptions import DataNotFoundError, DataValidationError


def test_missing_folder_and_empty_folder(tmp_path):
    with pytest.raises(DataNotFoundError):
        RetailDataLoader(tmp_path / "nope").load()
    with pytest.raises(DataNotFoundError):
        RetailDataLoader(tmp_path).load()


def test_loads_uci_column_names_and_dedupes(tmp_path):
    csv = tmp_path / "sample.csv"
    row = {"Invoice": "10001", "StockCode": "85123A", "Description": "MUG", "Quantity": 1,
           "InvoiceDate": "2011-01-01 10:00:00", "Price": 2.5, "Customer ID": 17850.0,
           "Country": "United Kingdom"}
    pd.DataFrame([row, row]).to_csv(csv, index=False)
    df = RetailDataLoader(tmp_path).load()
    assert len(df) == 1 and "customer_id" in df.columns


def test_missing_columns_raise(tmp_path):
    pd.DataFrame({"foo": [1]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(DataValidationError):
        RetailDataLoader(tmp_path).load()
