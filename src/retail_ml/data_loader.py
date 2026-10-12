"""Load the raw Online Retail II file (xlsx or csv) into a normalised DataFrame."""
import logging
from pathlib import Path

import pandas as pd

from .exceptions import DataNotFoundError, DataValidationError

logger = logging.getLogger(__name__)

COLUMN_MAP = {
    "InvoiceNo": "invoice", "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "UnitPrice": "price", "Price": "price",
    "CustomerID": "customer_id", "Customer ID": "customer_id",
    "Country": "country",
}
REQUIRED = ["invoice", "stock_code", "description", "quantity",
            "invoice_date", "price", "customer_id", "country"]
SUPPORTED = (".xlsx", ".xls", ".csv")


class RetailDataLoader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def find_file(self) -> Path:
        if not self.data_dir.exists():
            raise DataNotFoundError(f"Data folder '{self.data_dir}' does not exist.")
        files = sorted(p for p in self.data_dir.iterdir() if p.suffix.lower() in SUPPORTED)
        if not files:
            raise DataNotFoundError(
                f"No .xlsx/.csv file found in '{self.data_dir}'. "
                "See data/README.md for the download link."
            )
        return files[0]

    def load(self) -> pd.DataFrame:
        path = self.find_file()
        logger.info("Loading %s", path)
        try:
            if path.suffix.lower() == ".csv":
                try:
                    raw = pd.read_csv(path)
                except UnicodeDecodeError:
                    raw = pd.read_csv(path, encoding="latin-1")
            else:
                sheets = pd.read_excel(path, sheet_name=None)
                raw = pd.concat(sheets.values(), ignore_index=True)
        except Exception as exc:  # corrupt file, missing openpyxl, etc.
            raise DataValidationError(f"Could not read '{path.name}': {exc}") from exc
        return self._normalise(raw)

    @staticmethod
    def _normalise(raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.rename(columns=COLUMN_MAP)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise DataValidationError(f"Missing required columns: {missing}")
        df = df[REQUIRED].copy()
        df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
        df = df.dropna(subset=["invoice_date"])
        # The two yearly sheets overlap in December 2010, so de-duplicate.
        return df.drop_duplicates().reset_index(drop=True)
