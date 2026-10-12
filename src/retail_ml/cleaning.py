"""Transaction cleaning: separate sales from returns and drop non-product rows."""
import pandas as pd

from .exceptions import DataValidationError

PRODUCT_CODE_PATTERN = r"^\d{5}"  # real products start with a 5-digit code


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sales, returns). Sales have positive quantity and price plus a revenue column."""
    df = df.copy()
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce")
    df = df.dropna(subset=["customer_id"])
    df["customer_id"] = df["customer_id"].astype("int64")
    df["invoice"] = df["invoice"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str).str.upper().str.strip()
    df = df[df["stock_code"].str.match(PRODUCT_CODE_PATTERN)]

    is_cancel = df["invoice"].str.upper().str.startswith("C")
    returns = df[is_cancel].copy()
    sales = df[~is_cancel & (df["quantity"] > 0) & (df["price"] > 0)].copy()
    if sales.empty:
        raise DataValidationError("No valid sales rows left after cleaning.")
    sales["revenue"] = sales["quantity"] * sales["price"]
    return sales.reset_index(drop=True), returns.reset_index(drop=True)
