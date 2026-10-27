"""Build API artifacts from the public UCI Online Retail II dataset during cloud deployment.

This is intended for a deployment build step, not for every API startup. The source
dataset is CC BY 4.0; see data/README.md for attribution. The downloaded CSV is
deleted after training when this script created it, and artifacts remain in the
built service image (they are intentionally ignored by Git).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "online_retail_II.csv"
CANONICAL_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
]
COLUMN_ALIASES = {
    "InvoiceNo": ("InvoiceNo", "Invoice", "invoice"),
    "StockCode": ("StockCode", "stock_code"),
    "Description": ("Description", "description"),
    "Quantity": ("Quantity", "quantity"),
    "InvoiceDate": ("InvoiceDate", "invoice_date"),
    "UnitPrice": ("UnitPrice", "Price", "price"),
    "CustomerID": ("CustomerID", "Customer ID", "customer_id"),
    "Country": ("Country", "country"),
}


def normalize_source_columns(raw):
    """Accept common UCI/CSV header variants and return the loader's canonical schema."""
    selected = {}
    missing = []
    for canonical, aliases in COLUMN_ALIASES.items():
        match = next((column for column in aliases if column in raw.columns), None)
        if match is None:
            missing.append(canonical)
        else:
            selected[canonical] = match
    if missing:
        raise RuntimeError(
            "UCI dataset columns do not match the expected schema. "
            f"Missing: {missing}. Available columns: {list(raw.columns)}"
        )
    normalized = raw[[selected[column] for column in CANONICAL_COLUMNS]].copy()
    normalized.columns = CANONICAL_COLUMNS
    return normalized


def download_dataset() -> None:
    from ucimlrepo import fetch_ucirepo

    print("Fetching UCI Online Retail II (dataset ID 502; CC BY 4.0)...", flush=True)
    dataset = fetch_ucirepo(id=502)
    raw = dataset.data.original
    if raw is None or raw.empty:
        raise RuntimeError("UCI returned an empty Online Retail II dataset.")
    normalized = normalize_source_columns(raw)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(raw):,} source rows to temporary build data.", flush=True)


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    created_csv = not CSV_PATH.exists()
    try:
        if created_csv:
            download_dataset()
        else:
            print(f"Using existing dataset at {CSV_PATH}.", flush=True)

        env = os.environ.copy()
        source_path = str(ROOT / "src")
        env["PYTHONPATH"] = source_path + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        command = [
            sys.executable, "-m", "retail_ml.pipeline",
            "--data-dir", str(DATA_DIR),
            "--output-dir", str(ROOT / "reports"),
            "--artifact-dir", str(ROOT / "artifacts"),
        ]
        print("Training models and generating inference artifacts...", flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        required_artifacts = [
            "classifier.joblib", "regressor.joblib", "customer_features.csv",
            "purchases.csv", "catalog.csv",
        ]
        missing = [name for name in required_artifacts if not (ROOT / "artifacts" / name).is_file()]
        if missing:
            raise RuntimeError(f"Pipeline completed without required artifacts: {missing}")
        print("Model artifacts are ready for the API.", flush=True)
        return 0
    finally:
        if created_csv:
            CSV_PATH.unlink(missing_ok=True)
            print("Removed temporary downloaded dataset copy.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
