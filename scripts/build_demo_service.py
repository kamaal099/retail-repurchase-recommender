"""Build API artifacts from the official UCI Online Retail II archive for cloud deployment.

The UCI import API currently exposes metadata for Online Retail II but no tabular
CSV URL, so ucimlrepo.fetch_ucirepo(id=502) cannot import it. This helper downloads
UCI's published ZIP archive and extracts only the workbook required by the project
loader. The temporary dataset copy is removed after model generation.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
XLSX_PATH = DATA_DIR / "online_retail_II.xlsx"
TEMP_ZIP_PATH = DATA_DIR / "online_retail_II.zip"
UCI_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
EXPECTED_WORKBOOK = "online_retail_II.xlsx"
SUPPORTED_INPUT_SUFFIXES = {".xlsx", ".xls", ".csv"}


def extract_workbook_from_archive(archive_path: Path, destination: Path) -> Path:
    """Extract only the expected workbook from a ZIP; never extract arbitrary paths."""
    try:
        with zipfile.ZipFile(archive_path) as archive:
            matches = [
                name for name in archive.namelist()
                if Path(name).name.casefold() == EXPECTED_WORKBOOK.casefold()
            ]
            if not matches:
                raise RuntimeError(
                    f"UCI archive did not contain {EXPECTED_WORKBOOK!r}. "
                    f"Archive members: {archive.namelist()[:20]}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(matches[0], "r") as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
    except zipfile.BadZipFile as exc:
        raise RuntimeError("The UCI download was not a valid ZIP archive.") from exc

    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("The UCI workbook was extracted as an empty file.")
    return destination


def download_dataset() -> Path:
    """Download and extract the official UCI ZIP archive to the expected workbook path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        print("Downloading Online Retail II archive directly from UCI...", flush=True)
        with requests.get(
            UCI_ARCHIVE_URL,
            stream=True,
            timeout=(20, 180),
            headers={"User-Agent": "retail-repurchase-recommender/1.0"},
        ) as response:
            response.raise_for_status()
            with TEMP_ZIP_PATH.open("wb") as archive_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        archive_file.write(chunk)
        if TEMP_ZIP_PATH.stat().st_size < 1024:
            raise RuntimeError("The UCI download was unexpectedly small; refusing to train.")
        extracted = extract_workbook_from_archive(TEMP_ZIP_PATH, XLSX_PATH)
        print(f"Extracted UCI workbook to {extracted}.", flush=True)
        return extracted
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not download the UCI dataset from {UCI_ARCHIVE_URL}: {exc}"
        ) from exc
    finally:
        TEMP_ZIP_PATH.unlink(missing_ok=True)


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    source_files = [
        path for path in DATA_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
    ]
    created_xlsx = not source_files

    try:
        if created_xlsx:
            download_dataset()
        else:
            print(f"Using existing local dataset: {source_files[0]}.", flush=True)

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
        missing = [
            name for name in required_artifacts
            if not (ROOT / "artifacts" / name).is_file()
        ]
        if missing:
            raise RuntimeError(f"Pipeline completed without required artifacts: {missing}")
        print("Model artifacts are ready for the API.", flush=True)
        return 0
    finally:
        if created_xlsx:
            XLSX_PATH.unlink(missing_ok=True)
            print("Removed temporary downloaded dataset workbook.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
