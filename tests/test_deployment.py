import importlib.util
import zipfile
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_demo_service.py"
SPEC = importlib.util.spec_from_file_location("build_demo_service", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

extract_workbook_from_archive = MODULE.extract_workbook_from_archive


def test_extract_workbook_from_uci_archive(tmp_path):
    archive_path = tmp_path / "online_retail_II.zip"
    destination = tmp_path / "data" / "online_retail_II.xlsx"
    workbook_bytes = b"fake-xlsx-payload-for-extraction-test"

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("online_retail_II.xlsx", workbook_bytes)
        archive.writestr("unrelated.txt", "ignore me")

    result = extract_workbook_from_archive(archive_path, destination)

    assert result == destination
    assert destination.read_bytes() == workbook_bytes


def test_extract_workbook_from_nested_archive(tmp_path):
    archive_path = tmp_path / "uci.zip"
    destination = tmp_path / "online_retail_II.xlsx"
    workbook_bytes = b"nested-workbook"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("datasets/online_retail_II.xlsx", workbook_bytes)

    extract_workbook_from_archive(archive_path, destination)

    assert destination.read_bytes() == workbook_bytes


def test_reject_archive_without_expected_workbook(tmp_path):
    archive_path = tmp_path / "wrong.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("other.xlsx", b"not the expected workbook")

    with pytest.raises(RuntimeError, match="did not contain"):
        extract_workbook_from_archive(archive_path, tmp_path / "out.xlsx")


def test_reject_invalid_archive(tmp_path):
    archive_path = tmp_path / "invalid.zip"
    archive_path.write_text("not a zip")

    with pytest.raises(RuntimeError, match="not a valid ZIP"):
        extract_workbook_from_archive(archive_path, tmp_path / "out.xlsx")
