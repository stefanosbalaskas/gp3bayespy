import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_reference_counts():
    metadata = json.loads((ROOT / "dev/parity/r_reference_metadata.json").read_text())
    assert metadata["reference_version"] == "0.5.0"
    assert metadata["exports"] == 458
    assert metadata["s3_registrations"] == 230
    assert metadata["r_source_files"] == 60
    assert metadata["rd_files"] == 465
    assert metadata["vignette_rmd_files"] == 59
    assert metadata["testthat_case_files"] == 54


def test_all_exports_have_source_signature_and_help():
    path = ROOT / "dev/parity/function_map.csv"
    with path.open(newline="", encoding="utf-8") as file_handle:
        rows = list(csv.DictReader(file_handle))
    assert len(rows) == 458
    # An empty raw signature is valid for an R function declared as function().
    assert all("r_signature_raw" in row for row in rows)
    assert all(row["r_source_file"] for row in rows)
    assert all(row["r_help_file"] for row in rows)
