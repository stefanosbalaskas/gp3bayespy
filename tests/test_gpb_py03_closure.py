import csv
from pathlib import Path

from gp3bayespy import parity_counts, read_parity_manifest

ROOT = Path(__file__).resolve().parents[1]

IMPLEMENTED = {
    "create_model_contract",
    "audit_model_readiness",
    "build_model_formula",
    "create_prior_specification",
    "validate_prior_specification",
    "create_model_specification",
    "simulate_hierarchical_binary_data",
    "prepare_hierarchical_binary_data",
    "specify_binary_model",
    "check_binary_prior_predictive",
    "simulate_hierarchical_duration_data",
    "prepare_hierarchical_duration_data",
    "specify_duration_model",
    "check_duration_prior_predictive",
}


def _normalize(rows):
    return [
        {
            key: (value.replace("\r\n", "\n").replace("\r", "\n") if value else value)
            for key, value in row.items()
        }
        for row in rows
    ]


def test_gpb_py03_promoted_exports_remain_implemented():
    manifest = read_parity_manifest()
    status_by_export = {row["r_export"]: row["status"] for row in manifest}
    implemented = {name for name, status in status_by_export.items() if status == "implemented"}
    assert implemented >= IMPLEMENTED


def test_gpb_py03_ledger_still_covers_all_exports():
    counts = parity_counts()
    assert counts["implemented"] >= 14
    assert sum(counts.values()) == 458


def test_gpb_py03_dev_and_packaged_ledgers_match_semantically():
    with (ROOT / "dev/parity/function_map.csv").open(newline="", encoding="utf-8") as handle:
        dev_rows = list(csv.DictReader(handle))
    assert _normalize(dev_rows) == _normalize(read_parity_manifest())


def test_gpb_py03_duration_fixture_is_present():
    fixture = ROOT / "dev/parity/duration_foundation_reference_0.5.0.json"
    assert fixture.is_file()
