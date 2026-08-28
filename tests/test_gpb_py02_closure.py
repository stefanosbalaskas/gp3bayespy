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
}


def _normalize(rows):
    return [
        {
            key: (value.replace("\r\n", "\n").replace("\r", "\n") if value else value)
            for key, value in row.items()
        }
        for row in rows
    ]


def test_gpb_py02_promotions_remain_truthful():
    manifest = read_parity_manifest()
    status_by_export = {row["r_export"]: row["status"] for row in manifest}
    for name in IMPLEMENTED:
        assert status_by_export[name] == "implemented"
    assert status_by_export["backend_capabilities"] in {"implemented_initial", "implemented"}


def test_gpb_py02_ledger_total_remains_frozen():
    assert sum(parity_counts().values()) == 458


def test_gpb_py02_dev_and_packaged_ledgers_match_semantically():
    with (ROOT / "dev/parity/function_map.csv").open(newline="", encoding="utf-8") as handle:
        dev_rows = list(csv.DictReader(handle))
    assert _normalize(dev_rows) == _normalize(read_parity_manifest())


def test_gpb_py02_binary_fixture_is_present():
    fixture = ROOT / "dev/parity/binary_foundation_reference_0.5.0.json"
    assert fixture.is_file()
