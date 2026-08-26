import csv
import json
from pathlib import Path

from gp3bayespy import parity_counts, read_parity_manifest

ROOT = Path(__file__).resolve().parents[1]

GPB_PY05_EXPORTS = {
    "extract_posterior_draws",
    "diagnose_binary_fit",
    "summarise_binary_posterior",
    "diagnose_duration_fit",
    "summarise_duration_posterior",
}


def _normalize(rows):
    return [
        {
            key: (
                value.replace("\r\n", "\n").replace("\r", "\n")
                if value
                else value
            )
            for key, value in row.items()
        }
        for row in rows
    ]


def test_gpb_py05_posterior_exports_are_promoted_exactly():
    manifest = read_parity_manifest()
    status_by_export = {row["r_export"]: row["status"] for row in manifest}
    assert all(status_by_export[name] == "implemented" for name in GPB_PY05_EXPORTS)
    assert status_by_export["backend_capabilities"] == "implemented_initial"


def test_gpb_py05_ledger_counts_are_historical_minimum():
    counts = parity_counts()
    assert counts["implemented"] >= 23
    assert counts["implemented_initial"] == 1
    assert counts["mapped_not_implemented"] <= 434
    assert sum(counts.values()) == 458


def test_gpb_py05_dev_and_packaged_ledgers_match_semantically():
    with (ROOT / "dev/parity/function_map.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        dev_rows = list(csv.DictReader(handle))
    assert _normalize(dev_rows) == _normalize(read_parity_manifest())


def test_gpb_py05_reference_fixture_and_runtime_evidence_are_present():
    fixture = ROOT / "dev/parity/posterior_foundation_reference_0.5.0.json"
    evidence = ROOT / "dev/parity/posterior_backend_validation_0.1.0.dev0.json"
    assert fixture.is_file()
    assert evidence.is_file()


def test_gpb_py05_runtime_backend_evidence_is_conservative():
    path = ROOT / "dev/parity/posterior_backend_validation_0.1.0.dev0.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))
    assert evidence["validated_commit"] == "6bf5ecc"
    assert evidence["quality_gates"]["ruff"]["status"] == "pass"
    assert evidence["quality_gates"]["mypy"] == {
        "status": "pass",
        "source_files": 13,
    }
    assert evidence["quality_gates"]["pytest"] == {
        "status": "pass",
        "tests": 158,
    }
    assert evidence["quality_gates"]["build"]["status"] == "pass"
    smoke = evidence["backend_smoke"]
    assert smoke["status"] == "pass"
    assert smoke["binary_posterior_path"] == "pass"
    assert smoke["duration_posterior_path"] == "pass"
    assert smoke["binary_diagnostic_threshold_status"] == "fail"
    assert smoke["duration_diagnostic_threshold_status"] == "fail"
    assert smoke["diagnostic_fail_is_expected_for_short_smoke"] is True
    assert smoke["diagnostic_claims_permitted"] is False
    assert smoke["convergence_claim"] is False
    assert smoke["posterior_adequacy_established"] is False
