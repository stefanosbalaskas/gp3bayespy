import csv
import json
from pathlib import Path

from gp3bayespy import parity_counts, read_parity_manifest

ROOT = Path(__file__).resolve().parents[1]

GPB_PY09_EXPORTS = {
    "prediction_contrast",
    "prediction_exceedance_probability",
    "prediction_uncertainty_decomposition",
    "grouped_prediction_check",
    "predictive_residuals",
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


def test_gpb_py09_exports_are_promoted_exactly():
    manifest = read_parity_manifest()
    status_by_export = {row["r_export"]: row["status"] for row in manifest}
    assert all(status_by_export[name] == "implemented" for name in GPB_PY09_EXPORTS)
    assert status_by_export["backend_capabilities"] == "implemented_initial"


def test_gpb_py09_ledger_counts_are_frozen():
    counts = parity_counts()
    assert counts == {
        "implemented": 48,
        "implemented_initial": 1,
        "mapped_not_implemented": 409,
    }
    assert sum(counts.values()) == 458


def test_gpb_py09_dev_and_packaged_ledgers_match_semantically():
    with (ROOT / "dev/parity/function_map.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        dev_rows = list(csv.DictReader(handle))
    assert _normalize(dev_rows) == _normalize(read_parity_manifest())


def test_gpb_py09_reference_fixture_and_runtime_evidence_are_closed():
    fixture_path = ROOT / "dev/parity/predictive_diagnostics_reference_0.5.0.json"
    evidence_path = (
        ROOT / "dev/parity/predictive_diagnostics_backend_validation_0.1.0.dev0.json"
    )
    assert fixture_path.is_file()
    assert evidence_path.is_file()
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["ledger_promoted"] is True


def test_gpb_py09_runtime_backend_evidence_is_conservative():
    path = ROOT / "dev/parity/predictive_diagnostics_backend_validation_0.1.0.dev0.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))
    assert evidence["validated_commit"] == "c4c5221"
    assert evidence["validated_commit_sha"] == (
        "c4c5221516fc0456691b8152eeaf250985baf46d"
    )
    assert evidence["quality_gates"]["ruff"]["status"] == "pass"
    assert evidence["quality_gates"]["mypy"] == {
        "status": "pass",
        "source_files": 15,
    }
    assert evidence["quality_gates"]["pytest"] == {
        "status": "pass",
        "tests": 270,
    }
    assert evidence["quality_gates"]["build"]["status"] == "pass"
    smoke = evidence["backend_smoke"]
    assert smoke["status"] == "pass"
    assert smoke["binary_diagnostics_path"] == "pass"
    assert smoke["duration_diagnostics_path"] == "pass"
    assert smoke["automatic_decisions"] is False
    assert smoke["automatic_group_exclusion"] is False
    assert smoke["causal_variance_decomposition"] is False
    assert smoke["global_adequacy_established"] is False
    governance = evidence["governance"]
    assert governance["contrasts_make_automatic_decision"] is False
    assert governance["grouped_checks_exclude_automatically"] is False
    assert governance["uncertainty_is_causal_variance_decomposition"] is False
    assert governance["residuals_establish_global_adequacy"] is False
