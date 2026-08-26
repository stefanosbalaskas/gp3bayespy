import csv
import json
from pathlib import Path

from gp3bayespy import parity_counts, read_parity_manifest

ROOT = Path(__file__).resolve().parents[1]

GPB_PY08_EXPORTS = {
    "binary_prediction_scores",
    "binary_threshold_metrics",
    "binary_calibration_table",
    "duration_prediction_scores",
    "duration_quantile_calibration",
    "duration_pit_table",
    "predictive_coverage_table",
    "posterior_predictive_summary_table",
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


def test_gpb_py08_exports_are_promoted_exactly():
    manifest = read_parity_manifest()
    status_by_export = {row["r_export"]: row["status"] for row in manifest}
    assert all(status_by_export[name] == "implemented" for name in GPB_PY08_EXPORTS)
    assert status_by_export["backend_capabilities"] == "implemented_initial"


def test_gpb_py08_ledger_counts_are_frozen():
    counts = parity_counts()
    assert counts == {
        "implemented": 43,
        "implemented_initial": 1,
        "mapped_not_implemented": 414,
    }
    assert sum(counts.values()) == 458


def test_gpb_py08_dev_and_packaged_ledgers_match_semantically():
    with (ROOT / "dev/parity/function_map.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        dev_rows = list(csv.DictReader(handle))
    assert _normalize(dev_rows) == _normalize(read_parity_manifest())


def test_gpb_py08_reference_fixture_and_runtime_evidence_are_closed():
    fixture_path = ROOT / "dev/parity/predictive_scoring_reference_0.5.0.json"
    evidence_path = (
        ROOT / "dev/parity/predictive_scoring_backend_validation_0.1.0.dev0.json"
    )
    assert fixture_path.is_file()
    assert evidence_path.is_file()
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["ledger_promoted"] is True


def test_gpb_py08_runtime_backend_evidence_is_conservative():
    path = ROOT / "dev/parity/predictive_scoring_backend_validation_0.1.0.dev0.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))
    assert evidence["validated_commit"] == "cde4776"
    assert evidence["validated_commit_sha"] == (
        "cde4776c5e593ac087135190793e70382dfff0c6"
    )
    assert evidence["quality_gates"]["ruff"]["status"] == "pass"
    assert evidence["quality_gates"]["mypy"] == {
        "status": "pass",
        "source_files": 15,
    }
    assert evidence["quality_gates"]["pytest"] == {
        "status": "pass",
        "tests": 246,
    }
    assert evidence["quality_gates"]["build"]["status"] == "pass"
    smoke = evidence["backend_smoke"]
    assert smoke["status"] == "pass"
    assert smoke["binary_scoring_path"] == "pass"
    assert smoke["duration_scoring_path"] == "pass"
    assert smoke["automatic_decisions"] is False
    assert smoke["global_adequacy_established"] is False
    governance = evidence["governance"]
    assert governance["threshold_metrics_make_automatic_decision"] is False
    assert governance["calibration_establishes_global_adequacy"] is False
    assert governance["coverage_establishes_global_adequacy"] is False
