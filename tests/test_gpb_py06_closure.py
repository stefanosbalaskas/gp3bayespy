import csv
import json
from pathlib import Path

from gp3bayespy import parity_counts, read_parity_manifest

ROOT = Path(__file__).resolve().parents[1]

GPB_PY06_EXPORTS = {
    "create_prediction_grid",
    "audit_prediction_support",
    "prediction_support_table",
    "predict_model",
    "prediction_table",
    "extract_expected_predictions",
    "extract_posterior_predictions",
    "extract_linear_predictions",
    "predict_binary_probability",
    "predict_duration",
}


def _normalize(rows):
    return [
        {
            key: (value.replace("\r\n", "\n").replace("\r", "\n") if value else value)
            for key, value in row.items()
        }
        for row in rows
    ]


def test_gpb_py06_predictive_exports_are_promoted_exactly():
    manifest = read_parity_manifest()
    status_by_export = {row["r_export"]: row["status"] for row in manifest}
    assert all(status_by_export[name] == "implemented" for name in GPB_PY06_EXPORTS)
    assert status_by_export["backend_capabilities"] in {"implemented_initial", "implemented"}


def test_gpb_py06_ledger_still_covers_all_exports():
    counts = parity_counts()
    assert counts["implemented"] >= 33
    assert counts["implemented_initial"] in {0, 1}
    assert sum(counts.values()) == 458


def test_gpb_py06_dev_and_packaged_ledgers_match_semantically():
    with (ROOT / "dev/parity/function_map.csv").open(newline="", encoding="utf-8") as handle:
        dev_rows = list(csv.DictReader(handle))
    assert _normalize(dev_rows) == _normalize(read_parity_manifest())


def test_gpb_py06_reference_fixture_and_runtime_evidence_are_closed():
    fixture_path = ROOT / "dev/parity/predictive_foundation_reference_0.5.0.json"
    evidence_path = ROOT / "dev/parity/predictive_backend_validation_0.1.0.dev0.json"
    assert fixture_path.is_file()
    assert evidence_path.is_file()
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["ledger_promoted"] is True


def test_gpb_py06_runtime_backend_evidence_is_governed():
    path = ROOT / "dev/parity/predictive_backend_validation_0.1.0.dev0.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))
    assert evidence["validated_commit"] == "a8feb83"
    assert evidence["validated_commit_sha"] == ("a8feb83cb2aab15b91190569fd79373fcd0a809a")
    assert evidence["quality_gates"]["ruff"]["status"] == "pass"
    assert evidence["quality_gates"]["mypy"] == {
        "status": "pass",
        "source_files": 14,
    }
    assert evidence["quality_gates"]["pytest"] == {
        "status": "pass",
        "tests": 192,
    }
    assert evidence["quality_gates"]["build"]["status"] == "pass"
    smoke = evidence["backend_smoke"]
    assert smoke["status"] == "pass"
    assert smoke["binary_prediction_path"] == "pass"
    assert smoke["duration_prediction_path"] == "pass"
    assert smoke["binary_prediction_shape"] == [25, 2]
    assert smoke["duration_prediction_shape"] == [25, 2]
    assert smoke["automatic_support_rejection"] is False
    assert smoke["causal_claim"] is False
    assert smoke["out_of_sample_adequacy_claim"] is False
