import inspect
import json
from pathlib import Path

import gp3bayespy as gp

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "dev/parity/advanced_predictive_diagnostics_reference_0.5.0.json"

EXPECTED = [
    "binary_calibration_error",
    "binary_confusion_table",
    "binary_group_calibration",
    "binary_precision_recall_curve",
    "binary_roc_curve",
    "duration_qq_table",
    "duration_tail_check",
    "group_prediction_summary",
    "posterior_predictive_statistic",
    "ppc_statistic_table",
    "prediction_draws_long",
    "prediction_interval_width",
    "prediction_pairwise_contrasts",
    "prediction_rank_probabilities",
]


def test_gpb_py10_reference_fixture_tracks_exact_advanced_source_family():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fixture["reference_version"] == "0.5.0"
    assert fixture["reference_commit"] == "0ad32d8921c3b267d2021324c3b3568edf3ae01f"
    assert fixture["source_file"] == "R/predictive-diagnostics-advanced.R"
    assert fixture["ledger_promoted"] is False
    assert sorted(entry["name"] for entry in fixture["exports"]) == EXPECTED


def test_gpb_py10_all_frozen_exports_are_public_and_documented():
    api = (ROOT / "docs/api.md").read_text(encoding="utf-8")
    for name in EXPECTED:
        assert hasattr(gp, name)
        assert callable(getattr(gp, name))
        assert f"gp3bayespy.{name}" in api


def test_gpb_py10_python_signatures_keep_restricted_control_surfaces():
    expected = {
        "prediction_draws_long": ["x", "max_draws", "seed"],
        "posterior_predictive_statistic": ["x", "statistic", "threshold"],
        "ppc_statistic_table": ["x"],
        "binary_confusion_table": ["x", "observed", "threshold"],
        "binary_roc_curve": ["x", "observed", "thresholds"],
        "binary_precision_recall_curve": ["x", "observed", "thresholds"],
        "binary_calibration_error": ["x", "observed", "bins"],
        "binary_group_calibration": ["x", "group"],
        "duration_qq_table": ["x", "probs"],
        "duration_tail_check": ["x", "threshold"],
        "group_prediction_summary": ["x", "by", "probs"],
        "prediction_pairwise_contrasts": ["x", "rows", "measure", "max_rows", "probs"],
        "prediction_interval_width": ["x"],
        "prediction_rank_probabilities": ["x", "rows", "direction", "max_rows"],
    }
    for name, parameters in expected.items():
        assert list(inspect.signature(getattr(gp, name)).parameters) == parameters


def test_gpb_py10_governance_fixture_is_explicitly_conservative():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    text = " ".join(fixture["governance_invariants"]).lower()
    assert "never automatic adequacy" in text
    assert "never make automatic selections" in text
    assert "quantile(type = 8)" in " ".join(fixture["governance_invariants"])
