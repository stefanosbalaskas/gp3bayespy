import inspect
import json
from pathlib import Path

import gp3bayespy as gp

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "dev/parity/predictive_scoring_reference_0.5.0.json"

EXPORTS = {
    "binary_prediction_scores",
    "binary_threshold_metrics",
    "binary_calibration_table",
    "duration_prediction_scores",
    "duration_quantile_calibration",
    "duration_pit_table",
    "predictive_coverage_table",
    "posterior_predictive_summary_table",
}


def test_predictive_scoring_reference_fixture_accounts_for_exact_export_set():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert {entry["name"] for entry in payload["exports"]} == EXPORTS
    assert payload["reference_version"] == "0.5.0"
    assert payload["reference_commit"] == "0ad32d8921c3b267d2021324c3b3568edf3ae01f"
    assert payload["ledger_promoted"] is True


def test_predictive_scoring_exports_are_public_and_callable():
    for name in EXPORTS:
        assert callable(getattr(gp, name))


def test_predictive_scoring_public_signatures_expose_no_backend_escape_hatches():
    forbidden = {"formula", "family", "backend", "algorithm", "prior", "stanvars"}
    for name in EXPORTS:
        parameters = set(inspect.signature(getattr(gp, name)).parameters)
        assert parameters.isdisjoint(forbidden)
