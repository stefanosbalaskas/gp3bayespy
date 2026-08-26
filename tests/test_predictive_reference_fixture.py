import inspect
import json
from pathlib import Path

import gp3bayespy as gp

ROOT = Path(__file__).resolve().parents[1]

EXPORTS = [
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
]

EXPECTED_PARAMETER_ORDER = {
    "create_prediction_grid": ["x", "variables", "at", "numeric_at", "max_rows"],
    "audit_prediction_support": ["fit", "newdata"],
    "prediction_support_table": ["x"],
    "predict_model": [
        "fit",
        "newdata",
        "type",
        "include_group_effects",
        "allow_new_levels",
        "ndraws",
        "probs",
        "seed",
    ],
    "prediction_table": ["x"],
    "extract_expected_predictions": [
        "fit",
        "newdata",
        "include_group_effects",
        "allow_new_levels",
        "ndraws",
    ],
    "extract_posterior_predictions": [
        "fit",
        "newdata",
        "include_group_effects",
        "allow_new_levels",
        "ndraws",
        "seed",
    ],
    "extract_linear_predictions": [
        "fit",
        "newdata",
        "include_group_effects",
        "allow_new_levels",
        "ndraws",
    ],
    "predict_binary_probability": [
        "fit",
        "newdata",
        "include_group_effects",
        "allow_new_levels",
        "ndraws",
        "probs",
    ],
    "predict_duration": [
        "fit",
        "newdata",
        "type",
        "include_group_effects",
        "allow_new_levels",
        "ndraws",
        "probs",
        "seed",
    ],
}


def _fixture():
    path = ROOT / "dev/parity/predictive_foundation_reference_0.5.0.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_predictive_fixture_is_frozen_to_r_050():
    fixture = _fixture()
    assert fixture["reference_version"] == "0.5.0"
    assert fixture["reference_commit"] == "0ad32d8921c3b267d2021324c3b3568edf3ae01f"
    assert fixture["source_file"] == "R/prediction-support.R"


def test_predictive_fixture_covers_exact_first_tranche():
    assert [entry["name"] for entry in _fixture()["exports"]] == EXPORTS


def test_predictive_python_parameter_order_matches_r_public_contract():
    for name in EXPORTS:
        observed = list(inspect.signature(getattr(gp, name)).parameters)
        assert observed == EXPECTED_PARAMETER_ORDER[name]


def test_predictive_ledger_is_not_promoted_before_runtime_gate():
    counts = gp.parity_counts()
    assert counts == {
        "implemented": 23,
        "implemented_initial": 1,
        "mapped_not_implemented": 434,
    }
    manifest = {row["r_export"]: row["status"] for row in gp.read_parity_manifest()}
    assert all(manifest[name] == "mapped_not_implemented" for name in EXPORTS)


def test_predictive_governance_fixture_prohibits_automatic_claims():
    invariants = " ".join(_fixture()["governance_invariants"]).lower()
    assert "do not establish causal effects" in invariants
    assert "do not remove or reject rows automatically" in invariants
    assert "out-of-sample adequacy" in invariants
