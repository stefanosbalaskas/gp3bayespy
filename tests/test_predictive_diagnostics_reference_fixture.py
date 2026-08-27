import inspect
import json
from pathlib import Path

import gp3bayespy as gp

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "dev/parity/predictive_diagnostics_reference_0.5.0.json"
EXPORTS = [
    "prediction_contrast",
    "prediction_exceedance_probability",
    "prediction_uncertainty_decomposition",
    "grouped_prediction_check",
    "predictive_residuals",
]


def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_gpb_py09_reference_is_frozen_to_gp3bayes_050():
    fixture = _fixture()
    assert fixture["reference_version"] == "0.5.0"
    assert fixture["reference_commit"] == "0ad32d8921c3b267d2021324c3b3568edf3ae01f"
    assert fixture["source_file"] == "R/prediction-support.R"


def test_gpb_py09_reference_tracks_exact_five_exports():
    fixture = _fixture()
    assert [row["name"] for row in fixture["exports"]] == EXPORTS
    assert fixture["ledger_promoted"] is True


def test_gpb_py09_python_signatures_match_restricted_control_surfaces():
    expected = {
        "prediction_contrast": ["x", "row1", "row2", "measure", "probs"],
        "prediction_exceedance_probability": ["x", "threshold", "direction"],
        "prediction_uncertainty_decomposition": [
            "fit",
            "newdata",
            "include_group_effects",
            "allow_new_levels",
            "ndraws",
            "seed",
        ],
        "grouped_prediction_check": ["fit", "group", "ndraws", "probs", "seed"],
        "predictive_residuals": ["fit", "type", "ndraws"],
    }
    for name, parameters in expected.items():
        assert list(inspect.signature(getattr(gp, name)).parameters) == parameters


def test_gpb_py09_governance_invariants_remain_conservative():
    invariants = " ".join(_fixture()["governance_invariants"]).lower()
    assert "never make automatic decisions" in invariants
    assert "not a causal variance decomposition" in invariants
    assert "never exclude groups automatically" in invariants
    assert "do not establish model adequacy" in invariants
    assert "no unrestricted backend" in invariants
