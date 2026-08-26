from __future__ import annotations

import inspect
import json
from pathlib import Path

import gp3bayespy as gp

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "dev/parity/posterior_predictive_reference_0.5.0.json"


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_gpb_py07_reference_fixture_is_frozen_to_r_050():
    fixture = _fixture()
    assert fixture["reference_package"] == "gp3bayes"
    assert fixture["reference_version"] == "0.5.0"
    assert fixture["reference_commit"] == "0ad32d8921c3b267d2021324c3b3568edf3ae01f"


def test_gpb_py07_reference_fixture_tracks_exact_two_exports():
    fixture = _fixture()
    assert [entry["name"] for entry in fixture["exports"]] == [
        "check_binary_posterior_predictive",
        "check_duration_posterior_predictive",
    ]
    assert fixture["ledger_promoted"] is True


def test_gpb_py07_python_signatures_keep_restricted_r_control_surface():
    expected = ["fit", "draws", "seed", "pass_probability", "review_probability"]
    assert list(inspect.signature(gp.check_binary_posterior_predictive).parameters) == expected
    assert list(inspect.signature(gp.check_duration_posterior_predictive).parameters) == expected


def test_gpb_py07_reference_fixture_freezes_governance_boundary():
    invariants = _fixture()["governance_invariants"]
    assert any("not a global declaration" in value for value in invariants)
    assert any("No unrestricted backend arguments" in value for value in invariants)
