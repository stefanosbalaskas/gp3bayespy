import inspect
import json
from pathlib import Path

import gp3bayespy as gp

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "dev/parity/posterior_foundation_reference_0.5.0.json"


def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_posterior_fixture_is_frozen_to_r_050():
    fixture = _fixture()
    assert fixture["reference_package"] == "gp3bayes"
    assert fixture["reference_version"] == "0.5.0"
    assert fixture["runtime_captured_from_R"] is False


def test_posterior_fixture_covers_exact_tranche_exports():
    assert set(_fixture()["exports"]) == {
        "extract_posterior_draws",
        "diagnose_binary_fit",
        "summarise_binary_posterior",
        "diagnose_duration_fit",
        "summarise_duration_posterior",
    }


def test_posterior_fixture_signatures_match_python_argument_order():
    for name, entry in _fixture()["exports"].items():
        assert list(inspect.signature(getattr(gp, name)).parameters) == entry["signature"]


def test_posterior_fixture_preserves_conservative_governance():
    governance = _fixture()["governance"]
    assert governance["diagnostic_status_is_threshold_report"] is True
    assert governance["automatic_convergence_claim"] is False
    assert governance["posterior_adequacy_established"] is False
