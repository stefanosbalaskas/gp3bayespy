import inspect
import json
from pathlib import Path

import gp3bayespy as gp

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "dev/parity/duration_foundation_reference_0.5.0.json"


def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_duration_fixture_is_frozen_to_r_050():
    fixture = _fixture()
    assert fixture["r_package"] == "gp3bayes"
    assert fixture["r_version"] == "0.5.0"
    assert fixture["source_file"] == "R/duration-workflow-foundation.R"
    assert fixture["runtime_capture"] is False


def test_duration_foundation_python_argument_order_is_frozen():
    fixture = _fixture()
    for function_name, expected in fixture["python_signatures"].items():
        actual = list(inspect.signature(getattr(gp, function_name)).parameters)
        assert actual == expected


def test_duration_fixture_records_parity_modes():
    modes = _fixture()["parity_modes"]
    assert modes["simulate_hierarchical_duration_data"] == (
        "stochastic_distributional_and_structural"
    )
    assert modes["prepare_hierarchical_duration_data"] == "structural_semantic"
    assert modes["specify_duration_model"] == "structural_semantic"
    assert modes["check_duration_prior_predictive"] == ("stochastic_distributional_and_structural")


def test_duration_foundation_exports_are_public():
    for name in _fixture()["python_signatures"]:
        assert hasattr(gp, name)
        assert name in gp.__all__
