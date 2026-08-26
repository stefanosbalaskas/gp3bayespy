import inspect
import json
import tomllib
from pathlib import Path

import gp3bayespy as gp

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "dev/parity/fitting_foundation_reference_0.5.0.json"


def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fitting_fixture_is_pinned_to_frozen_r_reference():
    fixture = _fixture()
    assert fixture["reference_package"] == "gp3bayes"
    assert fixture["reference_version"] == "0.5.0"
    assert fixture["reference_commit"] == "0ad32d8921c3b267d2021324c3b3568edf3ae01f"
    assert "not an R-runtime snapshot" in fixture["fixture_kind"]


def test_fitting_fixture_accounts_for_exact_four_r_exports():
    assert set(_fixture()["functions"]) == {
        "translate_binary_model_to_brms",
        "fit_binary_model",
        "translate_duration_model_to_brms",
        "fit_duration_model",
    }


def test_fitting_python_parameter_order_matches_r_public_control_order():
    expected = [
        "specification",
        "chains",
        "iter",
        "warmup",
        "cores",
        "seed",
        "adapt_delta",
        "max_treedepth",
        "refresh",
    ]
    assert list(inspect.signature(gp.fit_binary_model).parameters) == expected
    assert list(inspect.signature(gp.fit_duration_model).parameters) == expected
    assert list(inspect.signature(gp.translate_binary_model_to_brms).parameters) == [
        "specification"
    ]
    assert list(inspect.signature(gp.translate_duration_model_to_brms).parameters) == [
        "specification"
    ]


def test_backend_divergence_is_explicit_not_silent():
    adaptation = _fixture()["backend_adaptation"]
    assert adaptation["r_interface"] == "brms"
    assert adaptation["r_sampling_backend"] == "rstan"
    assert adaptation["python_interface"] == "pymc"
    assert adaptation["python_algorithm"] == "NUTS"


def test_bayesian_extra_pins_compatible_pymc_arviz_generations():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    bayes = set(project["project"]["optional-dependencies"]["bayes"])
    assert "arviz>=0.18,<1; python_version < '3.12'" in bayes
    assert "pymc>=5.10,<6; python_version < '3.12'" in bayes
    assert "arviz>=1.0; python_version >= '3.12'" in bayes
    assert "pymc>=6.3.1; python_version >= '3.12'" in bayes
