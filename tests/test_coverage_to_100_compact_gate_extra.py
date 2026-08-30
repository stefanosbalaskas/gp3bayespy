from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
import gp3bayespy.fitting as fitting
import gp3bayespy.posterior_validation_core as pvc
import gp3bayespy.specification as specmod
import gp3bayespy.unified_workflow_api as unified
from gp3bayespy.exceptions import BackendUnavailableError, GP3BayesError

matplotlib.use("Agg")
matplotlib.rcParams["figure.max_open_warning"] = 0


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_fitting_validation_optional_backend_and_version_branches(monkeypatch):
    with pytest.raises(GP3BayesError):
        fitting._numeric_scalar(True, "x")
    with pytest.raises(GP3BayesError):
        fitting._numeric_scalar(float("inf"), "x")
    with pytest.raises(GP3BayesError):
        fitting._numeric_scalar(0.0, "x", lower=0.0, lower_open=True)
    with pytest.raises(GP3BayesError):
        fitting._numeric_scalar(1.0, "x", upper=1.0, upper_open=True)

    assert fitting._integer(3.0, "n", minimum=1) == 3
    with pytest.raises(GP3BayesError):
        fitting._integer(3.5, "n", minimum=1)

    monkeypatch.setattr(fitting.os, "cpu_count", lambda: None)
    assert fitting._default_cores(4) == 1
    monkeypatch.setattr(fitting.os, "cpu_count", lambda: 32)
    assert fitting._default_cores(6) == 2

    controls = fitting._validate_sampling_controls(
        chains=2,
        iter=120,
        warmup=20,
        cores=None,
        seed=0,
        adapt_delta=0.9,
        max_treedepth=8,
        refresh=0,
    )
    assert controls.post_warmup_iterations == 100
    assert controls.as_dict()["chains"] == 2

    with pytest.raises(GP3BayesError):
        fitting._validate_sampling_controls(
            chains=2,
            iter=100,
            warmup=100,
            cores=1,
            seed=1,
            adapt_delta=0.9,
            max_treedepth=8,
            refresh=0,
        )
    with pytest.raises(GP3BayesError):
        fitting._validate_sampling_controls(
            chains=1,
            iter=120,
            warmup=20,
            cores=2,
            seed=1,
            adapt_delta=0.9,
            max_treedepth=8,
            refresh=0,
        )

    assert fitting._prior_number(-0.0) == "0"
    with pytest.raises(GP3BayesError):
        fitting._prior_number(np.inf)

    monkeypatch.setattr(
        fitting,
        "version",
        lambda package: (_ for _ in ()).throw(fitting.PackageNotFoundError()),
    )
    assert fitting._package_version("definitely-missing") == "not_installed"

    fitting._importable.cache_clear()
    monkeypatch.setattr(fitting, "find_spec", lambda package: None)
    assert fitting._importable("missing_backend") is False

    fitting._importable.cache_clear()
    monkeypatch.setattr(fitting, "find_spec", lambda package: object())
    monkeypatch.setattr(
        fitting,
        "import_module",
        lambda package: (_ for _ in ()).throw(RuntimeError("broken import")),
    )
    assert fitting._importable("broken_backend") is False

    monkeypatch.setattr(fitting, "_pymc_available", lambda: False)
    with pytest.raises(BackendUnavailableError):
        fitting._require_pymc("fit a test model")

    sentinel = object()
    monkeypatch.setattr(fitting, "import_module", lambda package: sentinel)
    assert fitting._load_pymc() is sentinel

    monkeypatch.setattr(
        fitting,
        "_package_version",
        lambda package: f"{package}-version",
    )
    assert fitting._backend_versions() == {
        "pymc": "pymc-version",
        "arviz": "arviz-version",
    }


def _sampler_frame(parameter: str, values) -> pd.DataFrame:
    values = list(values)
    return pd.DataFrame(
        {
            "Parameter": [parameter] * len(values),
            "Chain": [1, 1, 2, 2][: len(values)],
            "Iteration": list(range(1, len(values) + 1)),
            "Value": values,
        }
    )


def test_posterior_validation_trace_and_sampler_branch_matrix(monkeypatch):
    monkeypatch.setattr(pvc, "_validate_fit_like", lambda fit: fit)

    components = {f"b{i}": np.arange(24, dtype=float).reshape(2, 12) + i for i in range(10)}
    monkeypatch.setattr(pvc, "_posterior_components", lambda fit: components)

    trace = pvc.plot_sampling_diagnostics(object(), type="trace")
    assert len(trace.axes) == 8

    trace_one = pvc.plot_sampling_diagnostics(
        object(),
        type="trace",
        variables="b1",
    )
    assert len(trace_one.axes) == 1

    with pytest.raises(GP3BayesError):
        pvc.plot_sampling_diagnostics(
            object(),
            type="trace",
            variables=("b1", "missing"),
        )
    with pytest.raises(GP3BayesError):
        pvc.plot_sampling_diagnostics(object(), type="invalid")

    monkeypatch.setattr(
        pvc,
        "extract_sampler_diagnostics",
        lambda fit: pd.DataFrame(),
    )
    with pytest.raises(GP3BayesError):
        pvc.plot_sampling_diagnostics(object(), type="energy")

    unrelated = _sampler_frame("other", [0.0, 1.0, 0.5, 0.2])
    monkeypatch.setattr(
        pvc,
        "extract_sampler_diagnostics",
        lambda fit: unrelated,
    )
    with pytest.raises(GP3BayesError):
        pvc.plot_sampling_diagnostics(object(), type="energy")
    with pytest.raises(GP3BayesError):
        pvc.plot_sampling_diagnostics(object(), type="treedepth")
    with pytest.raises(GP3BayesError):
        pvc.plot_sampling_diagnostics(object(), type="divergence")

    energy = _sampler_frame("energy", [1.0, 1.5, 1.2, 1.7])
    monkeypatch.setattr(
        pvc,
        "extract_sampler_diagnostics",
        lambda fit: energy,
    )
    assert pvc.plot_sampling_diagnostics(object(), type="energy").axes

    depth = _sampler_frame("tree_depth", [5, 6, 5, 7])
    monkeypatch.setattr(
        pvc,
        "extract_sampler_diagnostics",
        lambda fit: depth,
    )
    assert pvc.plot_sampling_diagnostics(object(), type="treedepth").axes

    divergence = _sampler_frame("diverging", [0, 1, 0, 1])
    monkeypatch.setattr(
        pvc,
        "extract_sampler_diagnostics",
        lambda fit: divergence,
    )
    assert pvc.plot_sampling_diagnostics(object(), type="divergence").axes


def test_specification_helper_and_prior_guard_branches():
    assert specmod._is_r_syntactic_name("alpha")
    assert specmod._is_r_syntactic_name(".alpha")
    assert not specmod._is_r_syntactic_name("if")
    assert not specmod._is_r_syntactic_name("")
    assert not specmod._is_r_syntactic_name(".1bad")
    assert not specmod._is_r_syntactic_name("_bad")
    assert not specmod._is_r_syntactic_name("a-b")

    assert specmod._quote_formula_name("alpha") == "alpha"
    assert specmod._quote_formula_name("a b") == "`a b`"
    assert specmod._quote_formula_name("a`b") == r"`a\`b`"
    with pytest.raises(GP3BayesError):
        specmod._quote_formula_name("")
    with pytest.raises(GP3BayesError):
        specmod._quote_formula_name(3)

    with pytest.raises(GP3BayesError):
        specmod._validate_numeric_scalar(True, "x")
    with pytest.raises(GP3BayesError):
        specmod._validate_numeric_scalar(np.inf, "x")
    with pytest.raises(GP3BayesError):
        specmod._validate_positive_scalar(0, "x")
    with pytest.raises(GP3BayesError):
        specmod._validate_probability_scalar(0, "p")
    with pytest.raises(GP3BayesError):
        specmod._validate_probability_scalar(1, "p")

    with pytest.raises(GP3BayesError):
        specmod._validate_specification_contract(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        specmod._validate_specification_audit(object())  # type: ignore[arg-type]

    binary_contract = gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=("x",),
        random_slope=False,
    )
    assert specmod.create_prior_specification(binary_contract).family == "binary"
    with pytest.raises(GP3BayesError):
        specmod.create_prior_specification(
            binary_contract,
            residual_scale=1.0,
        )
    with pytest.raises(GP3BayesError):
        specmod.create_prior_specification(
            binary_contract,
            correlation_eta=0.5,
        )

    duration_contract = gp.create_model_contract(
        family="duration",
        outcome_col="duration",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=("x",),
        random_slope=False,
        outcome_unit="milliseconds",
    )
    with pytest.raises(GP3BayesError):
        specmod.create_prior_specification(duration_contract)
    assert (
        specmod.create_prior_specification(
            duration_contract,
            baseline=500.0,
        ).family
        == "duration"
    )


def test_unified_unsupported_and_structural_else_paths():
    unknown_fit = SimpleNamespace(fit_performed=True)

    validation = unified.validate_gp3bayes_object(unknown_fit)
    assert validation.status == "pass"
    assert validation.to_frame().equals(validation.checks)

    with pytest.raises(GP3BayesError):
        unified.diagnose_model_fit(unknown_fit)
    with pytest.raises(GP3BayesError):
        unified.summarise_model_posterior(unknown_fit)
    with pytest.raises(GP3BayesError):
        unified.check_model_ppc(unknown_fit)
    with pytest.raises(GP3BayesError):
        unified.estimate_model_estimands(unknown_fit)

    weird = SimpleNamespace(
        components=["not", "a", "mapping"],
        fit_performed=False,
    )
    stages = unified.model_workflow_status(weird)
    completed = dict(zip(stages["stage"], stages["completed"], strict=True))
    assert completed["diagnostics"] is False
    assert completed["posterior_summary"] is False
    assert completed["ppc"] is False
    assert completed["estimands"] is False
    assert completed["sensitivity"] is False
    assert completed["predictive_validation"] is False
    assert completed["manifest"] is False
    assert stages.attrs["structural_stage_map_only"] is True

    ManifestLike = type("ManifestLike", (), {})
    manifest = ManifestLike()
    manifest.components = ["not", "a", "mapping"]
    stages_manifest = unified.model_workflow_status(manifest)
    completed_manifest = dict(
        zip(
            stages_manifest["stage"],
            stages_manifest["completed"],
            strict=True,
        )
    )
    assert completed_manifest["manifest"] is True
