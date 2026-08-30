from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
import gp3bayespy.binary as b
import gp3bayespy.duration as d
import gp3bayespy.predictive as predictive
from gp3bayespy.exceptions import GP3BayesError


def _binary_spec(seed: int = 1501):
    sim = gp.simulate_hierarchical_binary_data(
        n_participants=5,
        trials_per_participant=4,
        n_items=3,
        seed=seed,
    )
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=("participant_covariate", "trial_covariate"),
        interaction=("condition", "participant_covariate"),
        random_slope=False,
    )
    prepared = gp.prepare_hierarchical_binary_data(sim.data, contract)
    return sim, gp.specify_binary_model(prepared)


def _duration_spec(seed: int = 1502):
    sim = gp.simulate_hierarchical_duration_data(
        n_participants=5,
        trials_per_participant=4,
        n_items=3,
        seed=seed,
    )
    contract = gp.create_model_contract(
        family="duration",
        outcome_col="duration",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=("participant_covariate", "trial_covariate"),
        interaction=("condition", "participant_covariate"),
        random_slope=False,
        outcome_unit="milliseconds",
    )
    prepared = gp.prepare_hierarchical_duration_data(sim.data, contract)
    return sim, gp.specify_duration_model(prepared, baseline=500.0)


def _binary_fit(specification, seed: int = 41):
    return b.BinaryFit(
        fit_version="0.1",
        family="binary",
        model_family="hierarchical_binary",
        specification=specification,
        translation=SimpleNamespace(formula_text="selected ~ condition + participant_covariate"),
        backend_fit=None,
        backend_model=None,
        backend_interface="pymc",
        sampling_backend="pymc",
        algorithm="NUTS",
        sampling={
            "chains": 2,
            "iter": 100,
            "warmup": 50,
            "post_warmup_iterations": 50,
            "cores": 1,
            "seed": seed,
            "adapt_delta": 0.9,
            "max_treedepth": 10,
            "refresh": 0,
        },
        package_versions={"gp3bayespy": "0.5.0"},
    )


def _duration_fit(specification, seed: int = 51):
    return d.DurationFit(
        fit_version="0.1",
        family="duration",
        model_family="hierarchical_lognormal_duration",
        specification=specification,
        translation=SimpleNamespace(formula_text="duration ~ condition + participant_covariate"),
        backend_fit=None,
        backend_model=None,
        outcome_unit="milliseconds",
        backend_interface="pymc",
        sampling_backend="pymc",
        algorithm="NUTS",
        sampling={
            "chains": 2,
            "iter": 100,
            "warmup": 50,
            "post_warmup_iterations": 50,
            "cores": 1,
            "seed": seed,
            "adapt_delta": 0.9,
            "max_treedepth": 10,
            "refresh": 0,
        },
        package_versions={"gp3bayespy": "0.5.0"},
    )


def test_binary_fit_repr_type_guards_and_ppc(monkeypatch):
    _, spec = _binary_spec()
    fit = _binary_fit(spec)
    assert "<gp3bayes_binary_fit>" in repr(fit)
    assert "Iterations per chain: 100" in repr(fit)

    with pytest.raises(GP3BayesError):
        b.diagnose_binary_fit(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        b.summarise_binary_posterior(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        b.check_binary_posterior_predictive(object())  # type: ignore[arg-type]

    data = spec.prepared.data
    y = data["selected"].to_numpy(dtype=int)
    rng = np.random.default_rng(1503)
    yrep = rng.binomial(
        1,
        np.broadcast_to(np.linspace(0.2, 0.8, len(data))[None, :], (60, len(data))),
    ).astype(float)
    expected = np.broadcast_to(
        np.linspace(0.2, 0.8, len(data))[None, :],
        (60, len(data)),
    ).copy()
    monkeypatch.setattr(
        predictive,
        "extract_posterior_predictions",
        lambda *args, **kwargs: yrep,
    )
    monkeypatch.setattr(
        predictive,
        "extract_expected_predictions",
        lambda *args, **kwargs: expected,
    )
    result = b.check_binary_posterior_predictive(fit, draws=60, seed=7)
    assert result.draws == 60
    assert np.isfinite(result.brier_score)
    assert "overall_rate" in result.observed
    assert "participant_rate_sd" in result.observed

    # Reproduce a deterministic observed-like PPC branch.
    monkeypatch.setattr(
        predictive,
        "extract_posterior_predictions",
        lambda *args, **kwargs: np.broadcast_to(y[None, :], (50, len(y))).copy(),
    )
    result2 = b.check_binary_posterior_predictive(fit, draws=50, seed=8)
    assert result2.draws == 50


def test_duration_fit_repr_type_guards_and_ppc(monkeypatch):
    _, spec = _duration_spec()
    fit = _duration_fit(spec)
    assert "<gp3bayes_duration_fit>" in repr(fit)
    assert "Outcome unit: milliseconds" in repr(fit)

    with pytest.raises(GP3BayesError):
        d.diagnose_duration_fit(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        d.summarise_duration_posterior(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        d.check_duration_posterior_predictive(object())  # type: ignore[arg-type]

    data = spec.prepared.data
    n = len(data)
    rng = np.random.default_rng(1504)
    center = data["duration"].to_numpy(float)
    yrep = np.maximum(rng.normal(center[None, :], 40, size=(60, n)), 1.0)
    expected = np.maximum(rng.normal(center[None, :], 10, size=(60, n)), 1.0)
    monkeypatch.setattr(
        predictive,
        "extract_posterior_predictions",
        lambda *args, **kwargs: yrep,
    )
    monkeypatch.setattr(
        predictive,
        "extract_expected_predictions",
        lambda *args, **kwargs: expected,
    )
    result = d.check_duration_posterior_predictive(fit, draws=60, seed=9)
    assert result.draws == 60
    assert np.isfinite(result.log_scale_rmse)

    monkeypatch.setattr(
        predictive,
        "extract_posterior_predictions",
        lambda *args, **kwargs: np.zeros((50, n)),
    )
    with pytest.raises(GP3BayesError, match="strictly positive"):
        d.check_duration_posterior_predictive(fit, draws=50, seed=10)


def test_binary_prior_sensitivity_full_orchestration(monkeypatch):
    _, spec = _binary_spec(1510)
    fit = _binary_fit(spec, seed=70)

    reference = pd.DataFrame(
        {
            "variable": ["b_Intercept", "b_condition"],
            "median": [0.0, 0.2],
            "sd": [0.4, 0.5],
        }
    )

    def fake_summary(x, *args, **kwargs):
        shift = 0.0 if x is fit else 0.08
        table = reference.copy()
        table["median"] = table["median"] + shift
        return SimpleNamespace(table=table)

    monkeypatch.setattr(b, "summarise_binary_posterior", fake_summary)
    monkeypatch.setattr(
        b,
        "diagnose_binary_fit",
        lambda x, *args, **kwargs: SimpleNamespace(status="pass"),
    )
    monkeypatch.setattr(
        b,
        "fit_binary_model",
        lambda specification, **kwargs: _binary_fit(specification, kwargs["seed"]),
    )

    result = b.assess_binary_prior_sensitivity(
        fit,
        scale_multipliers={"tight": 0.5, "wide": 2.0},
        retain_fits=True,
        maximum_standardized_shift=0.25,
        review_standardized_shift=0.5,
    )
    assert len(result.comparison) == 4
    assert len(result.scenario_status) == 2
    assert set(result.alternative_fits) == {"tight", "wide"}
    assert result.status == "pass"

    no_retain = b.assess_binary_prior_sensitivity(
        fit,
        scale_multipliers={"wide": 2.0},
        retain_fits=False,
    )
    assert no_retain.alternative_fits is None

    with pytest.raises(GP3BayesError):
        b.assess_binary_prior_sensitivity(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        b.assess_binary_prior_sensitivity(fit, scale_multipliers={})
    with pytest.raises(GP3BayesError):
        b.assess_binary_prior_sensitivity(fit, scale_multipliers={"bad": 0})
    with pytest.raises(GP3BayesError):
        b.assess_binary_prior_sensitivity(
            fit,
            maximum_standardized_shift=0.5,
            review_standardized_shift=0.2,
        )
    with pytest.raises(GP3BayesError):
        b._prior_scale_from_table(
            SimpleNamespace(table=pd.DataFrame({"parameter_class": ["x"], "scale": [1.0]})),
            "b",
        )


def test_duration_prior_sensitivity_full_orchestration(monkeypatch):
    _, spec = _duration_spec(1520)
    fit = _duration_fit(spec, seed=80)

    reference = pd.DataFrame(
        {
            "variable": ["b_Intercept", "b_condition", "sigma"],
            "median": [6.0, 0.1, 0.3],
            "sd": [0.3, 0.2, 0.1],
        }
    )

    def fake_summary(x, *args, **kwargs):
        shift = 0.0 if x is fit else 0.03
        table = reference.copy()
        table["median"] = table["median"] + shift
        return SimpleNamespace(table=table)

    monkeypatch.setattr(d, "summarise_duration_posterior", fake_summary)
    monkeypatch.setattr(
        d,
        "diagnose_duration_fit",
        lambda x, *args, **kwargs: SimpleNamespace(status="pass"),
    )
    monkeypatch.setattr(
        d,
        "fit_duration_model",
        lambda specification, **kwargs: _duration_fit(specification, kwargs["seed"]),
    )

    result = d.assess_duration_prior_sensitivity(
        fit,
        scale_multipliers={"tight": 0.5, "wide": 2.0},
        retain_fits=True,
    )
    assert len(result.comparison) == 6
    assert len(result.scenario_status) == 2
    assert set(result.alternative_fits) == {"tight", "wide"}

    with pytest.raises(GP3BayesError):
        d.assess_duration_prior_sensitivity(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        d.assess_duration_prior_sensitivity(fit, scale_multipliers={})
    with pytest.raises(GP3BayesError):
        d.assess_duration_prior_sensitivity(fit, scale_multipliers={"bad": -1})
    with pytest.raises(GP3BayesError):
        d.assess_duration_prior_sensitivity(
            fit,
            maximum_standardized_shift=0.6,
            review_standardized_shift=0.5,
        )
    with pytest.raises(GP3BayesError):
        d._duration_prior_scale_from_table(
            SimpleNamespace(table=pd.DataFrame({"parameter_class": ["x"], "scale": [1.0]})),
            "sigma",
        )


def test_binary_recovery_success_error_and_stop(monkeypatch):
    monkeypatch.setattr(
        b,
        "fit_binary_model",
        lambda specification, **kwargs: _binary_fit(specification, kwargs["seed"]),
    )
    monkeypatch.setattr(
        b,
        "diagnose_binary_fit",
        lambda fit, *args, **kwargs: SimpleNamespace(status="pass"),
    )
    monkeypatch.setattr(
        b,
        "summarise_binary_posterior",
        lambda fit, **kwargs: SimpleNamespace(
            table=pd.DataFrame(
                {
                    "variable": ["b_Intercept"],
                    "median": [0.0],
                    "lower": [-20.0],
                    "upper": [20.0],
                }
            )
        ),
    )
    result = b.run_binary_recovery(
        repetitions=2,
        n_participants=4,
        trials_per_participant=4,
        n_items=3,
        include_items=False,
        random_slope=False,
        minimum_repetitions=2,
        seed=1530,
    )
    assert len(result.fit_status) == 2
    assert result.fit_status["completed"].all()
    assert not result.estimates.empty

    monkeypatch.setattr(
        b,
        "fit_binary_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fit failed")),
    )
    failed = b.run_binary_recovery(
        repetitions=2,
        n_participants=4,
        trials_per_participant=4,
        n_items=3,
        continue_on_error=True,
        seed=1540,
    )
    assert not failed.fit_status["completed"].any()
    assert failed.estimates.empty

    with pytest.raises(RuntimeError, match="fit failed"):
        b.run_binary_recovery(
            repetitions=2,
            n_participants=4,
            trials_per_participant=4,
            n_items=3,
            continue_on_error=False,
            seed=1550,
        )


def test_duration_recovery_success_error_and_stop(monkeypatch):
    monkeypatch.setattr(
        d,
        "fit_duration_model",
        lambda specification, **kwargs: _duration_fit(specification, kwargs["seed"]),
    )
    monkeypatch.setattr(
        d,
        "diagnose_duration_fit",
        lambda fit, *args, **kwargs: SimpleNamespace(status="pass"),
    )
    monkeypatch.setattr(
        d,
        "summarise_duration_posterior",
        lambda fit, **kwargs: SimpleNamespace(
            table=pd.DataFrame(
                {
                    "variable": ["b_Intercept"],
                    "median": [6.0],
                    "lower": [0.0],
                    "upper": [20.0],
                }
            )
        ),
    )
    result = d.run_duration_recovery(
        repetitions=2,
        n_participants=4,
        trials_per_participant=4,
        n_items=3,
        include_items=False,
        random_slope=False,
        minimum_repetitions=2,
        seed=1560,
    )
    assert len(result.fit_status) == 2
    assert result.fit_status["completed"].all()
    assert not result.estimates.empty

    monkeypatch.setattr(
        d,
        "fit_duration_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fit failed")),
    )
    failed = d.run_duration_recovery(
        repetitions=2,
        n_participants=4,
        trials_per_participant=4,
        n_items=3,
        continue_on_error=True,
        seed=1570,
    )
    assert not failed.fit_status["completed"].any()

    with pytest.raises(RuntimeError, match="fit failed"):
        d.run_duration_recovery(
            repetitions=2,
            n_participants=4,
            trials_per_participant=4,
            n_items=3,
            continue_on_error=False,
            seed=1580,
        )


def test_binary_and_duration_markdown_reports(tmp_path):
    _, bspec = _binary_spec(1590)
    _, dspec = _duration_spec(1591)
    bfit = _binary_fit(bspec)
    dfit = _duration_fit(dspec)

    diagnostics = SimpleNamespace(
        status="pass",
        component_table=pd.DataFrame({"component": ["rhat"], "status": ["pass"]}),
    )
    posterior = SimpleNamespace(table=pd.DataFrame({"variable": ["b_Intercept"], "median": [0.0]}))
    optional = SimpleNamespace(status="review")

    assert b._markdown_table(pd.DataFrame()) == "_(no rows)_"
    assert "| a |" in b._markdown_table(pd.DataFrame({"a": [1]}))
    assert d._duration_markdown_table(pd.DataFrame()) == "_(no rows)_"
    assert "| a |" in d._duration_markdown_table(pd.DataFrame({"a": [1]}))

    bpath = tmp_path / "binary.md"
    breport = b.create_binary_model_report(
        bfit,
        diagnostics=diagnostics,
        posterior_summary=posterior,
        posterior_predictive=optional,
        prior_sensitivity=SimpleNamespace(status="pass"),
        recovery=optional,
        file=str(bpath),
    )
    assert bpath.is_file()
    assert breport.family == "binary"
    assert len(breport.registry) == 5

    dpath = tmp_path / "duration.md"
    dreport = d.create_duration_model_report(
        dfit,
        diagnostics=diagnostics,
        posterior_summary=posterior,
        posterior_predictive=optional,
        prior_sensitivity=SimpleNamespace(status="pass"),
        recovery=optional,
        file=str(dpath),
    )
    assert dpath.is_file()
    assert dreport.family == "duration"

    with pytest.raises(GP3BayesError):
        b.create_binary_model_report(object(), file=str(tmp_path / "x.md"))  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        d.create_duration_model_report(object(), file=str(tmp_path / "y.md"))  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        b.create_binary_model_report(bfit, diagnostics=diagnostics, posterior_summary=posterior)
    with pytest.raises(GP3BayesError):
        d.create_duration_model_report(dfit, diagnostics=diagnostics, posterior_summary=posterior)
    with pytest.raises(GP3BayesError):
        b.create_binary_model_report(
            bfit,
            diagnostics=diagnostics,
            posterior_summary=posterior,
            file=str(bpath),
        )
    with pytest.raises(GP3BayesError):
        d.create_duration_model_report(
            dfit,
            diagnostics=diagnostics,
            posterior_summary=posterior,
            file=str(dpath),
        )
    with pytest.raises(GP3BayesError):
        b.create_binary_model_report(
            bfit,
            diagnostics=diagnostics,
            posterior_summary=posterior,
            file=str(tmp_path / "missing" / "x.md"),
        )
    with pytest.raises(GP3BayesError):
        d.create_duration_model_report(
            dfit,
            diagnostics=diagnostics,
            posterior_summary=posterior,
            file=str(tmp_path / "missing" / "y.md"),
        )
