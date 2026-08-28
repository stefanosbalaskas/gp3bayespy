import inspect
import tomllib
from pathlib import Path

import pytest

import gp3bayespy as gp
import gp3bayespy.binary as binary_module
import gp3bayespy.duration as duration_module


def _binary_spec(*, random_slope: bool = True) -> gp.BinaryModelSpecification:
    simulation = gp.simulate_hierarchical_binary_data(
        n_participants=8,
        trials_per_participant=6,
        n_items=4,
        random_slope_sd=0.2 if random_slope else 0.0,
        seed=4104,
    )
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=["participant_covariate", "trial_covariate"],
        interaction=["condition", "participant_covariate"],
        random_slope=random_slope,
    )
    prepared = gp.prepare_hierarchical_binary_data(
        simulation.data,
        contract,
        condition_levels=["control", "treatment"],
        scale_predictors=["participant_covariate", "trial_covariate"],
    )
    return gp.specify_binary_model(prepared, baseline=0.35)


def _duration_spec(*, random_slope: bool = True) -> gp.DurationModelSpecification:
    simulation = gp.simulate_hierarchical_duration_data(
        n_participants=8,
        trials_per_participant=6,
        n_items=4,
        random_slope_sd=0.15 if random_slope else 0.0,
        seed=4204,
    )
    contract = gp.create_model_contract(
        family="duration",
        outcome_col="duration",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=["participant_covariate", "trial_covariate"],
        interaction=["condition", "participant_covariate"],
        random_slope=random_slope,
        outcome_unit="milliseconds",
    )
    prepared = gp.prepare_hierarchical_duration_data(
        simulation.data,
        contract,
        condition_levels=["control", "treatment"],
        scale_predictors=["participant_covariate", "trial_covariate"],
    )
    return gp.specify_duration_model(prepared, baseline=500.0)


def test_binary_translation_is_restricted_and_preserves_prior_contract():
    specification = _binary_spec(random_slope=True)
    translated = gp.translate_binary_model_to_brms(specification)

    assert isinstance(translated, gp.BinaryBackendSpecification)
    assert translated.family == "binary"
    assert translated.model_family == "hierarchical_binary"
    assert translated.formula_text == specification.formula_text
    assert translated.family_object == {"family": "Bernoulli", "link": "logit"}
    assert translated.backend_interface == "pymc"
    assert translated.sampling_backend == "pymc"
    assert translated.algorithm == "NUTS"
    assert translated.source_backend_interface == "brms"
    assert translated.source_sampling_backend == "rstan"
    assert translated.source_algorithm == "sampling"
    assert not translated.unrestricted_formula
    assert not translated.compiled
    assert not translated.fit_performed
    assert not translated.diagnostics_assessed
    assert set(translated.prior_text) == {"Intercept", "b", "sd", "cor"}
    assert translated.prior_text["cor"] == "lkj(2)"
    user_classes = set(translated.parameter_table["class"])
    assert {"Intercept", "b", "sd", "L"}.issubset(user_classes)


def test_binary_translation_omits_correlation_prior_without_random_slope():
    translated = gp.translate_binary_model_to_brms(_binary_spec(random_slope=False))
    assert "cor" not in translated.prior_text
    assert "L" not in set(translated.parameter_table["class"])


def test_duration_translation_is_restricted_and_preserves_sigma_and_unit():
    specification = _duration_spec(random_slope=True)
    translated = gp.translate_duration_model_to_brms(specification)

    assert isinstance(translated, gp.DurationBackendSpecification)
    assert translated.family == "duration"
    assert translated.model_family == "hierarchical_lognormal_duration"
    assert translated.family_object == {"family": "lognormal", "link": "identity"}
    assert translated.outcome_unit == "milliseconds"
    assert translated.backend_interface == "pymc"
    assert translated.source_backend_interface == "brms"
    assert set(translated.prior_text) == {"Intercept", "b", "sd", "sigma", "cor"}
    user_classes = set(translated.parameter_table["class"])
    assert {"Intercept", "b", "sd", "sigma", "L"}.issubset(user_classes)


def test_translation_rejects_wrong_object_types():
    with pytest.raises(gp.GP3BayesError, match="gp3bayes_binary_model_specification"):
        gp.translate_binary_model_to_brms(object())  # type: ignore[arg-type]
    with pytest.raises(gp.GP3BayesError, match="gp3bayes_duration_model_specification"):
        gp.translate_duration_model_to_brms(object())  # type: ignore[arg-type]


def test_fitting_signatures_expose_only_approved_controls():
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
    forbidden = {"formula", "family", "backend", "algorithm", "prior", "stanvars"}
    for function in (gp.fit_binary_model, gp.fit_duration_model):
        observed = list(inspect.signature(function).parameters)
        assert observed == expected
        assert not forbidden.intersection(observed)


@pytest.mark.parametrize(
    "factory,fit",
    [(_binary_spec, gp.fit_binary_model), (_duration_spec, gp.fit_duration_model)],
)
def test_fitting_controls_fail_before_backend_execution(factory, fit):
    specification = factory()
    with pytest.raises(gp.GP3BayesError, match="`chains` must lie"):
        fit(specification, chains=0)
    with pytest.raises(gp.GP3BayesError, match="`warmup` must be smaller"):
        fit(specification, iter=500, warmup=500)
    with pytest.raises(gp.GP3BayesError, match="`cores` cannot exceed"):
        fit(specification, chains=1, cores=2)
    with pytest.raises(gp.GP3BayesError, match="`adapt_delta` must lie"):
        fit(specification, adapt_delta=1)
    with pytest.raises(gp.GP3BayesError, match="`max_treedepth` must lie"):
        fit(specification, max_treedepth=4)


def test_binary_fit_records_conservative_status_without_claiming_adequacy(monkeypatch):
    specification = _binary_spec(random_slope=False)
    monkeypatch.setattr(binary_module, "_require_pymc", lambda purpose: None)
    monkeypatch.setattr(
        binary_module,
        "_run_binary_pymc",
        lambda specification, controls: ("model", "idata"),
    )

    result = gp.fit_binary_model(
        specification,
        chains=1,
        iter=100,
        warmup=50,
        cores=1,
        seed=22,
    )
    assert isinstance(result, gp.BinaryFit)
    assert result.backend_model == "model"
    assert result.backend_fit == "idata"
    assert result.sampling["post_warmup_iterations"] == 50
    assert result.fit_performed
    assert not result.diagnostics_assessed
    assert not result.posterior_adequacy_established
    assert not result.unrestricted_formula


def test_duration_fit_records_conservative_status_without_claiming_adequacy(monkeypatch):
    specification = _duration_spec(random_slope=False)
    monkeypatch.setattr(duration_module, "_require_pymc", lambda purpose: None)
    monkeypatch.setattr(
        duration_module,
        "_run_duration_pymc",
        lambda specification, controls: ("model", "idata"),
    )

    result = gp.fit_duration_model(
        specification,
        chains=1,
        iter=100,
        warmup=50,
        cores=1,
        seed=23,
    )
    assert isinstance(result, gp.DurationFit)
    assert result.backend_model == "model"
    assert result.backend_fit == "idata"
    assert result.outcome_unit == "milliseconds"
    assert result.sampling["post_warmup_iterations"] == 50
    assert result.fit_performed
    assert not result.diagnostics_assessed
    assert not result.posterior_adequacy_established


def test_backend_and_fit_representations_remain_conservative(monkeypatch):
    binary_spec = _binary_spec(random_slope=False)
    binary_translation = gp.translate_binary_model_to_brms(binary_spec)
    assert "Family: Bernoulli-logit" in repr(binary_translation)
    assert "Fit performed: FALSE" in repr(binary_translation)

    duration_spec = _duration_spec(random_slope=False)
    duration_translation = gp.translate_duration_model_to_brms(duration_spec)
    assert "Family: lognormal" in repr(duration_translation)
    assert "Outcome unit: milliseconds" in repr(duration_translation)

    monkeypatch.setattr(binary_module, "_require_pymc", lambda purpose: None)
    monkeypatch.setattr(
        binary_module,
        "_run_binary_pymc",
        lambda specification, controls: ("model", "idata"),
    )
    binary_fit = gp.fit_binary_model(binary_spec, chains=1, iter=100, warmup=50, cores=1)
    assert "Diagnostics assessed: FALSE" in repr(binary_fit)
    assert "Posterior adequacy established: FALSE" in repr(binary_fit)


def test_translation_backend_available_is_a_boolean():
    binary = gp.translate_binary_model_to_brms(_binary_spec(random_slope=False))
    duration = gp.translate_duration_model_to_brms(_duration_spec(random_slope=False))
    assert isinstance(binary.backend_available, bool)
    assert isinstance(duration.backend_available, bool)


def test_bayes_extra_caps_pytensor_numba_numpy_compatibility():
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)

    bayes = project["project"]["optional-dependencies"]["bayes"]
    assert "numpy>=2.0,<2.5; python_version >= '3.12'" in bayes
    assert "numba>=0.64,<=0.66.0; python_version >= '3.12'" in bayes
    assert "python_version" not in project["tool"]["mypy"]


def test_dev_extra_keeps_pandas_typing_stubs():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert '"pandas-stubs>=2.2"' in pyproject
