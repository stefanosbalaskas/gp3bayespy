from __future__ import annotations

import importlib
from dataclasses import replace
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gp3bayespy as gp
import gp3bayespy.binary as binary
import gp3bayespy.duration as duration
import gp3bayespy.predictive as predictive
from gp3bayespy.exceptions import GP3BayesError

aow = importlib.import_module("gp3bayespy.advanced_optional_workflows")
ds = importlib.import_module("gp3bayespy.design_support_diagnostics")
hea = importlib.import_module("gp3bayespy.hierarchical_effects_advanced")
posterior = importlib.import_module("gp3bayespy.posterior")
postfit = importlib.import_module("gp3bayespy.postfit_exploration")
ppc = importlib.import_module("gp3bayespy.ppc")
readiness = importlib.import_module("gp3bayespy.readiness")
repro = importlib.import_module("gp3bayespy.reproducibility")
sensitivity = importlib.import_module("gp3bayespy.sensitivity")
closure = importlib.import_module("gp3bayespy.specification_closure")


def _arr(values):
    return SimpleNamespace(values=np.asarray(values, dtype=float))


def _binary_fit(seed: int = 11401, random_slope: bool = False):
    sim = gp.simulate_hierarchical_binary_data(
        n_participants=4,
        trials_per_participant=6,
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
        random_slope=random_slope,
    )
    prepared = gp.prepare_hierarchical_binary_data(sim.data, contract)
    spec = gp.specify_binary_model(prepared)
    draws = 8
    pcols = len(prepared.model_matrix_columns)
    npart = prepared.data["participant_id"].nunique()
    nitem = prepared.data["item_id"].nunique()
    posterior_data = {
        "b_Intercept": _arr(np.linspace(-0.2, 0.2, draws)[None, :]),
        "b": _arr(np.zeros((1, draws, max(pcols - 1, 0)))),
        "sd_item": _arr(np.full((1, draws), 0.15)),
        "item_z": _arr(np.zeros((1, draws, nitem))),
    }
    if random_slope:
        posterior_data["participant_re"] = _arr(np.zeros((1, draws, npart, 2)))
    else:
        posterior_data["sd_participant"] = _arr(np.full((1, draws), 0.2))
        posterior_data["participant_z"] = _arr(np.zeros((1, draws, npart)))
    return binary.BinaryFit(
        fit_version="0.1",
        family="binary",
        model_family="hierarchical_binary",
        specification=spec,
        translation=SimpleNamespace(formula_text="selected ~ condition"),
        backend_fit=SimpleNamespace(posterior=posterior_data),
        backend_model=None,
        backend_interface="pymc",
        sampling_backend="pymc",
        algorithm="NUTS",
        sampling={"seed": seed},
        package_versions={"gp3bayespy": "0.5.0"},
    )


def _duration_fit(seed: int = 11402):
    sim = gp.simulate_hierarchical_duration_data(
        n_participants=4,
        trials_per_participant=6,
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
        random_slope=False,
        outcome_unit="milliseconds",
    )
    prepared = gp.prepare_hierarchical_duration_data(sim.data, contract)
    spec = gp.specify_duration_model(prepared, baseline=500.0)
    draws = 8
    pcols = len(prepared.model_matrix_columns)
    npart = prepared.data["participant_id"].nunique()
    nitem = prepared.data["item_id"].nunique()
    posterior_data = {
        "b_Intercept": _arr(np.full((1, draws), np.log(500.0))),
        "b": _arr(np.zeros((1, draws, max(pcols - 1, 0)))),
        "sigma": _arr(np.full((1, draws), 0.1)),
        "sd_participant": _arr(np.full((1, draws), 0.15)),
        "participant_z": _arr(np.zeros((1, draws, npart))),
        "sd_item": _arr(np.full((1, draws), 0.10)),
        "item_z": _arr(np.zeros((1, draws, nitem))),
    }
    return duration.DurationFit(
        fit_version="0.1",
        family="duration",
        model_family="hierarchical_duration",
        specification=spec,
        translation=SimpleNamespace(formula_text="duration ~ condition"),
        backend_fit=SimpleNamespace(posterior=posterior_data),
        backend_model=None,
        outcome_unit="milliseconds",
        backend_interface="pymc",
        sampling_backend="pymc",
        algorithm="NUTS",
        sampling={"seed": seed},
        package_versions={"gp3bayespy": "0.5.0"},
    )


def _binary_intercept_fit(seed: int = 11403):
    sim = gp.simulate_hierarchical_binary_data(
        n_participants=4,
        trials_per_participant=5,
        n_items=2,
        include_items=False,
        seed=seed,
    )
    data = sim.data[["selected", "participant_id"]].copy()
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
    )
    prepared = gp.prepare_hierarchical_binary_data(data, contract)
    spec = gp.specify_binary_model(prepared)
    draws = 6
    npart = prepared.data["participant_id"].nunique()
    posterior_data = {
        "b_Intercept": _arr(np.linspace(-0.1, 0.1, draws)[None, :]),
        "sd_participant": _arr(np.full((1, draws), 0.2)),
        "participant_z": _arr(np.zeros((1, draws, npart))),
    }
    return binary.BinaryFit(
        fit_version="0.1",
        family="binary",
        model_family="hierarchical_binary",
        specification=spec,
        translation=SimpleNamespace(formula_text="selected ~ 1"),
        backend_fit=SimpleNamespace(posterior=posterior_data),
        backend_model=None,
        backend_interface="pymc",
        sampling_backend="pymc",
        algorithm="NUTS",
        sampling={"seed": seed},
        package_versions={"gp3bayespy": "0.5.0"},
    )


@pytest.fixture(autouse=True)
def _close_plots():
    yield
    plt.close("all")


def test_binary_formula_mapping_string_condition_and_time_scaling(monkeypatch):
    string_condition = pd.Series(["b", "a", "b", "a"], dtype=object)
    coded, levels, _ = binary._code_condition(
        string_condition,
        None,
        (-0.5, 0.5),
    )
    assert levels == ("a", "b")
    assert set(coded) == {-0.5, 0.5}

    with pytest.raises(GP3BayesError, match="outcome_mapping"):
        binary._map_binary_outcome(pd.Series([0, 2]), None)

    interaction_contract = gp.create_model_contract(
        family="binary",
        outcome_col="y",
        participant_col="p",
        predictors=("x", "z", "x:z"),
        interaction=("x", "z"),
    )
    with monkeypatch.context() as patch:
        patch.setattr(binary, "_quote_name", lambda value: value)
        formula = binary._fixed_formula_text(interaction_contract)
    assert formula.count("x:z") == 1

    sim = gp.simulate_hierarchical_binary_data(
        n_participants=4,
        trials_per_participant=6,
        n_items=3,
        seed=11410,
    )
    frame = sim.data.copy()
    frame["time_index"] = np.arange(len(frame), dtype=float)
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        time_col="time_index",
        predictors=("participant_covariate", "trial_covariate"),
    )
    prepared = binary.prepare_hierarchical_binary_data(
        frame,
        contract,
        scale_time=True,
    )
    assert "time_index" in prepared.transformations["numeric_scaling"]


def test_binary_readiness_specification_prior_row_and_report_guards(monkeypatch, tmp_path):
    fit = _binary_fit(11411)
    prepared = fit.specification.prepared
    assert prepared is not None

    bad_prepared = replace(prepared, audit=SimpleNamespace(ready=False))
    with pytest.raises(GP3BayesError, match="not ready"):
        binary.specify_binary_model(bad_prepared)

    with monkeypatch.context() as patch:
        patch.setattr(
            binary,
            "audit_model_readiness",
            lambda *args, **kwargs: SimpleNamespace(ready=False, status="fail"),
        )
        with pytest.raises(GP3BayesError, match="readiness"):
            binary.prepare_hierarchical_binary_data(
                prepared.data.copy(),
                prepared.contract,
            )

    with monkeypatch.context() as patch:
        patch.setattr(binary, "validate_prior_specification", lambda *args, **kwargs: None)
        fake_priors = SimpleNamespace(table=pd.DataFrame({"parameter_class": ["Intercept"]}))
        with pytest.raises(GP3BayesError, match="exactly one"):
            binary._prior_row(fake_priors, "b")

    diagnostics = SimpleNamespace(
        status="pass",
        component_table=pd.DataFrame({"check": ["rhat"], "status": ["pass"]}),
    )
    posterior_summary = SimpleNamespace(table=pd.DataFrame({"variable": ["b_x"], "mean": [0.0]}))
    target = tmp_path / "binary-j11.md"
    report = binary.create_binary_model_report(
        fit,
        diagnostics=diagnostics,
        posterior_summary=posterior_summary,
        posterior_predictive=SimpleNamespace(status="available"),
        prior_sensitivity=None,
        recovery=None,
        file=str(target),
    )
    assert report.file == str(target.resolve())
    assert target.exists()


def test_duration_formula_missing_time_readiness_and_specification(monkeypatch):
    interaction_contract = gp.create_model_contract(
        family="duration",
        outcome_col="y",
        participant_col="p",
        predictors=("x", "z", "x:z"),
        interaction=("x", "z"),
        outcome_unit="ms",
    )
    with monkeypatch.context() as patch:
        patch.setattr(duration, "_quote_name", lambda value: value)
        formula = duration._fixed_formula_text(interaction_contract)
    assert formula.count("x:z") == 1

    sim = gp.simulate_hierarchical_duration_data(
        n_participants=4,
        trials_per_participant=6,
        n_items=3,
        seed=11420,
    )
    contract = gp.create_model_contract(
        family="duration",
        outcome_col="duration",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=("participant_covariate", "trial_covariate"),
        outcome_unit="milliseconds",
    )
    with pytest.raises(GP3BayesError, match="Required duration columns"):
        duration.prepare_hierarchical_duration_data(
            sim.data.drop(columns="trial_covariate"),
            contract,
        )

    timed = sim.data.copy()
    timed["time_index"] = np.arange(len(timed), dtype=float)
    timed_contract = gp.create_model_contract(
        family="duration",
        outcome_col="duration",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        time_col="time_index",
        predictors=("participant_covariate", "trial_covariate"),
        outcome_unit="milliseconds",
    )
    timed_prepared = duration.prepare_hierarchical_duration_data(
        timed,
        timed_contract,
        scale_time=True,
    )
    assert "time_index" in timed_prepared.transformations["scaled_columns"]

    valid = duration.prepare_hierarchical_duration_data(sim.data, contract)
    spec = duration.specify_duration_model(valid, baseline=500.0)

    with monkeypatch.context() as patch:
        patch.setattr(
            duration,
            "audit_model_readiness",
            lambda *args, **kwargs: SimpleNamespace(ready=False),
        )
        with pytest.raises(GP3BayesError, match="readiness"):
            duration.prepare_hierarchical_duration_data(sim.data, contract)

    with pytest.raises(GP3BayesError, match="prepared"):
        duration._validate_duration_model_specification(replace(spec, prepared=None))
    with pytest.raises(GP3BayesError, match="readiness"):
        duration._validate_duration_model_specification(
            replace(spec, audit=SimpleNamespace(ready=False))
        )


def test_duration_intercept_only_prior_predictive_and_report(tmp_path):
    sim = gp.simulate_hierarchical_duration_data(
        n_participants=4,
        trials_per_participant=5,
        n_items=2,
        include_items=False,
        seed=11421,
    )
    frame = sim.data[["duration", "participant_id"]].copy()
    contract = gp.create_model_contract(
        family="duration",
        outcome_col="duration",
        participant_col="participant_id",
        outcome_unit="milliseconds",
    )
    prepared = duration.prepare_hierarchical_duration_data(frame, contract)
    spec = duration.specify_duration_model(prepared, baseline=500.0)
    check = duration.check_duration_prior_predictive(
        spec,
        draws=50,
        seed=11422,
        maximum_extreme_probability=1.0,
    )
    assert check.draws == 50

    fit = _duration_fit(11423)
    diagnostics = SimpleNamespace(
        status="pass",
        component_table=pd.DataFrame({"check": ["rhat"], "status": ["pass"]}),
    )
    posterior_summary = SimpleNamespace(table=pd.DataFrame({"variable": ["b_x"], "mean": [0.0]}))
    target = tmp_path / "duration-j11.md"
    report = duration.create_duration_model_report(
        fit,
        diagnostics=diagnostics,
        posterior_summary=posterior_summary,
        posterior_predictive=SimpleNamespace(status="available"),
        prior_sensitivity=None,
        recovery=None,
        file=str(target),
    )
    assert report.file == str(target.resolve())
    assert target.exists()


def test_duration_prior_sensitivity_retain_fits_false(monkeypatch):
    fit = _duration_fit(11424)
    fit = replace(
        fit,
        sampling={
            "chains": 1,
            "iter": 10,
            "warmup": 5,
            "cores": 1,
            "seed": 11424,
            "adapt_delta": 0.9,
            "max_treedepth": 10,
        },
    )
    summary_table = pd.DataFrame(
        {
            "variable": ["b_Intercept", "b_condition", "sigma"],
            "median": [np.log(500.0), 0.0, 0.1],
            "sd": [0.1, 0.1, 0.05],
        }
    )

    with monkeypatch.context() as patch:
        patch.setattr(
            duration,
            "summarise_duration_posterior",
            lambda fit: SimpleNamespace(table=summary_table.copy()),
        )
        patch.setattr(
            duration,
            "diagnose_duration_fit",
            lambda fit: SimpleNamespace(status="pass"),
        )
        patch.setattr(
            duration,
            "fit_duration_model",
            lambda specification, **kwargs: replace(
                fit,
                specification=specification,
            ),
        )
        result = duration.assess_duration_prior_sensitivity(
            fit,
            scale_multipliers={"same": 1.0},
            retain_fits=False,
        )
    assert result.family == "duration"
    assert result.alternative_fits is None


def test_predictive_required_at_groupless_and_intercept_only_paths():
    fit = _binary_fit(11430)
    participant = fit.specification.prepared.data["participant_id"].iloc[0]
    grid = predictive.create_prediction_grid(
        fit,
        variables=[],
        at={"participant_id": [participant]},
    )
    assert grid["participant_id"].iloc[0] == participant

    contract = SimpleNamespace(
        mappings={"participant": None, "item": None},
        random_slope=False,
    )
    fake_fit = SimpleNamespace(
        specification=SimpleNamespace(
            contract=contract,
            prepared=SimpleNamespace(data=pd.DataFrame({"x": [1, 2]})),
        )
    )
    newdata = pd.DataFrame({"x": [3, 4]})
    effects = predictive._group_effect_matrix(
        fake_fit,
        newdata,
        ndraws=2,
        allow_new_levels=False,
        seed=1,
    )
    assert effects.shape == (2, 2)
    assert np.allclose(effects, 0)

    intercept_fit = _binary_intercept_fit()
    eta = predictive._linear_prediction_matrix(
        intercept_fit,
        intercept_fit.specification.prepared.data.head(2),
        include_group_effects=False,
        allow_new_levels=False,
        ndraws=3,
        seed=1,
    )
    assert eta.shape == (3, 2)


def test_predictive_remaining_validation_and_statistic_guard(monkeypatch):
    fit = _binary_fit(11431)
    expected = predictive.predict_model(
        fit,
        type="expected",
        include_group_effects=False,
        ndraws=4,
    )
    with pytest.raises(GP3BayesError, match="direction"):
        predictive.prediction_exceedance_probability(
            expected,
            0.5,
            direction="sideways",
        )
    with pytest.raises(GP3BayesError, match="measure"):
        predictive.prediction_pairwise_contrasts(
            expected,
            rows=[1, 2],
            measure="bad",
        )

    pred = predictive.predict_model(
        fit,
        type="predictive",
        include_group_effects=False,
        ndraws=4,
        seed=2,
    )
    with monkeypatch.context() as patch:

        def nonfinite_stat(values, statistic, threshold, axis):
            if axis is None:
                return float("nan")
            return np.full(np.asarray(values).shape[0], np.nan)

        patch.setattr(predictive, "_statistic_values", nonfinite_stat)
        with pytest.raises(GP3BayesError, match="non-finite"):
            predictive.posterior_predictive_statistic(
                pred,
                statistic="mean",
            )


def test_prediction_contrast_profile_remaining_guards(monkeypatch):
    fit = _binary_fit(11432)

    with pytest.raises(GP3BayesError, match="measure"):
        predictive.create_prediction_contrast_profile(
            fit,
            variable="trial_covariate",
            contrast_variable="condition",
            contrast_levels=[-0.5, 0.5],
            values=[-0.1, 0.1],
            measure="bad",
            ndraws=4,
        )

    with pytest.raises(GP3BayesError, match="contrast_levels"):
        predictive.create_prediction_contrast_profile(
            fit,
            variable="trial_covariate",
            contrast_variable="participant_id",
            contrast_levels=None,
            values=[-0.1, 0.1],
            ndraws=4,
        )

    with monkeypatch.context() as patch:
        patch.setattr(
            predictive,
            "create_prediction_grid",
            lambda *args, **kwargs: pd.DataFrame(
                {
                    "trial_covariate": [0.1, 0.2],
                    "condition": [-0.5, -0.5],
                }
            ),
        )
        patch.setattr(
            predictive,
            "predict_model",
            lambda fit, newdata=None, **kwargs: SimpleNamespace(draws=np.ones((2, len(newdata)))),
        )
        with pytest.raises(GP3BayesError, match="Contrast grid"):
            predictive.create_prediction_contrast_profile(
                fit,
                variable="trial_covariate",
                contrast_variable="condition",
                contrast_levels=[-0.5, 0.5],
                values=[0.1, 0.2],
                ndraws=2,
            )

    with monkeypatch.context() as patch:
        patch.setattr(
            predictive,
            "predict_model",
            lambda fit, newdata=None, **kwargs: SimpleNamespace(draws=np.zeros((2, len(newdata)))),
        )
        with pytest.raises(GP3BayesError, match="positive denominators"):
            predictive.create_prediction_contrast_profile(
                fit,
                variable="trial_covariate",
                contrast_variable="condition",
                contrast_levels=[-0.5, 0.5],
                values=[0.1, 0.2],
                measure="ratio",
                ndraws=2,
            )


def test_predictive_atlas_and_valid_plot_paths():
    fit = _binary_fit(11433)
    atlas = predictive._atlas_get(fit, 4, False, 1)
    assert atlas.family == "binary"

    expected = predictive.predict_model(
        fit,
        type="expected",
        include_group_effects=False,
        ndraws=4,
    )
    fig1 = predictive.plot_prediction_draws(expected, max_draws=4)
    assert fig1.axes
    plt.close(fig1)

    fig2 = predictive.plot_binary_group_calibration(
        expected,
        group="participant_id",
    )
    assert fig2.axes
    plt.close(fig2)


def test_psis_no_tail_fit_branch():
    weights, k = aow._psis_smooth(np.zeros(10))
    assert np.isclose(weights.sum(), 1.0)
    assert np.isnan(k)


def test_hierarchical_four_dimensional_intercept_and_missing_sd(monkeypatch):
    variables = {
        "sd_participant": SimpleNamespace(values=np.ones((1, 2, 1))),
        "participant_z": SimpleNamespace(values=np.ones((1, 2, 3, 1))),
    }
    with monkeypatch.context() as patch:
        patch.setattr(hea, "_validate_fit_like", lambda fit: fit)
        patch.setattr(hea, "_posterior_data_vars", lambda fit: variables)
        groups = hea._group_arrays(SimpleNamespace(specification=None))
    assert groups["participant"][0].ndim == 4

    with monkeypatch.context() as patch:
        patch.setattr(hea, "_validate_fit_like", lambda fit: fit)
        patch.setattr(hea, "_posterior_components", lambda fit: {})
        with pytest.raises(GP3BayesError, match="random-intercept"):
            hea.random_intercept_variance_partition(SimpleNamespace(family="binary"))


def test_design_threshold_rank_zero_and_true_pass_branch(monkeypatch):
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="y",
        participant_col="p",
        predictors=("x",),
    )
    data = pd.DataFrame(
        {
            "y": [0, 1] * 10,
            "p": [f"p{i}" for i in range(20)],
            "x": [-1.0, 1.0] * 10,
        }
    )

    with pytest.raises(GP3BayesError, match="threshold"):
        ds.audit_missingness_structure(
            data,
            contract,
            review_fraction=0.5,
            fail_fraction=0.2,
        )

    interaction_contract = gp.create_model_contract(
        family="binary",
        outcome_col="y",
        participant_col="p",
        predictors=("x", "z"),
        interaction=("x", "z"),
    )
    declared = ds._declared_columns(interaction_contract)
    assert declared.count("x") == 1
    assert declared.count("z") == 1

    with monkeypatch.context() as patch:
        patch.setattr(
            ds,
            "_fixed_model_matrix",
            lambda frame, contract: (
                np.empty((len(frame), 0), dtype=float),
                (),
            ),
        )
        rank_zero = ds.audit_fixed_effect_design(data, contract)
    assert rank_zero.rank == 0
    assert rank_zero.status == "fail"

    with monkeypatch.context() as patch:
        patch.setattr(
            ds,
            "_fixed_model_matrix",
            lambda frame, contract: (
                np.column_stack(
                    [
                        np.ones(len(frame), dtype=float),
                        np.resize(np.array([-1.0, 1.0]), len(frame)),
                    ]
                ),
                ("Intercept", "x"),
            ),
        )
        passed = ds.audit_fixed_effect_design(
            data,
            contract,
            leverage_multiplier=3,
        )
    assert passed.status == "pass"


def test_posterior_parameter_only_selection_guards(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(
            posterior,
            "_posterior_components",
            lambda fit: {"other": np.ones((1, 2))},
        )
        with pytest.raises(GP3BayesError, match="supported posterior"):
            posterior._select_components(object(), parameters_only=True)

    with monkeypatch.context() as patch:
        patch.setattr(
            posterior,
            "_posterior_components",
            lambda fit: {
                "b_x": np.ones((1, 2)),
                "other": np.ones((1, 2)),
            },
        )
        with pytest.raises(GP3BayesError, match="Requested posterior"):
            posterior._select_components(
                object(),
                variables=["missing"],
                parameters_only=True,
            )


def test_postfit_participant_absent_item_fallback(monkeypatch):
    fake = SimpleNamespace(
        specification=SimpleNamespace(
            prepared=SimpleNamespace(data=pd.DataFrame({"item": ["level1", "level1"]})),
            contract=SimpleNamespace(mappings={"participant": None, "item": "item"}),
        )
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            postfit,
            "_posterior_components",
            lambda fit: {"r_item[level1]": np.array([[0.1, 0.2], [0.2, 0.3]])},
        )
        table = postfit.group_effect_table(fake)
    assert table["group"].tolist() == ["item"]


def test_ppc_no_condition_and_readiness_period_dtype():
    summary = ppc._duration_summary(
        [1.0, 2.0, 4.0, 8.0],
        condition=None,
        participant=["p1", "p1", "p2", "p2"],
        item=None,
    )
    assert np.isnan(summary["condition_median_ratio"])

    periods = pd.Series(pd.period_range("2026-01", periods=2, freq="M"))
    assert readiness._supported_identifier(periods) is False


def test_reproducibility_explicit_specification_branch():
    fit = _binary_fit(11434)
    manifest = repro.create_analysis_manifest(
        specification=fit.specification,
        fit=fit,
        label="j11",
    )
    assert manifest.family == "binary"


def test_sbc_missing_rank_guard():
    sbc = {
        "raw": {
            "stats": [
                {"variable": "b", "draws": 10},
                {"variable": "b", "draws": 10},
            ]
        }
    }
    with pytest.raises(GP3BayesError, match="rank"):
        sensitivity.plot_sbc_ecdf_gg(sbc)


def test_specification_closure_inversion_and_duration_scaling_guard(monkeypatch):
    binary_recipe = SimpleNamespace(
        family="binary",
        contract=SimpleNamespace(mappings={"outcome": "y", "condition": None}),
        transformations={
            "numeric_scaling": {},
            "condition": None,
            "outcome": {"mapping": {0: 0, 1: 1}},
        },
    )
    duration_recipe = SimpleNamespace(
        family="duration",
        contract=SimpleNamespace(mappings={"outcome": "duration", "condition": None}),
        transformations={
            "scaled_columns": {},
            "condition": None,
            "outcome": {"multiplier": 1.0},
        },
    )
    with monkeypatch.context() as patch:
        patch.setattr(closure, "_as_recipe", lambda recipe: recipe)
        binary_out = closure.invert_transformation_recipe(
            pd.DataFrame({"x": [1.0]}),
            binary_recipe,
        )
        duration_out = closure.invert_transformation_recipe(
            pd.DataFrame({"x": [1.0]}),
            duration_recipe,
        )
    assert list(binary_out) == ["x"]
    assert list(duration_out) == ["x"]

    fake_specification = SimpleNamespace(
        family="duration",
        contract=SimpleNamespace(predictors=("x",)),
        prepared=SimpleNamespace(
            transformations={"scaled_columns": {}},
            data=pd.DataFrame({"x": [0.0, 1.0]}),
        ),
    )
    with pytest.raises(GP3BayesError, match="scaled during duration"):
        closure.create_predictor_scaling_sensitivity_specification(
            fake_specification,
            predictor="x",
            scale_factor=2.0,
            coefficient_scale=1.0,
        )


def test_binary_ppc_details_item_absent_branch(monkeypatch):
    pred = SimpleNamespace(
        observed=np.array([0.0, 1.0]),
        draws=np.array(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ]
        ),
    )
    expected = SimpleNamespace()
    prepared = SimpleNamespace(
        contract=SimpleNamespace(mappings={"participant": "p", "item": None}),
        data=pd.DataFrame({"p": ["p1", "p2"]}),
    )
    fit = SimpleNamespace(
        family="binary",
        specification=SimpleNamespace(prepared=prepared),
    )

    with monkeypatch.context() as patch:
        patch.setattr(
            predictive,
            "predict_model",
            lambda fit, type="expected", **kwargs: pred if type == "predictive" else expected,
        )
        patch.setattr(
            predictive,
            "binary_calibration_table",
            lambda *args, **kwargs: pd.DataFrame({"bin": [1], "observed_rate": [0.5]}),
        )
        result = closure.check_binary_ppc_details(
            fit,
            draws=3,
            calibration_bins=2,
        )
    assert set(result["groups"]) == {"participant"}
