import math

import numpy as np
import pandas as pd
import pytest

from gp3bayespy import (
    DurationModelSpecification,
    DurationPrepared,
    DurationPriorPredictiveCheck,
    DurationSimulation,
    GP3BayesError,
    check_duration_prior_predictive,
    create_model_contract,
    prepare_hierarchical_duration_data,
    simulate_hierarchical_duration_data,
    specify_duration_model,
)


def _simulation(
    seed: int = 2026,
    *,
    random_slope: bool = True,
    include_items: bool = True,
) -> DurationSimulation:
    return simulate_hierarchical_duration_data(
        n_participants=12,
        trials_per_participant=8,
        n_items=6,
        random_slope_sd=0.20 if random_slope else 0.0,
        include_items=include_items,
        seed=seed,
    )


def _contract(*, random_slope: bool = True, include_item: bool = True):
    return create_model_contract(
        family="duration",
        outcome_col="duration",
        participant_col="participant_id",
        item_col="item_id" if include_item else None,
        trial_col="trial_id",
        condition_col="condition",
        predictors=("participant_covariate", "trial_covariate"),
        interaction=("condition", "participant_covariate"),
        random_slope=random_slope,
        outcome_unit="milliseconds",
    )


def _prepared(
    seed: int = 2026,
    *,
    random_slope: bool = True,
    include_items: bool = True,
) -> DurationPrepared:
    simulation = _simulation(seed, random_slope=random_slope, include_items=include_items)
    return prepare_hierarchical_duration_data(
        simulation.data,
        _contract(random_slope=random_slope, include_item=include_items),
        condition_levels=("control", "treatment"),
    )


def test_duration_simulation_is_deterministic_and_records_truth():
    first = _simulation(seed=2026)
    second = _simulation(seed=2026)
    assert isinstance(first, DurationSimulation)
    pd.testing.assert_frame_equal(first.data, second.data)
    assert first.truth == second.truth
    pd.testing.assert_frame_equal(
        first.random_effects["participant"], second.random_effects["participant"]
    )
    pd.testing.assert_frame_equal(first.random_effects["item"], second.random_effects["item"])
    assert len(first.data) == 96
    assert np.isfinite(first.data["duration"]).all()
    assert (first.data["duration"] > 0).all()
    assert first.truth["outcome_unit"] == "milliseconds"
    assert first.design["censored"] is False


def test_duration_simulation_supports_no_items_or_slopes():
    simulation = _simulation(random_slope=False, include_items=False)
    assert "item_id" not in simulation.data.columns
    assert simulation.random_effects["item"] is None
    assert simulation.design["random_slope"] is False
    assert simulation.design["n_items"] == 0


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"baseline_median": 0}, "baseline_median"),
        ({"residual_sd": 0}, "residual_sd"),
        ({"participant_sd": -1}, "participant_sd"),
        ({"random_slope_cor": 1}, "random_slope_cor"),
        ({"outcome_unit": ""}, "outcome_unit"),
    ],
)
def test_duration_simulation_validates_positive_scales(kwargs, fragment):
    with pytest.raises(GP3BayesError, match=fragment):
        simulate_hierarchical_duration_data(**kwargs)


def test_duration_preparation_records_explicit_unit_conversion():
    simulation = _simulation()
    prepared = prepare_hierarchical_duration_data(
        simulation.data,
        _contract(),
        condition_levels=("control", "treatment"),
        outcome_multiplier=0.001,
        converted_unit="seconds",
    )
    assert isinstance(prepared, DurationPrepared)
    assert prepared.outcome_unit == "seconds"
    np.testing.assert_allclose(prepared.data["duration"], simulation.data["duration"] * 0.001)
    outcome = prepared.transformations["outcome"]
    assert outcome["source_unit"] == "milliseconds"
    assert outcome["analysis_unit"] == "seconds"
    assert outcome["multiplier"] == 0.001
    assert prepared.contract.outcome_unit == "seconds"
    assert prepared.source_contract.outcome_unit == "milliseconds"


def test_duration_unit_conversion_requires_explicit_destination_unit():
    with pytest.raises(GP3BayesError, match="converted_unit"):
        prepare_hierarchical_duration_data(
            _simulation().data,
            _contract(),
            condition_levels=("control", "treatment"),
            outcome_multiplier=0.001,
        )


@pytest.mark.parametrize("replacement", [0.0, -1.0])
def test_duration_preparation_rejects_nonpositive_outcomes(replacement):
    simulation = _simulation()
    data = simulation.data.copy()
    data.loc[data.index[0], "duration"] = replacement
    with pytest.raises(GP3BayesError, match="strictly positive"):
        prepare_hierarchical_duration_data(
            data, _contract(), condition_levels=("control", "treatment")
        )


def test_duration_preparation_rejects_nonfinite_outcomes():
    simulation = _simulation()
    data = simulation.data.copy()
    data.loc[data.index[0], "duration"] = np.inf
    with pytest.raises(GP3BayesError, match="finite numeric"):
        prepare_hierarchical_duration_data(
            data, _contract(), condition_levels=("control", "treatment")
        )


def test_duration_preparation_requires_explicit_missing_decision():
    simulation = _simulation()
    data = simulation.data.copy()
    data.loc[data.index[0], "duration"] = np.nan
    with pytest.raises(GP3BayesError, match="Missing values"):
        prepare_hierarchical_duration_data(
            data,
            _contract(),
            condition_levels=("control", "treatment"),
            missing="error",
        )
    prepared = prepare_hierarchical_duration_data(
        data,
        _contract(),
        condition_levels=("control", "treatment"),
        missing="drop",
    )
    assert prepared.n_input_rows == 96
    assert prepared.n_analysis_rows == 95
    assert prepared.rows_removed == 1
    assert prepared.transformations["missing"]["dropped_row_positions"] == (1,)


def test_duration_preparation_records_predictor_scaling():
    prepared = prepare_hierarchical_duration_data(
        _simulation().data,
        _contract(),
        condition_levels=("control", "treatment"),
        scale_predictors=("participant_covariate", "trial_covariate"),
    )
    assert math.isclose(float(prepared.data["participant_covariate"].mean()), 0.0, abs_tol=1e-12)
    assert math.isclose(float(prepared.data["trial_covariate"].std(ddof=1)), 1.0, abs_tol=1e-12)
    assert set(prepared.transformations["scaled_columns"]) == {
        "participant_covariate",
        "trial_covariate",
    }


def test_duration_specification_uses_approved_duration_priors():
    specification = specify_duration_model(
        _prepared(),
        baseline=500,
        intercept_scale=1.2,
        coefficient_scale=0.4,
        group_sd_scale=0.8,
        residual_scale=0.6,
        correlation_eta=3,
        student_df=4,
    )
    assert isinstance(specification, DurationModelSpecification)
    assert specification.family == "duration"
    assert specification.outcome_unit == "milliseconds"
    assert specification.priors.table["parameter_class"].tolist() == [
        "Intercept",
        "b",
        "sd",
        "sigma",
        "cor",
    ]
    assert specification.fit_performed is False
    assert specification.unrestricted_formula is False


def test_duration_specification_rejects_wrong_prepared_object():
    with pytest.raises(GP3BayesError, match="duration_prepared"):
        specify_duration_model(object(), baseline=500)  # type: ignore[arg-type]


def test_duration_prior_predictive_is_deterministic_and_backend_independent():
    specification = specify_duration_model(_prepared(), baseline=500)
    first = check_duration_prior_predictive(specification, draws=60, seed=2027)
    second = check_duration_prior_predictive(specification, draws=60, seed=2027)
    assert isinstance(first, DurationPriorPredictiveCheck)
    pd.testing.assert_frame_equal(first.summaries, second.summaries)
    pd.testing.assert_frame_equal(first.checks, second.checks)
    assert first.backend == "none"
    assert first.fitting_performed is False
    assert first.posterior_adequacy_established is False
    assert set(first.checks["status"]).issubset({"pass", "fail", "not_applicable"})


def test_duration_prior_predictive_condition_not_applicable_without_condition():
    simulation = simulate_hierarchical_duration_data(
        n_participants=8,
        trials_per_participant=6,
        include_items=False,
        random_slope_sd=0,
        seed=42,
    )
    data = simulation.data.drop(columns=["condition"])
    contract = create_model_contract(
        family="duration",
        outcome_col="duration",
        participant_col="participant_id",
        trial_col="trial_id",
        predictors=("participant_covariate", "trial_covariate"),
        outcome_unit="milliseconds",
    )
    prepared = prepare_hierarchical_duration_data(data, contract)
    specification = specify_duration_model(prepared, baseline=500)
    result = check_duration_prior_predictive(specification, draws=50, seed=12)
    row = result.checks[result.checks["check"] == "condition_median_ratio"].iloc[0]
    assert row["status"] == "not_applicable"
    assert math.isnan(float(row["violation_probability"]))


def test_duration_prior_predictive_validates_thresholds_and_draw_count():
    specification = specify_duration_model(_prepared(), baseline=500)
    with pytest.raises(GP3BayesError, match="draws"):
        check_duration_prior_predictive(specification, draws=49)
    with pytest.raises(GP3BayesError, match="plausible_median"):
        check_duration_prior_predictive(specification, draws=50, plausible_median=(500, 100))
    with pytest.raises(GP3BayesError, match="maximum_condition_ratio"):
        check_duration_prior_predictive(specification, draws=50, maximum_condition_ratio=1)


def test_duration_repr_states_conservative_boundaries():
    simulation = _simulation()
    prepared = _prepared()
    specification = specify_duration_model(prepared, baseline=500)
    prior_check = check_duration_prior_predictive(specification, draws=50, seed=20)
    assert "Censored: FALSE" in repr(simulation)
    assert "Fit performed: FALSE" in repr(prepared)
    assert "Family: lognormal" in repr(specification)
    assert "Posterior adequacy established: FALSE" in repr(prior_check)
