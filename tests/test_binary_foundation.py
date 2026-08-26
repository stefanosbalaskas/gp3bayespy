import math

import numpy as np
import pandas as pd
import pytest

from gp3bayespy import (
    BinaryModelSpecification,
    BinaryPrepared,
    BinaryPriorPredictiveCheck,
    BinarySimulation,
    GP3BayesError,
    ModelSpecification,
    check_binary_prior_predictive,
    create_model_contract,
    prepare_hierarchical_binary_data,
    simulate_hierarchical_binary_data,
    specify_binary_model,
)


def _simulation(seed: int = 2026) -> BinarySimulation:
    return simulate_hierarchical_binary_data(
        n_participants=12,
        trials_per_participant=8,
        n_items=6,
        random_slope_sd=0.25,
        seed=seed,
    )


def _contract(
    *,
    random_slope: bool = True,
    include_item: bool = True,
    include_condition: bool = True,
):
    return create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="item_id" if include_item else None,
        trial_col="trial_id",
        condition_col="condition" if include_condition else None,
        predictors=("participant_covariate", "trial_covariate"),
        interaction=("condition", "participant_covariate") if include_condition else None,
        random_slope=random_slope and include_condition,
    )


def _prepared(
    *,
    seed: int = 2026,
    random_slope: bool = True,
    include_item: bool = True,
    include_condition: bool = True,
) -> BinaryPrepared:
    return prepare_hierarchical_binary_data(
        _simulation(seed).data,
        _contract(
            random_slope=random_slope,
            include_item=include_item,
            include_condition=include_condition,
        ),
        condition_levels=("control", "treatment") if include_condition else None,
    )


def test_binary_simulation_is_deterministic_and_records_truth():
    first = _simulation(2026)
    second = _simulation(2026)

    assert isinstance(first, BinarySimulation)
    pd.testing.assert_frame_equal(first.data, second.data)
    assert first.truth == second.truth
    pd.testing.assert_frame_equal(
        first.random_effects["participant"], second.random_effects["participant"]
    )
    pd.testing.assert_frame_equal(first.random_effects["item"], second.random_effects["item"])
    assert len(first.data) == 96
    assert first.data["participant_id"].nunique() == 12
    assert first.data["item_id"].nunique() == 6
    assert list(first.data["condition"].cat.categories) == ["control", "treatment"]
    assert set(first.data["selected"].unique()).issubset({0, 1})
    assert first.data["true_probability"].between(0, 1, inclusive="neither").all()
    assert first.truth["seed"] == 2026
    assert first.truth["condition_coding"] == {"control": -0.5, "treatment": 0.5}


def test_binary_simulation_supports_design_without_items_or_slopes():
    simulation = simulate_hierarchical_binary_data(
        n_participants=10,
        trials_per_participant=6,
        include_items=False,
        random_slope_sd=0,
        seed=18,
    )
    assert "item_id" not in simulation.data
    assert simulation.random_effects["item"] is None
    assert simulation.design["n_items"] == 0
    assert simulation.design["random_slope"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_participants": 1}, "n_participants"),
        ({"trials_per_participant": 1}, "trials_per_participant"),
        ({"participant_sd": -1}, "participant_sd"),
        ({"random_slope_cor": 1}, "random_slope_cor"),
        ({"condition_probability": 0}, "condition_probability"),
    ],
)
def test_binary_simulation_validates_design_arguments(kwargs, message):
    with pytest.raises(GP3BayesError, match=message):
        simulate_hierarchical_binary_data(**kwargs)


def test_binary_preparation_codes_outcome_condition_and_scaling():
    simulation = _simulation()
    prepared = prepare_hierarchical_binary_data(
        simulation.data,
        _contract(),
        condition_levels=("control", "treatment"),
        scale_predictors=("participant_covariate", "trial_covariate"),
    )

    assert isinstance(prepared, BinaryPrepared)
    assert prepared.audit.ready
    assert prepared.audit.status in {"ready", "ready_with_warnings"}
    assert set(prepared.data["selected"].unique()).issubset({0, 1})
    np.testing.assert_allclose(sorted(prepared.data["condition"].unique()), [-0.5, 0.5])
    assert math.isclose(float(prepared.data["participant_covariate"].mean()), 0.0, abs_tol=1e-12)
    assert math.isclose(
        float(prepared.data["participant_covariate"].std(ddof=1)),
        1.0,
        abs_tol=1e-12,
    )
    assert math.isclose(float(prepared.data["trial_covariate"].mean()), 0.0, abs_tol=1e-12)
    assert math.isclose(
        float(prepared.data["trial_covariate"].std(ddof=1)),
        1.0,
        abs_tol=1e-12,
    )
    assert prepared.n_input_rows == 96
    assert prepared.n_analysis_rows == 96
    assert prepared.rows_removed == 0
    assert prepared.contains_data is True
    assert prepared.backend == "none"
    assert prepared.fit_performed is False
    expected = {
        "(Intercept)",
        "condition",
        "participant_covariate",
        "trial_covariate",
        "condition:participant_covariate",
    }
    assert expected.issubset(set(prepared.model_matrix_columns))


def test_labelled_outcomes_require_and_preserve_explicit_mapping():
    data = _simulation().data.copy()
    data["selected"] = pd.Categorical(
        np.where(data["selected"].eq(1), "yes", "no"),
        categories=["no", "yes"],
    )
    contract = _contract()
    with pytest.raises(GP3BayesError, match="outcome_mapping"):
        prepare_hierarchical_binary_data(
            data,
            contract,
            condition_levels=("control", "treatment"),
        )

    prepared = prepare_hierarchical_binary_data(
        data,
        contract,
        outcome_mapping={"no": 0, "yes": 1},
        condition_levels=("control", "treatment"),
    )
    assert set(prepared.data["selected"].unique()).issubset({0, 1})
    assert prepared.transformations["outcome"]["mapping"] == {"no": 0, "yes": 1}


def test_missing_row_decisions_are_explicit_and_reproducible():
    data = _simulation().data.copy()
    data.loc[data.index[0], "trial_covariate"] = np.nan
    contract = _contract()
    with pytest.raises(GP3BayesError, match="Missing values"):
        prepare_hierarchical_binary_data(
            data,
            contract,
            condition_levels=("control", "treatment"),
            missing="error",
        )

    prepared = prepare_hierarchical_binary_data(
        data,
        contract,
        condition_levels=("control", "treatment"),
        missing="drop",
    )
    assert prepared.n_input_rows == 96
    assert prepared.n_analysis_rows == 95
    assert prepared.rows_removed == 1
    assert prepared.transformations["missing"]["dropped_row_positions"] == (1,)
    assert prepared.transformations["missing"]["action"] == "drop"


def test_preparation_rejects_undeclared_scaling_and_non_binary_contracts():
    simulation = _simulation()
    with pytest.raises(GP3BayesError, match="Undeclared"):
        prepare_hierarchical_binary_data(
            simulation.data,
            _contract(),
            condition_levels=("control", "treatment"),
            scale_predictors="true_probability",
        )

    duration_contract = create_model_contract(
        family="duration",
        outcome_col="true_probability",
        participant_col="participant_id",
        outcome_unit="probability_unit",
    )
    with pytest.raises(GP3BayesError, match="binary model contract"):
        prepare_hierarchical_binary_data(simulation.data, duration_contract)


def test_binary_specification_extends_validated_core_without_fitting():
    prepared = _prepared()
    specification = specify_binary_model(
        prepared,
        baseline=0.35,
        intercept_scale=1.25,
        coefficient_scale=0.7,
        group_sd_scale=0.8,
        correlation_eta=3,
        student_df=4,
    )
    assert isinstance(specification, BinaryModelSpecification)
    assert isinstance(specification, ModelSpecification)
    assert specification.family == "binary"
    assert specification.prepared is prepared
    assert "(1 + condition | participant_id)" in specification.formula_text
    assert "(1 | item_id)" in specification.formula_text
    assert specification.priors.baseline == 0.35
    assert math.isclose(
        specification.priors.transformed_baseline,
        math.log(0.35 / 0.65),
    )
    assert specification.fitting_engine == "none"
    assert specification.backend_dependency == "none"
    assert specification.unrestricted_formula is False
    assert specification.fit_performed is False
    assert specification.prior_predictive_performed is False


def test_prior_predictive_checks_are_deterministic_and_backend_independent():
    specification = specify_binary_model(_prepared(), baseline=0.35)
    first = check_binary_prior_predictive(specification, draws=60, seed=2027)
    second = check_binary_prior_predictive(specification, draws=60, seed=2027)

    assert isinstance(first, BinaryPriorPredictiveCheck)
    pd.testing.assert_frame_equal(first.summaries, second.summaries)
    pd.testing.assert_frame_equal(first.checks, second.checks)
    assert first.draws == 60
    assert first.seed == 2027
    assert first.backend == "none"
    assert first.fitting_performed is False
    expected = {
        "overall_rate",
        "condition_low_rate",
        "condition_high_rate",
        "condition_rate_contrast",
        "participant_rate_sd",
        "item_rate_sd",
        "participant_all_zero",
        "participant_all_one",
        "probability_below_boundary",
        "probability_above_boundary",
    }
    assert expected.issubset(first.summaries.columns)
    assert set(first.checks["status"]).issubset({"pass", "fail", "not_applicable"})
    assert isinstance(first.adequate, bool)


def test_condition_checks_are_not_applicable_without_condition():
    specification = specify_binary_model(
        _prepared(random_slope=False, include_item=False, include_condition=False),
        baseline=0.35,
    )
    check = check_binary_prior_predictive(specification, draws=50, seed=2027)
    condition_rows = check.checks["check"].isin(["condition_rates", "condition_contrast"])
    assert (check.checks.loc[condition_rows, "status"] == "not_applicable").all()
    assert check.checks.loc[condition_rows, "probability"].isna().all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"draws": 49}, "draws"),
        ({"plausible_rate": (0.9, 0.1)}, "plausible_rate"),
        ({"maximum_extreme_probability": 2}, "maximum_extreme_probability"),
    ],
)
def test_prior_predictive_controls_are_validated(kwargs, message):
    specification = specify_binary_model(_prepared())
    with pytest.raises(GP3BayesError, match=message):
        check_binary_prior_predictive(specification, **kwargs)


def test_binary_workflow_representations_are_concise():
    simulation = _simulation()
    prepared = _prepared()
    specification = specify_binary_model(prepared)
    check = check_binary_prior_predictive(specification, draws=50, seed=2027)
    assert "<gp3bayes_binary_simulation>" in repr(simulation)
    assert "Fit performed: FALSE" in repr(prepared)
    assert "Fitting engine: none" in repr(specification)
    assert "Fit performed: FALSE" in repr(check)
