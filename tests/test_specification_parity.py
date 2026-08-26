from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from gp3bayespy import (
    GP3BayesError,
    audit_model_readiness,
    build_model_formula,
    create_model_contract,
    create_model_specification,
    create_prior_specification,
    validate_prior_specification,
)


def _data():
    return pd.DataFrame(
        {
            "participant_id": [f"p{i}" for i in range(1, 5) for _ in range(4)],
            "stimulus_id": [f"s{i}" for _ in range(4) for i in range(1, 5)],
            "trial_id": list(range(1, 5)) * 4,
            "condition": pd.Categorical(
                ["control", "treatment"] * 8,
                categories=["control", "treatment"],
            ),
            "trial_order": list(range(1, 5)) * 4,
            "age_z": [-1.0] * 4 + [-0.5] * 4 + [0.5] * 4 + [1.0] * 4,
            "selected": [0, 1, 1, 0] * 4,
            "response_time": [
                410,
                520,
                470,
                610,
                390,
                505,
                455,
                590,
                430,
                540,
                485,
                625,
                405,
                515,
                465,
                600,
            ],
        }
    )


def _binary_contract(random_slope=True):
    return create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="stimulus_id",
        trial_col="trial_id",
        condition_col="condition",
        time_col="trial_order",
        predictors="age_z",
        interaction=("condition", "age_z"),
        random_slope=random_slope,
    )


def _table_copy(priors):
    return priors.table.copy(deep=True)


def test_formula_contains_all_approved_terms_in_r_order():
    formula = build_model_formula(_binary_contract())
    assert formula == (
        "selected ~ condition + trial_order + age_z + condition:age_z + "
        "(1 + condition | participant_id) + (1 | stimulus_id)"
    )


def test_formula_quotes_non_syntactic_and_reserved_r_names():
    contract = create_model_contract(
        "binary",
        "selected outcome",
        "participant id",
        predictors=["age score", "if"],
    )
    assert build_model_formula(contract) == (
        "`selected outcome` ~ `age score` + `if` + (1 | `participant id`)"
    )


def test_binary_prior_schema_and_repr_match_reference():
    priors = create_prior_specification(
        _binary_contract(),
        baseline=0.25,
        intercept_scale=1.25,
        coefficient_scale=0.75,
        group_sd_scale=0.8,
        correlation_eta=3,
        student_df=4,
    )
    assert priors.transformed_baseline == pytest.approx(np.log(0.25 / 0.75))
    assert priors.table.parameter_class.tolist() == ["Intercept", "b", "sd", "cor"]
    assert priors.table.distribution.tolist() == ["normal", "normal", "student_t", "lkj"]
    assert priors.table.columns.tolist() == [
        "parameter_class",
        "distribution",
        "target",
        "location",
        "scale",
        "df",
        "shape",
        "lower",
        "upper",
        "rationale",
    ]
    assert np.isnan(priors.table.loc[3, "location"])
    assert np.isnan(priors.table.loc[3, "scale"])
    assert repr(priors).splitlines()[-3:] == [
        "  Parameter classes: Intercept, b, sd, cor",
        "  Backend: none",
        "  Executable: FALSE",
    ]


def test_duration_prior_schema_and_defaults_match_reference():
    contract = create_model_contract(
        "duration",
        "response_time",
        "participant_id",
        outcome_unit="milliseconds",
    )
    priors = create_prior_specification(contract, baseline=500, residual_scale=0.7)
    assert priors.transformed_baseline == pytest.approx(np.log(500))
    assert priors.table.parameter_class.tolist() == ["Intercept", "b", "sd", "sigma"]
    sigma = priors.table[priors.table.parameter_class == "sigma"].iloc[0]
    assert sigma["distribution"] == "student_t"
    assert sigma["scale"] == 0.7
    assert sigma["lower"] == 0.0
    assert "  Outcome unit: milliseconds" in repr(priors)


def test_numeric_prior_validation_uses_r_error_partition():
    binary = create_model_contract("binary", "y", "id")
    duration = create_model_contract(
        "duration", "y", "id", outcome_unit="milliseconds"
    )
    with pytest.raises(GP3BayesError, match="one finite numeric value"):
        create_prior_specification(binary, baseline="0.5")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError, match="strictly between zero and one"):
        create_prior_specification(binary, baseline=0)
    with pytest.raises(GP3BayesError, match="strictly positive"):
        create_prior_specification(duration, baseline=0)
    with pytest.raises(GP3BayesError, match="one finite numeric value"):
        create_prior_specification(duration, baseline=500, correlation_eta=np.inf)
    with pytest.raises(GP3BayesError, match="greater than or equal to one"):
        create_prior_specification(duration, baseline=500, correlation_eta=0.5)


def test_prior_validation_rejects_missing_extra_and_duplicate_classes():
    contract = _binary_contract()
    priors = create_prior_specification(contract)

    table = _table_copy(priors)
    table = table[table.parameter_class != "sd"].reset_index(drop=True)
    with pytest.raises(GP3BayesError, match="missing: sd"):
        validate_prior_specification(replace(priors, table=table), contract)

    table = _table_copy(priors)
    table = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    with pytest.raises(GP3BayesError, match="must be unique"):
        validate_prior_specification(replace(priors, table=table), contract)

    table = _table_copy(priors)
    table.loc[table.parameter_class == "b", "parameter_class"] = "shape"
    with pytest.raises(GP3BayesError, match="incomplete or unsupported"):
        validate_prior_specification(replace(priors, table=table), contract)


def test_prior_validation_rejects_distribution_and_domain_errors():
    contract = create_model_contract(
        "duration", "response_time", "participant_id", outcome_unit="milliseconds"
    )
    priors = create_prior_specification(contract, baseline=500)

    table = _table_copy(priors)
    table.loc[table.parameter_class == "sigma", "distribution"] = "normal"
    with pytest.raises(GP3BayesError, match="Incorrect prior distributions"):
        validate_prior_specification(replace(priors, table=table), contract)

    table = _table_copy(priors)
    table.loc[table.parameter_class == "b", "scale"] = 0
    with pytest.raises(GP3BayesError, match="strictly positive finite scales"):
        validate_prior_specification(replace(priors, table=table), contract)

    table = _table_copy(priors)
    table.loc[table.parameter_class == "sd", "scale"] = -1
    with pytest.raises(GP3BayesError, match="Half-Student-t priors"):
        validate_prior_specification(replace(priors, table=table), contract)

    table = _table_copy(priors)
    table.loc[table.parameter_class == "sigma", "df"] = 0
    with pytest.raises(GP3BayesError, match="Half-Student-t priors"):
        validate_prior_specification(replace(priors, table=table), contract)


def test_prior_validation_rejects_metadata_baseline_and_lkj_errors():
    contract = _binary_contract()
    priors = create_prior_specification(contract)

    table = _table_copy(priors)
    table.loc[0, "rationale"] = ""
    with pytest.raises(GP3BayesError, match="non-empty rationale"):
        validate_prior_specification(replace(priors, table=table), contract)

    with pytest.raises(GP3BayesError, match="inconsistent"):
        validate_prior_specification(
            replace(priors, transformed_baseline=2.0), contract
        )
    with pytest.raises(GP3BayesError, match='must be "none"'):
        validate_prior_specification(replace(priors, backend="brms"), contract)
    with pytest.raises(GP3BayesError, match="must be FALSE"):
        validate_prior_specification(replace(priors, executable=True), contract)

    table = _table_copy(priors)
    table.loc[table.parameter_class == "cor", "shape"] = 0.5
    with pytest.raises(GP3BayesError, match="LKJ priors require"):
        validate_prior_specification(replace(priors, table=table), contract)


def test_contract_incompatibility_names_the_incompatible_fields():
    binary = create_model_contract("binary", "selected", "participant_id")
    duration = create_model_contract(
        "duration", "duration", "participant_id", outcome_unit="milliseconds"
    )
    priors = create_prior_specification(binary)
    with pytest.raises(GP3BayesError) as error:
        validate_prior_specification(priors, duration)
    assert str(error.value) == (
        "`priors` is incompatible with `contract`: family, model_family, outcome_unit."
    )


def test_ready_audit_produces_complete_model_specification_and_repr():
    contract = _binary_contract()
    audit = audit_model_readiness(_data(), contract)
    priors = create_prior_specification(contract)
    specification = create_model_specification(contract, audit, priors)

    assert audit.ready is True
    assert audit.status == "ready"
    assert specification.family == "binary"
    assert specification.readiness_status == "ready"
    assert specification.warning_count == 0
    assert specification.formula == specification.formula_text
    assert specification.backend == "none"
    assert specification.fit_performed is False
    assert specification.contract == contract
    assert specification.audit == audit
    assert specification.priors == priors
    assert repr(specification).splitlines()[-3:] == [
        "  Prior classes: Intercept, b, sd, cor",
        "  Backend: none",
        "  Fit performed: FALSE",
    ]


def test_ready_with_warnings_can_proceed_but_not_ready_cannot():
    contract = create_model_contract("binary", "selected", "participant_id")
    warning_data = pd.DataFrame(
        {"participant_id": ["p1", "p1", "p2"], "selected": [0, 1, 0]}
    )
    warning_audit = audit_model_readiness(warning_data, contract)
    priors = create_prior_specification(contract)
    specification = create_model_specification(contract, warning_audit, priors)
    assert warning_audit.status == "ready_with_warnings"
    assert specification.warning_count > 0

    failed_data = pd.DataFrame(
        {"participant_id": ["p1", "p2"], "selected": [0, 1]}
    )
    failed_audit = audit_model_readiness(failed_data, contract)
    with pytest.raises(GP3BayesError, match="not ready for model specification"):
        create_model_specification(contract, failed_audit, priors)


def test_invalid_public_object_types_are_rejected():
    with pytest.raises(GP3BayesError, match="`contract` must inherit"):
        build_model_formula([])  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError, match="`priors` must inherit"):
        validate_prior_specification([])  # type: ignore[arg-type]

    contract = create_model_contract("binary", "selected", "participant_id")
    priors = create_prior_specification(contract)
    with pytest.raises(GP3BayesError, match="`audit` must inherit"):
        create_model_specification(contract, [], priors)  # type: ignore[arg-type]
