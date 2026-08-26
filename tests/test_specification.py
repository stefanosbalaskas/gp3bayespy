from gp3bayespy import build_model_formula, create_model_contract, create_prior_specification


def test_binary_formula_matches_r_contract_shape():
    contract = create_model_contract(
        "binary",
        "selected",
        "participant_id",
        item_col="stimulus_id",
        condition_col="condition",
        predictors=["age_z"],
    )
    expected = (
        "selected ~ condition + age_z + (1 | participant_id) + (1 | stimulus_id)"
    )
    assert build_model_formula(contract) == expected


def test_random_slope_formula():
    contract = create_model_contract(
        "binary",
        "selected",
        "participant_id",
        condition_col="condition",
        random_slope=True,
    )
    assert build_model_formula(contract) == (
        "selected ~ condition + (1 + condition | participant_id)"
    )


def test_binary_default_prior_classes():
    contract = create_model_contract("binary", "selected", "participant_id")
    priors = create_prior_specification(contract, baseline=0.35)
    assert priors.table.parameter_class.tolist() == ["Intercept", "b", "sd"]
    assert priors.backend == "none"


def test_duration_prior_has_sigma():
    contract = create_model_contract(
        "duration",
        "duration",
        "participant_id",
        outcome_unit="milliseconds",
    )
    priors = create_prior_specification(contract, baseline=500)
    assert priors.table.parameter_class.tolist() == ["Intercept", "b", "sd", "sigma"]
