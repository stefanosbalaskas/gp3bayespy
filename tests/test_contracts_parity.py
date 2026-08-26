import pytest

from gp3bayespy import GP3BayesError, create_model_contract


def test_binary_contract_preserves_approved_metadata():
    contract = create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="stimulus_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors="age_z",
        interaction=("condition", "age_z"),
        random_slope=True,
    )
    assert contract.family == "binary"
    assert contract.model_family == "hierarchical_binary"
    assert contract.likelihood == "Bernoulli"
    assert contract.link == "logit"
    assert contract.mappings["outcome"] == "selected"
    assert contract.interaction == ("condition", "age_z")
    assert contract.random_slope is True
    assert contract.outcome_unit is None


def test_duration_contract_requires_and_preserves_unit():
    contract = create_model_contract(
        "duration",
        "response_time",
        "participant_id",
        trial_col="trial_id",
        condition_col="condition",
        outcome_unit="milliseconds",
    )
    assert contract.model_family == "hierarchical_lognormal_duration"
    assert contract.likelihood == "lognormal"
    assert contract.outcome_unit == "milliseconds"


def test_contract_validation_messages_match_r_reference():
    with pytest.raises(
        GP3BayesError,
        match=r"^`family` must be one non-missing character value\.$",
    ):
        create_model_contract(None, "y", "id")  # type: ignore[arg-type]

    with pytest.raises(
        GP3BayesError,
        match=r"^Unsupported `family`: count\. Supported values are: binary, duration\.$",
    ):
        create_model_contract("count", "events", "participant_id")

    with pytest.raises(
        GP3BayesError,
        match=r"^`outcome_col` must be one non-empty character value\.$",
    ):
        create_model_contract("binary", "", "participant_id")

    with pytest.raises(
        GP3BayesError,
        match=r"^`random_slope` must be TRUE or FALSE\.$",
    ):
        create_model_contract(
            "binary", "y", "id", random_slope=1  # type: ignore[arg-type]
        )


def test_character_vectors_require_unique_nonempty_values():
    with pytest.raises(GP3BayesError, match="character vector of unique"):
        create_model_contract(
            "binary", "y", "id", predictors=None  # type: ignore[arg-type]
        )
    with pytest.raises(GP3BayesError, match="character vector of unique"):
        create_model_contract("binary", "y", "id", predictors=["x", "x"])
    with pytest.raises(GP3BayesError, match="character vector of unique"):
        create_model_contract("binary", "y", "id", notes=[""])


def test_duplicate_column_message_preserves_r_encounter_order():
    with pytest.raises(GP3BayesError) as error:
        create_model_contract(
            "binary",
            "z",
            "a",
            item_col="b",
            predictors=["b", "a"],
        )
    assert str(error.value) == (
        "Column mappings and predictors must be unique. Duplicated: b, a."
    )


def test_interaction_and_outcome_unit_rules_match_reference():
    with pytest.raises(
        GP3BayesError,
        match=r"^`interaction` must contain exactly two declared variables\.$",
    ):
        create_model_contract(
            "binary",
            "y",
            "id",
            predictors=["x", "z"],
            interaction=["x", "z", "q"],
        )

    with pytest.raises(GP3BayesError, match="Every interaction variable"):
        create_model_contract(
            "binary",
            "y",
            "id",
            predictors=["x"],
            interaction=["x", "undeclared"],
        )

    with pytest.raises(GP3BayesError, match="condition_col"):
        create_model_contract("binary", "y", "id", random_slope=True)

    with pytest.raises(GP3BayesError, match="must be NULL"):
        create_model_contract(
            "binary", "y", "id", outcome_unit="probability"
        )

    with pytest.raises(GP3BayesError, match="one non-empty character"):
        create_model_contract("duration", "y", "id")


def test_contract_templates_and_repr_are_complete():
    binary = create_model_contract("binary", "selected", "participant_id")
    duration = create_model_contract(
        "duration",
        "response_time",
        "participant_id",
        outcome_unit="milliseconds",
    )
    for contract in (binary, duration):
        assert len(contract.repeated_measures_structure) == 3
        assert len(contract.supported_predictors) == 3
        assert isinstance(contract.supported_interactions, str)
        assert contract.supported_offsets_or_exposures == "Not supported"
        assert len(contract.prior_rationale) == 4
        assert len(contract.interpretation_boundaries) >= 3

    assert binary.supported_censoring == "Not applicable"
    assert "uncensored" in duration.supported_censoring
    assert repr(duration).splitlines() == [
        "<gp3bayes_model_contract>",
        "  Family: duration",
        "  Likelihood: lognormal",
        "  Link: identity on mean log duration",
        "  Outcome: response_time",
        "  Participant: participant_id",
        "  Outcome unit: milliseconds",
        "  Random slope requested: FALSE",
        "  Fitting performed: FALSE",
    ]
