import pytest

from gp3bayespy import GP3BayesError, create_model_contract


def test_binary_contract_basics():
    contract = create_model_contract(
        "binary",
        "selected",
        "participant_id",
        item_col="stimulus_id",
        condition_col="condition",
    )
    assert contract.family == "binary"
    assert contract.model_family == "hierarchical_binary"
    assert contract.likelihood == "Bernoulli"
    assert contract.link == "logit"
    assert contract.mappings["item"] == "stimulus_id"


def test_duration_requires_unit():
    with pytest.raises(GP3BayesError, match="outcome_unit"):
        create_model_contract("duration", "duration", "participant_id")


def test_random_slope_requires_condition():
    with pytest.raises(GP3BayesError, match="condition_col"):
        create_model_contract("binary", "y", "id", random_slope=True)


def test_duplicate_mapping_rejected():
    with pytest.raises(GP3BayesError, match="Duplicated"):
        create_model_contract("binary", "y", "id", predictors=["y"])
