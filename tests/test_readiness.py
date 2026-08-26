import pandas as pd

from gp3bayespy import (
    audit_model_readiness,
    create_model_contract,
    create_model_specification,
    create_prior_specification,
)


def data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": ["p1"] * 4 + ["p2"] * 4,
            "trial_id": [1, 2, 3, 4] * 2,
            "condition": ["control", "treatment"] * 4,
            "selected": [0, 1, 0, 1, 1, 0, 1, 0],
        }
    )


def test_ready_binary_example():
    contract = create_model_contract(
        "binary",
        "selected",
        "participant_id",
        trial_col="trial_id",
        condition_col="condition",
    )
    audit = audit_model_readiness(data(), contract)
    assert audit.ready
    assert audit.status in {"ready", "ready_with_warnings"}
    assert audit.status_counts["fail"] == 0


def test_missing_column_blocks():
    contract = create_model_contract(
        "binary",
        "selected",
        "participant_id",
        predictors=["missing_x"],
    )
    audit = audit_model_readiness(data(), contract)
    assert not audit.ready
    assert "missing_x" in audit.columns["missing"]


def test_specification_gate():
    contract = create_model_contract(
        "binary",
        "selected",
        "participant_id",
        trial_col="trial_id",
        condition_col="condition",
    )
    audit = audit_model_readiness(data(), contract)
    priors = create_prior_specification(contract, baseline=0.5)
    specification = create_model_specification(contract, audit, priors)
    assert specification.fit_performed is False
    assert specification.backend == "none"
