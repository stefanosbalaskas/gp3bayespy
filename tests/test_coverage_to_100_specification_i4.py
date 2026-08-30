from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

import gp3bayespy as gp
import gp3bayespy.specification as s
from gp3bayespy.exceptions import GP3BayesError


def _binary(seed: int = 3201, *, random_slope: bool = False):
    sim = gp.simulate_hierarchical_binary_data(
        n_participants=6,
        trials_per_participant=8,
        n_items=4,
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
    audit = gp.audit_model_readiness(sim.data, contract)
    priors = s.create_prior_specification(contract)
    return sim.data, contract, audit, priors


def test_specification_contract_and_audit_validation_matrix():
    _, contract, audit, _ = _binary()

    with pytest.raises(GP3BayesError):
        s._validate_specification_contract(replace(contract, mappings=[]))

    missing = dict(contract.mappings)
    missing.pop("trial")
    with pytest.raises(GP3BayesError):
        s._validate_specification_contract(replace(contract, mappings=missing))

    with pytest.raises(GP3BayesError):
        s._validate_specification_contract(
            replace(
                contract,
                template={
                    **dict(contract.template),
                    "prior_rationale": ("only", "three", "items"),
                },
            )
        )
    with pytest.raises(GP3BayesError):
        s._validate_specification_contract(
            replace(
                contract,
                template={
                    **dict(contract.template),
                    "prior_rationale": (
                        contract.prior_rationale[0],
                        "",
                        contract.prior_rationale[2],
                        contract.prior_rationale[3],
                    ),
                },
            )
        )
    with pytest.raises(GP3BayesError):
        s._validate_specification_contract(
            replace(contract, random_slope=1)  # type: ignore[arg-type]
        )

    no_condition = dict(contract.mappings)
    no_condition["condition"] = None
    with pytest.raises(GP3BayesError):
        s._validate_specification_contract(
            replace(
                contract,
                random_slope=True,
                mappings=no_condition,
            )
        )

    with pytest.raises(GP3BayesError):
        s._validate_specification_audit(
            replace(audit, ready=1)  # type: ignore[arg-type]
        )
    with pytest.raises(GP3BayesError):
        s._validate_specification_audit(replace(audit, status="mystery"))
    with pytest.raises(GP3BayesError):
        s._validate_specification_audit(replace(audit, status_counts={"pass": 1}))
    with pytest.raises(GP3BayesError):
        s._validate_specification_audit(
            replace(
                audit,
                status_counts={"pass": True, "warn": 0, "fail": 0},
            )
        )


def test_prior_validation_guard_matrix():
    _, contract, audit, priors = _binary(3210)

    with pytest.raises(GP3BayesError):
        s.validate_prior_specification(
            replace(priors, random_slope=1),  # type: ignore[arg-type]
            contract,
        )
    with pytest.raises(GP3BayesError):
        s.validate_prior_specification(
            replace(priors, backend="pymc"),
            contract,
        )
    with pytest.raises(GP3BayesError):
        s.validate_prior_specification(
            replace(priors, executable=True),
            contract,
        )

    incompatible = replace(contract, outcome_unit="seconds")
    with pytest.raises(GP3BayesError):
        s.validate_prior_specification(priors, incompatible)

    with pytest.raises(GP3BayesError):
        s.validate_prior_specification(
            replace(priors, transformed_baseline=1.0),
            contract,
        )
    with pytest.raises(GP3BayesError):
        s.validate_prior_specification(
            replace(priors, table="bad"),  # type: ignore[arg-type]
            contract,
        )

    missing_col = priors.table.drop(columns="rationale")
    with pytest.raises(GP3BayesError):
        s.validate_prior_specification(
            replace(priors, table=missing_col),
            contract,
        )

    with pytest.raises(GP3BayesError):
        s.validate_prior_specification(
            replace(priors, table=priors.table.iloc[0:0].copy()),
            contract,
        )

    duplicate = pd.concat(
        [priors.table, priors.table.iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(GP3BayesError):
        s.validate_prior_specification(
            replace(priors, table=duplicate),
            contract,
        )

    missing_class = priors.table.loc[priors.table["parameter_class"] != "sd"].copy()
    with pytest.raises(GP3BayesError):
        s.validate_prior_specification(
            replace(priors, table=missing_class),
            contract,
        )

    unsupported = priors.table.copy()
    unsupported.loc[len(unsupported)] = [
        "mystery",
        "normal",
        "x",
        0.0,
        1.0,
        float("nan"),
        float("nan"),
        float("-inf"),
        float("inf"),
        "r",
    ]
    with pytest.raises(GP3BayesError):
        s.validate_prior_specification(
            replace(priors, table=unsupported),
            contract,
        )

    wrong_dist = priors.table.copy()
    wrong_dist.loc[wrong_dist["parameter_class"] == "b", "distribution"] = "student_t"
    with pytest.raises(GP3BayesError):
        s.validate_prior_specification(
            replace(priors, table=wrong_dist),
            contract,
        )

    blank_target = priors.table.copy()
    blank_target.loc[0, "target"] = ""
    with pytest.raises(GP3BayesError):
        s.validate_prior_specification(
            replace(priors, table=blank_target),
            contract,
        )

    blank_rationale = priors.table.copy()
    blank_rationale.loc[0, "rationale"] = ""
    with pytest.raises(GP3BayesError):
        s.validate_prior_specification(
            replace(priors, table=blank_rationale),
            contract,
        )

    bad_normal = priors.table.copy()
    bad_normal.loc[bad_normal["distribution"] == "normal", "scale"] = 0.0
    with pytest.raises(GP3BayesError):
        s.validate_prior_specification(
            replace(priors, table=bad_normal),
            contract,
        )

    bad_student = priors.table.copy()
    bad_student.loc[bad_student["distribution"] == "student_t", "df"] = 0.0
    with pytest.raises(GP3BayesError):
        s.validate_prior_specification(
            replace(priors, table=bad_student),
            contract,
        )

    _, slope_contract, _, slope_priors = _binary(
        3215,
        random_slope=True,
    )
    bad_lkj = slope_priors.table.copy()
    bad_lkj.loc[bad_lkj["distribution"] == "lkj", "shape"] = 0.5
    with pytest.raises(GP3BayesError):
        s.validate_prior_specification(
            replace(slope_priors, table=bad_lkj),
            slope_contract,
        )

    other_audit = replace(
        audit, contract=replace(contract, interaction=("condition", "trial_covariate"))
    )
    with pytest.raises(GP3BayesError):
        s.create_model_specification(contract, other_audit, priors)

    not_ready = replace(
        audit,
        ready=False,
        status="not_ready",
    )
    with pytest.raises(GP3BayesError):
        s.create_model_specification(contract, not_ready, priors)
