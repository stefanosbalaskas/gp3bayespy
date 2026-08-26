import numpy as np
import pandas as pd

from gp3bayespy import audit_model_readiness, create_model_contract


def _binary_contract(**kwargs):
    values = {
        "family": "binary",
        "outcome_col": "selected",
        "participant_col": "participant_id",
        "trial_col": "trial_id",
        "condition_col": "condition",
    }
    values.update(kwargs)
    return create_model_contract(**values)


def _balanced_binary():
    return pd.DataFrame(
        {
            "participant_id": ["p1"] * 4 + ["p2"] * 4,
            "trial_id": [1, 2, 3, 4] * 2,
            "condition": ["control", "treatment"] * 4,
            "selected": [0, 1, 0, 1, 1, 0, 1, 0],
        }
    )


def _check(audit, check_id):
    rows = audit.checks[audit.checks["check_id"] == check_id]
    assert len(rows) == 1
    return rows.iloc[0]


def test_readiness_exact_core_ids_and_messages():
    audit = audit_model_readiness(_balanced_binary(), _binary_contract())
    assert _check(audit, "participant_levels")["category"] == "repeated_measures"
    assert _check(audit, "participant_levels")["message"] == (
        "2 participants are observed."
    )
    assert _check(audit, "repeated_measurement")["message"] == (
        "Every participant contributes repeated observations."
    )
    assert _check(audit, "trial_key")["message"] == (
        "Participant-trial identifiers are unique."
    )
    assert _check(audit, "condition_levels")["message"] == (
        "2 condition levels are observed."
    )
    assert _check(audit, "random_slope_support")["message"] == (
        "No participant-level random slope was requested."
    )


def test_single_observation_participants_block_when_none_repeat():
    data = pd.DataFrame(
        {
            "participant_id": ["p1", "p2"],
            "trial_id": [1, 1],
            "condition": ["control", "treatment"],
            "selected": [0, 1],
        }
    )
    audit = audit_model_readiness(data, _binary_contract())
    row = _check(audit, "repeated_measurement")
    assert row["status"] == "fail"
    assert row["message"] == "No participant contributes repeated observations."


def test_mixed_repetition_is_warning():
    data = _balanced_binary().iloc[:5].copy()
    data.loc[:, "trial_id"] = [1, 2, 3, 4, 1]
    audit = audit_model_readiness(data, _binary_contract())
    row = _check(audit, "repeated_measurement")
    assert row["status"] == "warn"
    assert row["n_affected"] == 1


def test_declared_item_with_one_level_is_failure():
    data = _balanced_binary()
    data["item_id"] = "i1"
    contract = _binary_contract(item_col="item_id")
    audit = audit_model_readiness(data, contract)
    row = _check(audit, "item_levels")
    assert row["status"] == "fail"
    assert row["message"] == (
        "At least two items must be observed when an item is declared."
    )


def test_item_crossing_warning_is_recorded():
    data = _balanced_binary()
    data["item_id"] = ["i1", "i1", "i2", "i2", "i1", "i1", "i1", "i1"]
    contract = _binary_contract(item_col="item_id")
    audit = audit_model_readiness(data, contract)
    row = _check(audit, "item_crossing")
    assert row["status"] == "warn"
    assert row["n_affected"] >= 1


def test_duplicate_participant_trial_key_is_failure():
    data = _balanced_binary()
    data.loc[1, "trial_id"] = 1
    audit = audit_model_readiness(data, _binary_contract())
    row = _check(audit, "trial_key")
    assert row["status"] == "fail"
    assert row["n_affected"] == 2


def test_condition_accepts_more_than_two_observed_levels():
    data = _balanced_binary()
    data.loc[0:1, "condition"] = "third"
    audit = audit_model_readiness(data, _binary_contract())
    row = _check(audit, "condition_levels")
    assert row["status"] == "pass"
    assert row["message"] == "3 condition levels are observed."


def test_random_slope_replication_warning():
    data = _balanced_binary().iloc[[0, 1, 4, 5]].copy()
    data.loc[:, "trial_id"] = [1, 2, 1, 2]
    contract = _binary_contract(random_slope=True)
    audit = audit_model_readiness(data, contract)
    assert _check(audit, "random_slope_support")["status"] == "pass"
    row = _check(audit, "random_slope_replication")
    assert row["status"] == "warn"
    assert row["n_affected"] == 4


def test_time_type_finite_variation_and_within_participant():
    data = _balanced_binary()
    data["time"] = [1, 2, 3, 4, 1, 1, 1, 1]
    contract = _binary_contract(time_col="time")
    audit = audit_model_readiness(data, contract)
    assert _check(audit, "time_type")["status"] == "pass"
    assert _check(audit, "time_finite")["status"] == "pass"
    assert _check(audit, "time_variation")["status"] == "pass"
    row = _check(audit, "time_within_participant")
    assert row["status"] == "warn"
    assert row["n_affected"] == 1


def test_predictor_finite_blank_and_unused_levels_checks():
    data = _balanced_binary()
    data["x"] = [1.0, 2.0, np.inf, 4.0, 5.0, 6.0, 7.0, 8.0]
    data["label"] = pd.Categorical(
        ["a", "b", "", "a", "b", "a", "b", "a"],
        categories=["a", "b", "", "unused"],
    )
    contract = _binary_contract(predictors=["x", "label"])
    audit = audit_model_readiness(data, contract)
    assert _check(audit, "predictor_finite")["status"] == "fail"
    assert _check(audit, "predictor_blanks")["status"] == "fail"
    assert _check(audit, "predictor_factor_levels")["status"] == "warn"


def test_categorical_interaction_singleton_combination_warns():
    data = _balanced_binary()
    data["group"] = ["a", "a", "a", "a", "a", "a", "a", "b"]
    contract = _binary_contract(
        predictors=["group"],
        interaction=["condition", "group"],
    )
    audit = audit_model_readiness(data, contract)
    row = _check(audit, "interaction_support")
    assert row["status"] == "warn"
    assert row["message"].endswith(
        "categorical interaction combinations contain one row."
    )


def test_nonfinite_time_is_failure():
    data = _balanced_binary()
    data["time"] = [1, 2, 3, np.inf, 1, 2, 3, 4]
    contract = _binary_contract(time_col="time")
    audit = audit_model_readiness(data, contract)
    row = _check(audit, "time_finite")
    assert row["status"] == "fail"
    assert row["n_affected"] == 1
