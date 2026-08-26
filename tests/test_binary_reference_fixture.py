import inspect
import json
from pathlib import Path

import gp3bayespy as gp

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "dev" / "parity" / "binary_foundation_reference_0.5.0.json"


def test_binary_foundation_fixture_is_frozen_to_r_050():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["reference_package"] == "gp3bayes"
    assert payload["reference_version"] == "0.5.0"
    assert set(payload["exports"]) == {
        "simulate_hierarchical_binary_data",
        "prepare_hierarchical_binary_data",
        "specify_binary_model",
        "check_binary_prior_predictive",
    }


def test_binary_simulation_python_signature_preserves_r_argument_order():
    parameters = list(inspect.signature(gp.simulate_hierarchical_binary_data).parameters)
    assert parameters == [
        "n_participants",
        "trials_per_participant",
        "n_items",
        "intercept",
        "condition_effect",
        "participant_covariate_effect",
        "trial_covariate_effect",
        "interaction_effect",
        "participant_sd",
        "item_sd",
        "random_slope_sd",
        "random_slope_cor",
        "condition_probability",
        "balanced_condition",
        "include_items",
        "seed",
    ]


def test_binary_preparation_and_specification_argument_order_is_frozen():
    prepare = list(inspect.signature(gp.prepare_hierarchical_binary_data).parameters)
    specify = list(inspect.signature(gp.specify_binary_model).parameters)
    assert prepare == [
        "data",
        "contract",
        "outcome_mapping",
        "condition_levels",
        "condition_coding",
        "scale_predictors",
        "scale_time",
        "missing",
    ]
    assert specify == [
        "prepared",
        "baseline",
        "intercept_scale",
        "coefficient_scale",
        "group_sd_scale",
        "correlation_eta",
        "student_df",
    ]


def test_binary_prior_predictive_argument_order_is_frozen():
    parameters = list(inspect.signature(gp.check_binary_prior_predictive).parameters)
    assert parameters == [
        "specification",
        "draws",
        "seed",
        "plausible_rate",
        "boundary_probability",
        "extreme_contrast",
        "maximum_degenerate_participant_fraction",
        "maximum_boundary_mass",
        "maximum_extreme_probability",
    ]
