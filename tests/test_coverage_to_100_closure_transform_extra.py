from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
import gp3bayespy.specification_closure as c
from gp3bayespy.exceptions import GP3BayesError


def _binary_objects(seed: int = 1701):
    sim = gp.simulate_hierarchical_binary_data(
        n_participants=5,
        trials_per_participant=6,
        n_items=3,
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
        interaction=("condition", "participant_covariate"),
        random_slope=False,
    )
    prepared = gp.prepare_hierarchical_binary_data(
        sim.data,
        contract,
        scale_predictors=("participant_covariate", "trial_covariate"),
    )
    spec = gp.specify_binary_model(prepared)
    return sim, contract, prepared, spec


def _duration_objects(seed: int = 1702):
    sim = gp.simulate_hierarchical_duration_data(
        n_participants=5,
        trials_per_participant=6,
        n_items=3,
        seed=seed,
    )
    contract = gp.create_model_contract(
        family="duration",
        outcome_col="duration",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=("participant_covariate", "trial_covariate"),
        interaction=("condition", "participant_covariate"),
        random_slope=False,
        outcome_unit="milliseconds",
    )
    prepared = gp.prepare_hierarchical_duration_data(
        sim.data,
        contract,
        scale_predictors=("participant_covariate", "trial_covariate"),
    )
    spec = gp.specify_duration_model(prepared, baseline=500.0)
    return sim, contract, prepared, spec


def test_identifier_like_predictor_review_numeric_and_nonnumeric():
    data = pd.DataFrame(
        {
            "y": [0, 1, 0, 1],
            "p": ["p1", "p1", "p2", "p2"],
            "row_id": [1, 2, 3, 4],
            "label": ["a", "b", "a", "b"],
        }
    )
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="y",
        participant_col="p",
        predictors=("row_id", "label"),
    )
    audit = c.identify_identifier_like_predictors(data, contract)
    assert "row_id" in audit.flagged
    label_row = audit.table.loc[audit.table["predictor"] == "label"].iloc[0]
    assert not bool(label_row["numeric"])
    assert label_row["reason"] == "non_numeric"

    with pytest.raises(GP3BayesError):
        c.identify_identifier_like_predictors(data.drop(columns="row_id"), contract)
    boundary = c.identify_identifier_like_predictors(
        data,
        contract,
        unique_fraction=0,
    )
    assert isinstance(boundary.flagged, tuple)


def test_duration_extremes_boundary_and_censor_paths():
    sim, contract, _, _ = _duration_objects()
    data = sim.data.copy()

    normal = c.review_duration_extremes(data, contract, mad_cutoff=20, iqr_multiplier=20)
    assert normal.n == len(data)

    outlier = data.copy()
    outlier.loc[outlier.index[0], "duration"] *= 1000
    reviewed = c.review_duration_extremes(outlier, contract)
    assert reviewed.n_flagged >= 1
    assert reviewed.status == "review"

    invalid = data.copy()
    invalid.loc[0, "duration"] = 0
    with pytest.raises(GP3BayesError):
        c.review_duration_extremes(invalid, contract)

    assert c._censor_flags(pd.Series([True, False])).tolist() == [True, False]
    assert c._censor_flags(pd.Series([0, 1, 2])).tolist() == [False, True, True]
    assert c._censor_flags(pd.Series(["no", "censored", "TRUE"])).tolist() == [
        False,
        True,
        True,
    ]

    base = data.copy()
    base["deadline_flag"] = 0
    candidate = c.audit_duration_boundaries(base, contract)
    assert "deadline_flag" in candidate.candidate_columns
    assert candidate.status == "review"

    ranged = c.audit_duration_boundaries(
        outlier,
        contract,
        allowed_range=(1.0, float(data["duration"].max()) * 2),
    )
    assert len(ranged.range_violations) >= 1

    censored = data.copy()
    censored["censored"] = [1] + [0] * (len(censored) - 1)
    audit = c.audit_duration_boundaries(
        censored,
        contract,
        censor_col="censored",
        detect_candidate_columns=False,
    )
    assert audit.censored_rows == (1,)
    assert audit.status == "fail"

    no_detect = c.audit_duration_boundaries(
        data,
        contract,
        detect_candidate_columns=False,
    )
    assert "not_applicable" in set(no_detect.checks["status"])

    with pytest.raises(GP3BayesError):
        c.audit_duration_boundaries(data, contract, allowed_range=(2, 1))
    with pytest.raises(GP3BayesError):
        c.audit_duration_boundaries(data, contract, censor_col="missing")


def test_strict_readiness_binary_and_duration_paths():
    _, bcontract, bprepared, _ = _binary_objects(1710)
    binary = c.audit_model_readiness_strict(
        bprepared.data,
        bcontract,
        run_separation=False,
    )
    assert binary.family == "binary"
    assert binary.binary_group_variation is not None
    assert binary.duration_extremes is None

    _, dcontract, dprepared, _ = _duration_objects(1711)
    data = dprepared.data.copy()
    data["censor_marker"] = 0
    duration = c.audit_model_readiness_strict(
        data,
        dcontract,
        duration_allowed_range=(
            max(1e-6, float(data["duration"].min()) * 0.5),
            float(data["duration"].max()) * 2,
        ),
        censor_col="censor_marker",
    )
    assert duration.family == "duration"
    assert duration.duration_extremes is not None
    assert duration.duration_boundaries is not None


def test_binary_transformation_recipe_roundtrip_and_validation():
    sim, _, prepared, _ = _binary_objects(1720)
    recipe = c.create_transformation_recipe(prepared)
    assert recipe.family == "binary"
    assert c._as_recipe(recipe) is recipe
    assert c._as_recipe(prepared).family == "binary"

    raw = c.invert_transformation_recipe(prepared.data, recipe)
    replay = c.apply_transformation_recipe(
        raw,
        recipe,
        input_scale="raw",
        require_outcome=True,
    )
    assert len(replay) == len(prepared.data)
    assert "gp3bayes_transformation_recipe" in replay.attrs

    prepared_copy = c.apply_transformation_recipe(
        prepared.data,
        recipe,
        input_scale="prepared",
    )
    assert "gp3bayes_transformation_recipe" in prepared_copy.attrs

    audit = c.validate_transformation_replay(prepared)
    assert audit.replay_established

    with pytest.raises(GP3BayesError):
        c.create_transformation_recipe(object())
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe([], recipe)  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(raw, recipe, input_scale="bad")
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(
            raw.drop(columns="selected"),
            recipe,
            require_outcome=True,
        )

    condition = str(prepared.contract.mappings["condition"])
    unknown = raw.copy()
    unknown.loc[unknown.index[0], condition] = "__unknown__"
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(unknown, recipe)

    scaled = next(iter(prepared.transformations["numeric_scaling"]))
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(raw.drop(columns=scaled), recipe)

    nonfinite = raw.copy()
    nonfinite.loc[nonfinite.index[0], scaled] = np.inf
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(nonfinite, recipe)

    bad_prepared = prepared.data.copy()
    bad_prepared.loc[bad_prepared.index[0], condition] = 99
    with pytest.raises(GP3BayesError):
        c.invert_transformation_recipe(bad_prepared, recipe)

    bad_outcome = prepared.data.copy()
    bad_outcome.loc[bad_outcome.index[0], "selected"] = 9
    with pytest.raises(GP3BayesError):
        c.invert_transformation_recipe(bad_outcome, recipe)


def test_duration_transformation_recipe_roundtrip_and_errors():
    _, _, prepared, _ = _duration_objects(1730)
    recipe = c.create_transformation_recipe(prepared)
    raw = c.invert_transformation_recipe(prepared.data, recipe)
    replay = c.apply_transformation_recipe(
        raw,
        recipe,
        require_outcome=True,
        input_unit=recipe.transformations["outcome"]["source_unit"],
    )
    assert len(replay) == len(prepared.data)
    assert c.validate_transformation_replay(prepared).replay_established

    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(raw, recipe, input_unit="wrong-unit")

    bad_y = raw.copy()
    bad_y.loc[bad_y.index[0], "duration"] = 0
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(bad_y, recipe)

    condition = str(prepared.contract.mappings["condition"])
    bad_c = raw.copy()
    bad_c.loc[bad_c.index[0], condition] = "__unknown__"
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(bad_c, recipe)

    scaled = next(iter(prepared.transformations["scaled_columns"]))
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(raw.drop(columns=scaled), recipe)

    bad_scaled = raw.copy()
    bad_scaled.loc[bad_scaled.index[0], scaled] = np.inf
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(bad_scaled, recipe)


def test_sensitivity_specification_builders_and_plan_guards():
    _, _, bprepared, bspec = _binary_objects(1740)
    _, _, _, dspec = _duration_objects(1741)

    rs = c.create_random_slope_sensitivity_plan(bspec)
    assert rs.family == "binary"
    assert set(rs.intercept_only) >= {"ready", "contract", "prepared", "specification"}
    assert set(rs.random_slope) >= {"ready", "contract", "prepared", "specification"}

    participant_col = str(bspec.contract.mappings["participant"])
    first = str(bspec.prepared.data[participant_col].iloc[0])
    gd = c.create_group_deletion_sensitivity_plan(
        bspec,
        group="participant",
        units=(first,),
    )
    assert gd.units == (first,)
    assert len(gd.table) == 1

    with pytest.raises(GP3BayesError):
        c.create_group_deletion_sensitivity_plan(bspec, group="bad")
    with pytest.raises(GP3BayesError):
        c.create_group_deletion_sensitivity_plan(
            bspec,
            group="participant",
            units=("unknown",),
        )
    with pytest.raises(GP3BayesError):
        c.create_group_deletion_sensitivity_plan(
            bspec,
            group="participant",
            max_units=1,
        )

    contrast = c.create_contrast_coding_sensitivity_specification(
        bspec,
        (-1.0, 1.0),
        0.5,
    )
    assert contrast.family == "binary"
    with pytest.raises(GP3BayesError):
        c.create_contrast_coding_sensitivity_specification(
            bspec,
            (0.0, 0.0),
            0.5,
        )

    scaled = c.create_predictor_scaling_sensitivity_specification(
        bspec,
        "participant_covariate",
        2.0,
        1.25,
    )
    assert scaled.family == "binary"
    with pytest.raises(GP3BayesError):
        c.create_predictor_scaling_sensitivity_specification(
            bspec,
            "missing",
            2.0,
            1.0,
        )

    unit = c.create_duration_unit_sensitivity_specification(
        dspec,
        0.001,
        "seconds",
    )
    assert unit.contract.outcome_unit == "seconds"
    with pytest.raises(GP3BayesError):
        c.create_duration_unit_sensitivity_specification(
            bspec,
            0.001,
            "seconds",
        )
    with pytest.raises(GP3BayesError):
        c.create_duration_unit_sensitivity_specification(
            dspec,
            1.0,
            "",
        )
