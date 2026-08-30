from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

import gp3bayespy as gp
from gp3bayespy.exceptions import GP3BayesError

c = importlib.import_module("gp3bayespy.specification_closure")
s = importlib.import_module("gp3bayespy.sensitivity")


def _binary(seed=5101):
    sim = gp.simulate_hierarchical_binary_data(
        n_participants=4, trials_per_participant=6, n_items=3, seed=seed
    )
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=("trial_covariate",),
        random_slope=False,
    )
    prepared = gp.prepare_hierarchical_binary_data(
        sim.data, contract, scale_predictors=("trial_covariate",)
    )
    return gp.specify_binary_model(prepared)


def _duration(seed=5102):
    sim = gp.simulate_hierarchical_duration_data(
        n_participants=4, trials_per_participant=6, n_items=3, seed=seed
    )
    contract = gp.create_model_contract(
        family="duration",
        outcome_col="duration",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=("trial_covariate",),
        random_slope=False,
        outcome_unit="milliseconds",
    )
    prepared = gp.prepare_hierarchical_duration_data(
        sim.data,
        contract,
        scale_predictors=("trial_covariate",),
    )
    return gp.specify_duration_model(prepared, baseline=500.0)


def test_closure_reprepare_rebuild_and_duration_contrast_paths():
    bspec = _binary()
    raw_b = c.invert_transformation_recipe(bspec.prepared.data, bspec.prepared)
    prepared_b = c._reprepare(bspec, bspec.contract, raw_b)
    assert prepared_b.audit.ready
    rebuilt_b = c._rebuild(prepared_b, bspec, baseline=0.4, coefficient_scale=0.6)
    assert rebuilt_b.family == "binary"

    dspec = _duration()
    raw_d = c.invert_transformation_recipe(dspec.prepared.data, dspec.prepared)
    prepared_d = c._reprepare(dspec, dspec.contract, raw_d)
    assert prepared_d.audit.ready
    rebuilt_d = c._rebuild(prepared_d, dspec, baseline=400.0, coefficient_scale=0.4)
    assert rebuilt_d.family == "duration"

    recoded = c.create_contrast_coding_sensitivity_specification(dspec, (-1.0, 1.0), 500.0)
    assert recoded.family == "duration"

    deletion = c.create_group_deletion_sensitivity_plan(bspec, units=None, max_units=10)
    assert len(deletion.units) == bspec.prepared.data["participant_id"].nunique()

    with pytest.raises(GP3BayesError):
        c.create_group_deletion_sensitivity_plan(bspec, units=None, max_units=1)


def test_closure_identifier_duration_boundaries_remaining_paths():
    bspec = _binary(5110)
    data = bspec.prepared.data.copy()
    data["subject_id_numeric"] = np.arange(1, len(data) + 1)
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=("subject_id_numeric",),
        random_slope=False,
    )
    audit = c.identify_identifier_like_predictors(
        data,
        contract,
        unique_fraction=0.5,
        integer_fraction=0.5,
        monotone_correlation=0.5,
    )
    assert "subject_id_numeric" in audit.flagged

    text_data = data.copy()
    text_data["label"] = ["x"] * len(text_data)
    text_contract = gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        predictors=("label",),
    )
    text_audit = c.identify_identifier_like_predictors(text_data, text_contract)
    assert text_audit.table.iloc[0]["reason"] == "non_numeric"

    dspec = _duration(5111)
    ddata = dspec.prepared.data.copy()
    ddata["deadline_flag"] = False
    noheur = c.audit_duration_boundaries(
        ddata,
        dspec.contract,
        detect_candidate_columns=False,
    )
    assert "not_applicable" in set(noheur.checks["status"])

    cens = ddata.copy()
    cens["cens"] = ["no"] * len(cens)
    explicit = c.audit_duration_boundaries(cens, dspec.contract, censor_col="cens")
    assert explicit.censored_rows == ()

    constant = ddata.copy()
    constant["duration"] = 10.0
    review = c.review_duration_extremes(constant, dspec.contract)
    assert review.n_flagged == 0


def test_sensitivity_suite_every_orchestration_branch(monkeypatch):
    b = importlib.import_module("gp3bayespy.binary")
    aow = importlib.import_module("gp3bayespy.advanced_optional_workflows")
    closure = importlib.import_module("gp3bayespy.specification_closure")
    u = importlib.import_module("gp3bayespy.unified_workflow_api")

    token = SimpleNamespace(status="pass", interpretation="ok")
    monkeypatch.setattr(b, "assess_binary_prior_sensitivity", lambda **kwargs: token)
    monkeypatch.setattr(aow, "assess_powerscaled_sensitivity", lambda **kwargs: token)
    monkeypatch.setattr(aow, "compute_psis_loo", lambda **kwargs: token)
    monkeypatch.setattr(closure, "run_random_slope_sensitivity", lambda **kwargs: token)
    monkeypatch.setattr(closure, "run_group_deletion_sensitivity", lambda **kwargs: token)
    monkeypatch.setattr(closure, "compare_estimand_sensitivity", lambda **kwargs: token)
    monkeypatch.setattr(closure, "audit_duration_unit_invariance", lambda **kwargs: token)
    monkeypatch.setattr(u, "estimate_model_estimands", lambda fit: SimpleNamespace(status="pass"))

    plan = s.create_sensitivity_suite_plan(
        prior_scale=True,
        powerscale=True,
        psis_loo=True,
        random_slope_plan=object(),
        group_deletion_plan=object(),
        alternative_estimands={"alt": object()},
        duration_unit={"estimand": object(), "multiplier": 0.001},
    )
    fit = SimpleNamespace(family="binary")
    suite = s.run_sensitivity_suite(fit, plan)
    assert suite.status == "completed"
    assert len(suite.results) == 7

    bad_plan = s.create_sensitivity_suite_plan(duration_unit={"estimand": object()})
    with pytest.raises(GP3BayesError):
        s.run_sensitivity_suite(fit, bad_plan)

    monkeypatch.setattr(
        u,
        "estimate_model_estimands",
        lambda fit: (_ for _ in ()).throw(RuntimeError("estimate-fail")),
    )
    alt_plan = s.create_sensitivity_suite_plan(alternative_estimands={"a": object()})
    suite2 = s.run_sensitivity_suite(fit, alt_plan, stop_on_error=False)
    assert suite2.reference_estimand.status == "error"

    with pytest.raises(RuntimeError):
        s.run_sensitivity_suite(fit, alt_plan, stop_on_error=True)
