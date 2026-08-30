from __future__ import annotations

import importlib
from types import SimpleNamespace

import pandas as pd
import pytest

import gp3bayespy as gp
import gp3bayespy.design_support_diagnostics as d
import gp3bayespy.unified_workflow_api as u
from gp3bayespy.exceptions import GP3BayesError

be = importlib.import_module("gp3bayespy.backends")
pupil = importlib.import_module("gp3bayespy.pupil")
closure = importlib.import_module("gp3bayespy.specification_closure")


def _binary(seed: int = 3301):
    sim = gp.simulate_hierarchical_binary_data(
        n_participants=4,
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
        random_slope=False,
    )
    return sim.data.copy(), contract


def test_design_audit_frame_adapters():
    data, contract = _binary()

    missingness = d.audit_missingness_structure(data, contract)
    fixed = d.audit_fixed_effect_design(data, contract)
    random = d.audit_random_effects_support(data, contract)
    combined = d.audit_design_support(
        data,
        contract,
        separation=False,
        strict_readiness=False,
    )

    assert isinstance(missingness.to_frame(), pd.DataFrame)
    assert isinstance(fixed.to_frame(), pd.DataFrame)
    assert isinstance(random.to_frame(), pd.DataFrame)
    assert isinstance(combined.to_frame(), pd.DataFrame)

    assert missingness.to_frame().equals(missingness.column_table)
    assert fixed.to_frame().equals(fixed.column_table)
    assert random.to_frame().equals(random.component_table)
    assert combined.to_frame().equals(combined.component_table)


def test_backend_missing_distribution_version(monkeypatch):
    def missing_version(package: str):
        raise importlib.metadata.PackageNotFoundError(package)

    monkeypatch.setattr(
        be.importlib.metadata,
        "version",
        missing_version,
    )
    assert be._version("definitely-not-installed-gp3bayespy-test") is None


def test_unified_guarded_import_failures(monkeypatch):
    pupil_fit = SimpleNamespace(
        fit_performed=True,
        family="pupil",
    )

    monkeypatch.delattr(
        pupil,
        "diagnose_pupil_fit",
    )
    with pytest.raises(
        GP3BayesError,
        match="Pupil diagnostics are not available",
    ):
        u.diagnose_model_fit(pupil_fit)

    monkeypatch.delattr(
        pupil,
        "summarise_pupil_posterior",
    )
    with pytest.raises(
        GP3BayesError,
        match="Pupil posterior summaries are not available",
    ):
        u.summarise_model_posterior(pupil_fit)

    monkeypatch.delattr(
        pupil,
        "check_pupil_posterior_predictive",
    )
    with pytest.raises(
        GP3BayesError,
        match="Pupil posterior predictive checks are not available",
    ):
        u.check_model_ppc(pupil_fit)

    binary_fit = SimpleNamespace(
        fit_performed=True,
        family="binary",
    )
    monkeypatch.delattr(
        closure,
        "estimate_standardized_probability_contrast",
    )
    with pytest.raises(
        GP3BayesError,
        match="Binary standardized estimands are not available",
    ):
        u.estimate_model_estimands(binary_fit)

    duration_fit = SimpleNamespace(
        fit_performed=True,
        family="duration",
    )
    monkeypatch.delattr(
        closure,
        "estimate_standardized_duration_estimands",
    )
    with pytest.raises(
        GP3BayesError,
        match="Duration standardized estimands are not available",
    ):
        u.estimate_model_estimands(duration_fit)


def test_unified_unsupported_family_dispatchers():
    unsupported = SimpleNamespace(
        fit_performed=True,
        family="mystery",
    )

    with pytest.raises(
        GP3BayesError,
        match="Unsupported gp3bayes fit family",
    ):
        u.diagnose_model_fit(unsupported)

    with pytest.raises(
        GP3BayesError,
        match="Unsupported gp3bayes fit family",
    ):
        u.summarise_model_posterior(unsupported)

    with pytest.raises(
        GP3BayesError,
        match="Unsupported gp3bayes fit family",
    ):
        u.check_model_ppc(unsupported)

    with pytest.raises(
        GP3BayesError,
        match="Unsupported gp3bayes fit family",
    ):
        u.estimate_model_estimands(unsupported)
