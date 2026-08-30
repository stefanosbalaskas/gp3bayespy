from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import gp3bayespy.pupil as pupil
import gp3bayespy.unified_workflow_api as u
from gp3bayespy.exceptions import GP3BayesError


def test_unified_structural_validation_branch_matrix():
    with pytest.raises(GP3BayesError):
        u.validate_gp3bayes_object(
            SimpleNamespace(fit_performed=True),
            recursive=1,  # type: ignore[arg-type]
        )
    with pytest.raises(GP3BayesError):
        u.validate_gp3bayes_object(
            SimpleNamespace(fit_performed=True),
            strict=1,  # type: ignore[arg-type]
        )

    contract_like = SimpleNamespace(
        contract_version="0.1",
        family="binary",
    )
    result = u.validate_gp3bayes_object(contract_like)
    assert result.status == "fail"

    prepared_like = SimpleNamespace(
        fit_performed=True,
        data=[1, 2],
        contract=contract_like,
        transformations={},
    )
    result = u.validate_gp3bayes_object(prepared_like)
    assert result.status == "fail"

    spec_like = SimpleNamespace(
        fit_performed=True,
        priors=object(),
        formula="y ~ x",
    )
    result = u.validate_gp3bayes_object(spec_like)
    assert result.status == "fail"

    fit_like = SimpleNamespace(
        fit_performed=False,
        backend_fit=None,
        sampling_backend="pymc",
    )
    result = u.validate_gp3bayes_object(fit_like)
    assert result.status == "fail"
    assert "review" in set(result.checks["status"])

    DiagnosticLike = type("SyntheticDiagnostic", (), {})
    diagnostic = DiagnosticLike()
    diagnostic.fit_performed = True
    diagnostic.status = None
    result = u.validate_gp3bayes_object(diagnostic)
    assert result.status == "fail"

    SummaryLike = type("SyntheticSummary", (), {})
    summary = SummaryLike()
    summary.fit_performed = True
    summary.table = pd.DataFrame()
    result = u.validate_gp3bayes_object(summary)
    assert result.status == "fail"

    bad_child = SimpleNamespace(
        fit_performed=False,
        backend_fit=None,
        sampling_backend="pymc",
    )
    parent = SimpleNamespace(
        fit_performed=True,
        specification=bad_child,
    )
    result = u.validate_gp3bayes_object(parent, recursive=True)
    assert result.status == "fail"
    assert "nested_specification" in set(result.checks["check"])

    with pytest.raises(GP3BayesError):
        u.validate_gp3bayes_object(parent, strict=True)


def test_unified_pupil_dispatch_and_workflow_tail(monkeypatch):
    fake = SimpleNamespace(
        fit_performed=True,
        family="pupil",
    )

    monkeypatch.setattr(
        pupil,
        "diagnose_pupil_fit",
        lambda fit: "diagnosed",
    )
    monkeypatch.setattr(
        pupil,
        "summarise_pupil_posterior",
        lambda fit, probability=0.95: ("summary", probability),
    )
    monkeypatch.setattr(
        pupil,
        "check_pupil_posterior_predictive",
        lambda fit, ndraws=500: ("ppc", ndraws),
    )

    assert u.diagnose_model_fit(fake) == "diagnosed"
    assert u.summarise_model_posterior(
        fake,
        probability=0.9,
    ) == ("summary", 0.9)
    assert u.check_model_ppc(
        fake,
        draws=77,
    ) == ("ppc", 77)

    with pytest.raises(GP3BayesError):
        u.estimate_model_estimands(fake)

    prepared = SimpleNamespace(
        fit_performed=False,
        data=pd.DataFrame({"x": [1]}),
        transformations={},
        contract=SimpleNamespace(contract_version="0.1"),
    )
    stages = u.model_workflow_status(prepared)
    completed = dict(zip(stages["stage"], stages["completed"], strict=True))
    assert completed["prepared_data"] is True
    assert completed["contract"] is True
