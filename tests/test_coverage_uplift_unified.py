from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

matplotlib.use("Agg")

import gp3bayespy.binary as binary
import gp3bayespy.duration as duration
import gp3bayespy.pupil as pupil
import gp3bayespy.specification_closure as closure
import gp3bayespy.unified_workflow_api as unified
from gp3bayespy.exceptions import GP3BayesError


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class FakeFit:
    __module__ = "gp3bayespy.coverage_fixture"

    def __init__(self, family: str):
        self.family = family
        self.specification = SimpleNamespace(family=family)
        self.backend_fit = object()
        self.sampling_backend = "analytic"
        self.algorithm = "coverage"
        self.sampling = {"chains": 2}
        self.fit_performed = True


class FakeContract:
    __module__ = "gp3bayespy.coverage_fixture"

    def __init__(self):
        self.contract_version = "0.2"
        self.family = "binary"
        self.model_family = "Bernoulli"
        self.mappings = {}
        self.predictors = ()
        self.likelihood = "Bernoulli"
        self.link = "logit"


class FakePrepared:
    __module__ = "gp3bayespy.coverage_fixture"

    def __init__(self, contract):
        self.family = "binary"
        self.data = pd.DataFrame({"y": [0, 1]})
        self.contract = contract
        self.transformations = {}


class FakeSpecification:
    __module__ = "gp3bayespy.coverage_fixture"

    def __init__(self, contract, prepared):
        self.family = "binary"
        self.contract = contract
        self.prepared = prepared
        self.formula = "y ~ 1"
        self.priors = object()


def test_unified_structural_validation_covers_contract_prepared_spec_fit_summary_and_errors():
    contract = FakeContract()
    prepared = FakePrepared(contract)
    specification = FakeSpecification(contract, prepared)

    for obj in (contract, prepared, specification, FakeFit("binary")):
        result = unified.validate_gp3bayes_object(obj)
        assert result.status in {"pass", "review"}
        assert isinstance(result.to_frame(), pd.DataFrame)

    broken = FakeContract()
    del broken.link
    assert unified.validate_gp3bayes_object(broken).status == "fail"
    with pytest.raises(GP3BayesError, match="validation failed"):
        unified.validate_gp3bayes_object(broken, strict=True)

    class BinaryDiagnostic:
        __module__ = "gp3bayespy.coverage_fixture"
        family = "binary"
        status = "review"
        fit_performed = False

    class BinarySummary:
        __module__ = "gp3bayespy.coverage_fixture"
        family = "binary"
        table = pd.DataFrame({"x": [1]})
        fit_performed = False

    assert unified.validate_gp3bayes_object(BinaryDiagnostic()).status == "pass"
    assert unified.validate_gp3bayes_object(BinarySummary()).status == "pass"

    with pytest.raises(GP3BayesError, match="TRUE or FALSE"):
        unified.validate_gp3bayes_object(contract, recursive="yes")  # type: ignore[arg-type]


def test_unified_dispatchers_cover_all_families(monkeypatch):
    sentinel = object()

    monkeypatch.setattr(binary, "diagnose_binary_fit", lambda fit, **kwargs: ("binary-d", kwargs))
    monkeypatch.setattr(
        duration, "diagnose_duration_fit", lambda fit, **kwargs: ("duration-d", kwargs)
    )
    monkeypatch.setattr(pupil, "diagnose_pupil_fit", lambda fit: "pupil-d")

    monkeypatch.setattr(
        binary, "summarise_binary_posterior", lambda fit, **kwargs: ("binary-s", kwargs)
    )
    monkeypatch.setattr(
        duration, "summarise_duration_posterior", lambda fit, **kwargs: ("duration-s", kwargs)
    )
    monkeypatch.setattr(
        pupil, "summarise_pupil_posterior", lambda fit, **kwargs: ("pupil-s", kwargs)
    )

    monkeypatch.setattr(
        binary, "check_binary_posterior_predictive", lambda fit, **kwargs: ("binary-p", kwargs)
    )
    monkeypatch.setattr(
        duration,
        "check_duration_posterior_predictive",
        lambda fit, **kwargs: ("duration-p", kwargs),
    )
    monkeypatch.setattr(
        pupil, "check_pupil_posterior_predictive", lambda fit, **kwargs: ("pupil-p", kwargs)
    )

    monkeypatch.setattr(closure, "estimate_standardized_probability_contrast", lambda fit: sentinel)
    monkeypatch.setattr(closure, "estimate_standardized_duration_estimands", lambda fit: sentinel)

    for family in ("binary", "duration", "pupil"):
        fit = FakeFit(family)
        assert unified.diagnose_model_fit(fit) is not None
        assert unified.summarise_model_posterior(fit, probability=0.8) is not None
        assert unified.check_model_ppc(fit, draws=20) is not None

    assert unified.estimate_model_estimands(FakeFit("binary")) is sentinel
    assert unified.estimate_model_estimands(FakeFit("duration")) is sentinel
    with pytest.raises(GP3BayesError, match="binary and duration"):
        unified.estimate_model_estimands(FakeFit("pupil"))


def test_workflow_status_maps_component_evidence_without_decisions():
    fit = FakeFit("binary")
    wrapper = SimpleNamespace(
        fit=fit,
        specification=fit.specification,
        components={
            "diagnostics": object(),
            "posterior": object(),
            "ppc": object(),
            "estimands": object(),
            "sensitivity": object(),
            "loo": object(),
            "manifest": object(),
        },
    )
    status = unified.model_workflow_status(wrapper)
    assert status.attrs["structural_stage_map_only"] is True
    assert status.loc[status["stage"] == "fit", "completed"].item()
    assert status.loc[status["stage"] == "predictive_validation", "completed"].item()
