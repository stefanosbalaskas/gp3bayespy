from __future__ import annotations

from types import SimpleNamespace
import copy
import importlib as pyimportlib

import numpy as np
import pandas as pd
import pytest

import gp3bayespy.multilevel_mediation as mm


class Prepared:
    def __init__(self, data: pd.DataFrame, provenance=None):
        frame = data.copy(deep=True)
        xmean = frame.groupby("participant_id")["ai_correct"].transform("mean")
        mmean = frame.groupby("participant_id")["source_dwell"].transform("mean")
        frame["X_within"] = frame["ai_correct"] - xmean
        frame["X_between"] = xmean
        frame["M_within"] = frame["source_dwell"] - mmean
        frame["M_between"] = mmean
        frame["mediation_analysis_eligible"] = frame[
            ["ai_correct", "source_dwell", "correct_override"]
        ].notna().all(axis=1)
        self.data = frame
        self.columns = {
            "participant": "participant_id",
            "trial": "trial_id",
            "x": "ai_correct",
            "mediator": "source_dwell",
            "outcome": "correct_override",
        }
        self.provenance = {"fixture": True} if provenance is None else provenance


def prepared(n=8, trials=6, seed=81):
    return Prepared(mm.simulate_multilevel_gaze_mediation(
        n_participants=n, trials_per_participant=trials, seed=seed
    ))


def augment_serial(p: Prepared, *, constant=False):
    p = copy.deepcopy(p)
    d = p.data
    d["trust"] = 2.0 if constant else 2 + 0.4 * d["ai_correct"] + 0.2 * d["source_dwell"] + 0.01 * np.arange(len(d))
    mean = d.groupby("participant_id")["trust"].transform("mean")
    d["M2_within"] = d["trust"] - mean
    d["M2_between"] = mean
    p.columns.update({"mediator2": "trust", "mediator2_within": "M2_within", "mediator2_between": "M2_between"})
    return p


def augment_moderator(p: Prepared, *, component="within", constant=False):
    p = copy.deepcopy(p)
    d = p.data
    if constant:
        d["moderator"] = 1.0
    else:
        d["moderator"] = np.resize(np.array([-1.0, 0.0, 1.0]), len(d))
    mean = d.groupby("participant_id")["moderator"].transform("mean")
    d["Z_within"] = d["moderator"] - mean
    d["Z_between"] = mean
    # ensure a varying between component when explicitly requested
    if component == "between" and not constant:
        participant_code = pd.factorize(d["participant_id"])[0].astype(float)
        d["Z_between"] = participant_code
    p.columns.update({"moderator": "moderator", "moderator_within": "Z_within", "moderator_between": "Z_between"})
    return p


class FakeModel:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False


class FakePM:
    def __init__(self, n_rows=48):
        self.n_rows = n_rows
        self.model = FakeModel()
        self.record = {}
        self.math = SimpleNamespace(exp=np.exp, sigmoid=lambda x: 1 / (1 + np.exp(-np.asarray(x))))
        self.distributions = SimpleNamespace(transforms=SimpleNamespace(ordered=object()))

    def Model(self):
        return self.model

    def Normal(self, name, mu=0.0, sigma=1.0, shape=None, **kwargs):
        value = np.full(shape, 0.1) if shape is not None else 0.1
        self.record[name] = value
        return value

    def HalfNormal(self, name, sigma=1.0, **kwargs):
        self.record[name] = 0.2
        return 0.2

    def Exponential(self, name, lam=1.0, **kwargs):
        self.record[name] = 1.5
        return 1.5

    def Deterministic(self, name, value):
        arr = np.asarray(value)
        self.record[name] = arr
        return arr

    def _obs(self, name, **kwargs):
        self.record[name] = kwargs
        return kwargs

    def LogNormal(self, name, **kwargs): return self._obs(name, **kwargs)
    def Gamma(self, name, **kwargs): return self._obs(name, **kwargs)
    def Beta(self, name, **kwargs): return self._obs(name, **kwargs)
    def Bernoulli(self, name, **kwargs): return self._obs(name, **kwargs)
    def Poisson(self, name, **kwargs): return self._obs(name, **kwargs)
    def NegativeBinomial(self, name, **kwargs): return self._obs(name, **kwargs)
    def OrderedLogistic(self, name, **kwargs): return self._obs(name, **kwargs)

    def sample(self, **kwargs):
        names = [
            "a_within", "a_between", "b_within", "b_between", "cprime_within", "cprime_between",
            "indirect_within", "indirect_between", "total_within", "total_between",
            "participant_a_slope", "participant_b_slope", "participant_indirect_within",
            "a1_within", "a1_between", "a2_within", "a2_between", "d_within", "d_between",
            "b1_within", "b1_between", "b2_within", "b2_between", "serial_indirect_within",
            "serial_indirect_between", "m1_indirect_within", "m2_indirect_within",
            "total_indirect_within", "moderation",
        ]
        posterior = {name: np.ones((2, 3)) * (i + 1) / 10 for i, name in enumerate(names)}
        return SimpleNamespace(
            posterior=posterior,
            sample_stats={"diverging": np.zeros((2, 3), dtype=bool), "tree_depth": np.ones((2, 3), dtype=int)},
        )

    def sample_posterior_predictive(self, *args, **kwargs):
        n = self.n_rows
        return SimpleNamespace(posterior_predictive={
            "M_obs": np.arange(5 * n, dtype=float).reshape(1, 5, n) / max(n, 1),
            "Y_obs": np.resize(np.array([0.0, 1.0]), 5 * n).reshape(1, 5, n),
        })

    def sample_prior_predictive(self, draws, random_seed):
        n = self.n_rows
        return SimpleNamespace(prior_predictive={
            "M_obs": np.ones((1, draws, n)),
            "Y_obs": np.zeros((1, draws, n)),
        })


class FakeDataVars:
    def __init__(self, values):
        self.data_vars = {f"v{i}": np.asarray(v) for i, v in enumerate(values)}


class FakeArviz:
    def __init__(self, *, loo_fail=False):
        self.loo_fail = loo_fail
    def rhat(self, idata):
        return FakeDataVars([np.array([1.001, 1.005])])
    def ess(self, idata, method="bulk"):
        return FakeDataVars([np.array([700.0, 600.0])])
    def loo(self, idata, var_name=None):
        if self.loo_fail:
            raise RuntimeError("loo failed")
        offset = float(np.asarray(idata.log_likelihood["joint"]).mean())
        return SimpleNamespace(elpd_loo=10 + offset, se=0.5, p_loo=1.2, warning=False)


class FakeBackendFit:
    def __init__(self, offset=0.0):
        self.log_likelihood = {
            "M_obs": np.ones((2, 3, 4)) * offset,
            "Y_obs": np.ones((2, 3, 4)) * (offset + 0.1),
        }
    def copy(self):
        out = FakeBackendFit()
        out.log_likelihood = {k: np.array(v, copy=True) for k, v in self.log_likelihood.items()}
        return out


def passing_fit(p=None, posterior=None, backend_fit=None, backend_model=None, kind="simple"):
    p = p or prepared()
    spec = mm.specify_multilevel_gaze_mediation(p, random_slopes=())
    if kind != "simple":
        spec = mm.MultilevelMediationSpecification(
            **{**{name: getattr(spec, name) for name in spec.__dataclass_fields__}, "model_kind": kind}
        )
    x = np.linspace(-0.1, 0.1, 20)
    default = {
        "a_within": x + 0.6, "b_within": x + 0.7,
        "a_between": x + 0.2, "b_between": x + 0.3,
        "cprime_within": x + 0.1, "cprime_between": x + 0.05,
        "indirect_within": (x + 0.6) * (x + 0.7),
        "indirect_between": (x + 0.2) * (x + 0.3),
        "total_within": x + 0.1 + (x + 0.6) * (x + 0.7),
        "total_between": x + 0.05 + (x + 0.2) * (x + 0.3),
    }
    if posterior is not None:
        default = posterior
