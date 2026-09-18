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
    return mm.MultilevelMediationFit(
        specification=spec, posterior=default,
        sampler_diagnostics={"max_rhat": 1.001, "min_ess_bulk": 800, "divergences": 0, "treedepth_hits": 0},
        backend_fit=backend_fit, backend_model=backend_model, backend="fixture",
    )


def test_low_level_validation_contracts(monkeypatch):
    for bad in [None, "", "   "]:
        with pytest.raises(mm.GP3BayesError, match="non-empty family"):
            mm._normalize_family(bad, {"gaussian"}, "family")
    assert mm._normalize_family("negative-binomial", mm._MEDIATOR_FAMILIES, "family") == "negative_binomial"
    assert mm._normalize_family("normal", mm._MEDIATOR_FAMILIES, "family") == "gaussian"
    with pytest.raises(mm.GP3BayesError, match="Unsupported"):
        mm._normalize_family("weird", {"gaussian"}, "family")
    for bad in [True, 0, -1, np.inf, "x"]:
        with pytest.raises(mm.GP3BayesError, match="finite positive"):
            mm._finite_positive(bad, "x")
    priors = mm.create_mediation_prior_specification()
    assert priors.as_dict()["intercept_sd"] == 2.5

    with pytest.raises(mm.GP3BayesError, match="preparation object"):
        mm._prepared_contract(object())
    x = SimpleNamespace(data=pd.DataFrame(), columns=[], provenance={})
    with pytest.raises(mm.GP3BayesError, match="canonical mediation column contract"):
        mm._prepared_contract(x)
    x.columns = {"participant": "p", "trial": "t", "mediator": "m", "outcome": "y"}
    x.provenance = "not-map"
    x.data = pd.DataFrame({"X_within": [], "X_between": [], "M_within": [], "M_between": [], "mediation_analysis_eligible": []})
    _, _, provenance = mm._prepared_contract(x)
    assert provenance == {}
    x.columns = {"participant": "p"}
    with pytest.raises(mm.GP3BayesError, match="missing canonical mediation semantics"):
        mm._prepared_contract(x)
    x.columns = {"participant": "p", "trial": "t", "mediator": "m", "outcome": "y"}
    x.data = pd.DataFrame()
    with pytest.raises(mm.GP3BayesError, match="missing canonical columns"):
        mm._prepared_contract(x)

    monkeypatch.setattr(mm, "version", lambda name: (_ for _ in ()).throw(mm.PackageNotFoundError()))
    assert mm._package_version("missing") == "not_installed"
    fake = object()
    monkeypatch.setattr(mm, "import_module", lambda name: fake)
    assert mm._load_pymc() is fake


def test_family_data_all_failures_and_valid_ordinal():
    base = pd.DataFrame({"m": [1.0, 2.0, 3.0], "y": [0, 1, 2]})
    with pytest.raises(mm.GP3BayesError, match="bernoulli.*mediator"):
        mm._validate_family_data(base, "m", "y", "bernoulli", "ordinal")
    with pytest.raises(mm.GP3BayesError, match="bernoulli.*outcome"):
        mm._validate_family_data(pd.DataFrame({"m": [1., 2.], "y": [0., 2.]}), "m", "y", "gaussian", "bernoulli")
    with pytest.raises(mm.GP3BayesError, match="Outcome family.*integer counts"):
        mm._validate_family_data(pd.DataFrame({"m": [1., 2.], "y": [0.2, 1.]}), "m", "y", "gaussian", "poisson")
    mm._validate_family_data(base, "m", "y", "gaussian", "ordinal")
    with pytest.raises(mm.GP3BayesError, match="contiguous integer"):
        mm._validate_family_data(pd.DataFrame({"m": [1., 2.], "y": [1., 2.]}), "m", "y", "gaussian", "ordinal")


def test_estimability_and_sampling_helpers_cover_edges():
    assert not mm._has_variation([1.0])
    assert not mm._has_variation([1.0, 1.0])
    assert mm._has_variation([1.0, 2.0])
    df = pd.DataFrame({
        "X_within": [0, 1], "X_between": [0, 1], "M_within": [0, 1], "M_between": [0, 1]
    })
    assert set(mm._estimable_simple_paths(df)) == {"a_within", "cprime_within", "a_between", "cprime_between", "b_within", "b_between"}
    zeros = df * 0
    assert mm._estimable_simple_paths(zeros) == ()

    controls = mm._validate_sampling_controls(chains=3, draws=50, tune=0, cores=None, seed=0, target_accept=.9, max_treedepth=5)
    assert controls["cores"] == 2
    controls2 = mm._validate_sampling_controls(chains=2, draws=50, tune=1, cores=1, seed=2, target_accept=.8, max_treedepth=6)
    assert controls2["cores"] == 1
    cases = [
        ({"chains": 1}, "chains"), ({"tune": -1}, "tune"), ({"max_treedepth": 4}, "max_treedepth"),
        ({"cores": 0}, "cores"), ({"seed": -1}, "seed"), ({"target_accept": True}, "target_accept"),
    ]
    defaults = dict(chains=2, draws=50, tune=0, cores=1, seed=0, target_accept=.9, max_treedepth=5)
    for update, msg in cases:
        args = {**defaults, **update}
        with pytest.raises(mm.GP3BayesError, match=msg):
            mm._validate_sampling_controls(**args)
    for d, s, msg in [(0, 1, "draws"), (True, 1, "draws"), (1, -1, "seed"), (1, True, "seed")]:
        with pytest.raises(mm.GP3BayesError, match=msg):
            mm._validate_draw_seed(d, s)
    assert mm._validate_draw_seed(2, 3) == (2, 3)


def test_specification_validation_remaining_branches(monkeypatch):
    p = prepared()
    with pytest.raises(mm.GP3BayesError, match="missingness_policy"):
        mm.specify_multilevel_gaze_mediation(p, missingness_policy="magic", random_slopes=())
    with pytest.raises(mm.GP3BayesError, match="duplicates"):
        mm.specify_multilevel_gaze_mediation(p, random_slopes=("mediator_x", "mediator_x"))
    with pytest.raises(mm.GP3BayesError, match="MediationPriorSpecification"):
        mm.specify_multilevel_gaze_mediation(p, priors={"bad": 1}, random_slopes=())

    q = copy.deepcopy(p)
    q.data = q.data.drop(columns=["correct_override"])
    with pytest.raises(mm.GP3BayesError, match="missing model columns"):
        mm.specify_multilevel_gaze_mediation(q, random_slopes=())

    q = copy.deepcopy(p)
    q.data["mediation_analysis_eligible"] = False
    with pytest.raises(mm.GP3BayesError, match="No rows remain"):
        mm.specify_multilevel_gaze_mediation(q, missingness_policy="quality_eligible", random_slopes=())

    q = copy.deepcopy(p)
    q.data = q.data[q.data["participant_id"].eq(q.data["participant_id"].iloc[0])].copy()
    with pytest.raises(mm.GP3BayesError, match="At least two participants"):
        mm.specify_multilevel_gaze_mediation(q, random_slopes=())

    q = copy.deepcopy(p)
    q.data["M_within"] = 0.0
    with pytest.raises(mm.GP3BayesError, match="not estimable"):
        mm.specify_multilevel_gaze_mediation(q, random_slopes=())

    # Artificially isolate the guard for requested but unsupported slope paths.
    monkeypatch.setattr(mm, "_estimable_simple_paths", lambda data: ("a_within", "b_within"))
    with pytest.raises(mm.GP3BayesError, match="Requested random slope"):
        mm.specify_multilevel_gaze_mediation(p, random_slopes=("outcome_x",))


def test_random_slope_repeated_support_guard(monkeypatch):
    p = prepared(n=2, trials=3)
    # Retain one repeated participant and one singleton, but preserve data variation.
    d = pd.concat([p.data[p.data.participant_id.eq("p001")], p.data[p.data.participant_id.eq("p002")].head(1)], ignore_index=True)
    p.data = d
    monkeypatch.setattr(mm, "_estimable_simple_paths", lambda data: ("a_within", "b_within", "cprime_within"))
    with pytest.raises(mm.GP3BayesError, match="Random slopes require repeated trials"):
        mm.specify_multilevel_gaze_mediation(p, random_slopes=("mediator_x",))


def test_observation_families_and_model_builders(monkeypatch):
    fake = FakePM()
    priors = mm.create_mediation_prior_specification()
    eta = np.array([0.1, 0.2, 0.3])
    families = {
        "gaussian": np.array([1., 2., 3.]), "lognormal": np.array([1., 2., 3.]),
        "gamma": np.array([1., 2., 3.]), "beta": np.array([.2, .4, .6]),
        "bernoulli": np.array([0., 1., 0.]), "poisson": np.array([0., 1., 2.]),
        "negative_binomial": np.array([0., 1., 2.]), "ordinal": np.array([0., 1., 2.]),
    }
    for fam, obs in families.items():
        mm._observe_family(fake, f"obs_{fam}", fam, eta, obs, priors)
    with pytest.raises(mm.GP3BayesError, match="Unsupported family"):
        mm._observe_family(fake, "bad", "bad", eta, eta, priors)
    assert len(mm._normal_paths(fake, 1.0)) == 6
    idx = np.array([0, 1, 0])
    assert np.asarray(mm._random_effect(fake, "r", 2, idx, 1.0)).shape == (3,)
    assert np.asarray(mm._random_slope(fake, "s", eta, 2, idx, 1.0)).shape == (3,)

    p = prepared()
    spec = mm.specify_multilevel_gaze_mediation(
        p, random_slopes=("mediator_x", "outcome_x", "outcome_m")
    )
    fake.n_rows = spec.analysis_rows
    monkeypatch.setattr(mm, "_load_pymc", lambda: fake)
    assert mm._build_pymc_model(spec) is fake.model
    assert "participant_indirect_within" in fake.record

    # Balanced-X setup omits between-X deterministic branches.
    q = prepared(n=4, trials=4)
    for _, gidx in q.data.groupby("participant_id").groups.items():
        q.data.loc[gidx, "ai_correct"] = [0, 1, 0, 1]
    xmean = q.data.groupby("participant_id")["ai_correct"].transform("mean")
    q.data["X_within"] = q.data["ai_correct"] - xmean
    q.data["X_between"] = xmean
    spec2 = mm.specify_multilevel_gaze_mediation(q, random_slopes=())
    mm._build_pymc_model(spec2)


def test_flatten_posterior_and_sampler_diagnostics(monkeypatch):
    assert mm._flatten_posterior(SimpleNamespace(), ["a"]) == {}
    idata = SimpleNamespace(posterior={"a": np.arange(12).reshape(2, 3, 2), "scalar": np.array([1.0])})
    out = mm._flatten_posterior(idata, ["a", "missing", "scalar"])
    assert out["a"].shape == (6, 2)
    assert out["scalar"].shape == (1,)

    fake_az = FakeArviz()
    monkeypatch.setattr(mm, "import_module", lambda name: fake_az if name == "arviz" else pyimportlib.import_module(name))
    idata2 = SimpleNamespace(sample_stats={"diverging": np.array([[False, True]]), "tree_depth": np.array([[4, 12]])})
    d = mm._sampler_diagnostics(idata2, max_treedepth=12)
    assert d["max_rhat"] == pytest.approx(1.005)
    assert d["min_ess_bulk"] == 600
    assert d["divergences"] == 1
    assert d["treedepth_hits"] == 1

    # ArviZ failure + fallback treedepth key + missing diverging field.
    def fail_arviz(name):
        if name == "arviz": raise RuntimeError("no arviz")
        return pyimportlib.import_module(name)
    monkeypatch.setattr(mm, "import_module", fail_arviz)
    d2 = mm._sampler_diagnostics(SimpleNamespace(sample_stats={"treedepth": np.array([[5, 6]])}), max_treedepth=6)
    assert np.isnan(d2["max_rhat"])
    assert d2["treedepth_hits"] == 1
    d3 = mm._sampler_diagnostics(SimpleNamespace(), max_treedepth=6)
    assert d3["divergences"] == 0
