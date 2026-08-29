from __future__ import annotations

import numpy as np
import pytest

import gp3bayespy as gp
import gp3bayespy.binary as binary
import gp3bayespy.duration as duration


class _FakeModel:
    def __init__(self, owner):
        self.owner = owner

    def __enter__(self):
        self.owner.context_entries += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self.owner.context_exits += 1
        return False


class _FakeHalfStudentT:
    def __init__(self, owner):
        self.owner = owner

    def __call__(self, name, *, nu, sigma):
        self.owner.half_student_calls.append(
            {
                "name": name,
                "nu": float(nu),
                "sigma": float(sigma),
            }
        )
        return 1.0

    def dist(self, *, nu, sigma):
        self.owner.half_student_dist_calls.append(
            {
                "nu": float(nu),
                "sigma": float(sigma),
            }
        )
        return ("HalfStudentT.dist", float(nu), float(sigma))


class _FakeMath:
    @staticmethod
    def dot(a, b):
        return np.dot(a, b)


class _FakePM:
    def __init__(self):
        self.context_entries = 0
        self.context_exits = 0
        self.normal_calls = []
        self.half_student_calls = []
        self.half_student_dist_calls = []
        self.lkj_calls = []
        self.deterministic_calls = []
        self.bernoulli_calls = []
        self.lognormal_calls = []
        self.sample_calls = []
        self.math = _FakeMath()
        self.HalfStudentT = _FakeHalfStudentT(self)

    def Model(self):
        return _FakeModel(self)

    def Normal(self, name, *, mu, sigma, shape=None):
        self.normal_calls.append(
            {
                "name": name,
                "mu": float(mu),
                "sigma": float(sigma),
                "shape": shape,
            }
        )
        if shape is None:
            return 0.0
        return np.zeros(shape, dtype=float)

    def LKJCholeskyCov(
        self,
        name,
        *,
        n,
        eta,
        sd_dist,
        compute_corr,
    ):
        self.lkj_calls.append(
            {
                "name": name,
                "n": int(n),
                "eta": float(eta),
                "sd_dist": sd_dist,
                "compute_corr": bool(compute_corr),
            }
        )
        return np.eye(int(n)), np.ones(int(n)), np.eye(int(n))

    def Deterministic(self, name, value):
        self.deterministic_calls.append(name)
        return np.asarray(value)

    def Bernoulli(self, name, *, logit_p, observed):
        self.bernoulli_calls.append(
            {
                "name": name,
                "logit_p": np.asarray(logit_p, dtype=float),
                "observed": np.asarray(observed),
            }
        )
        return None

    def LogNormal(self, name, *, mu, sigma, observed):
        self.lognormal_calls.append(
            {
                "name": name,
                "mu": np.asarray(mu, dtype=float),
                "sigma": float(sigma),
                "observed": np.asarray(observed, dtype=float),
            }
        )
        return None

    def sample(self, **kwargs):
        self.sample_calls.append(dict(kwargs))
        return {
            "kind": "fake_inferencedata",
            "draws": int(kwargs["draws"]),
            "tune": int(kwargs["tune"]),
        }


def _controls():
    return {
        "chains": 2,
        "iter": 120,
        "warmup": 20,
        "post_warmup_iterations": 100,
        "cores": 1,
        "seed": 2501,
        "adapt_delta": 0.91,
        "max_treedepth": 9,
        "refresh": 1,
    }


def _binary_spec(*, random_slope: bool, include_item: bool, seed: int):
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
        item_col="item_id" if include_item else None,
        trial_col="trial_id",
        condition_col="condition",
        predictors=("participant_covariate", "trial_covariate"),
        random_slope=random_slope,
    )
    prepared = gp.prepare_hierarchical_binary_data(sim.data, contract)
    return gp.specify_binary_model(prepared)


def _duration_spec(*, random_slope: bool, include_item: bool, seed: int):
    sim = gp.simulate_hierarchical_duration_data(
        n_participants=4,
        trials_per_participant=6,
        n_items=3,
        seed=seed,
    )
    contract = gp.create_model_contract(
        family="duration",
        outcome_col="duration",
        participant_col="participant_id",
        item_col="item_id" if include_item else None,
        trial_col="trial_id",
        condition_col="condition",
        predictors=("participant_covariate", "trial_covariate"),
        random_slope=random_slope,
        outcome_unit="milliseconds",
    )
    prepared = gp.prepare_hierarchical_duration_data(sim.data, contract)
    return gp.specify_duration_model(prepared, baseline=500.0)


@pytest.mark.parametrize(
    ("random_slope", "include_item"),
    [
        (False, True),
        (True, False),
    ],
)
def test_binary_pymc_kernel_with_deterministic_backend_double(
    monkeypatch,
    random_slope,
    include_item,
):
    spec = _binary_spec(
        random_slope=random_slope,
        include_item=include_item,
        seed=2510 + int(random_slope) + 10 * int(include_item),
    )
    fake = _FakePM()
    monkeypatch.setattr(binary, "_load_pymc", lambda: fake)

    model, idata = binary._run_binary_pymc(spec, _controls())

    assert isinstance(model, _FakeModel)
    assert idata == {
        "kind": "fake_inferencedata",
        "draws": 100,
        "tune": 20,
    }
    assert fake.context_entries == fake.context_exits == 1
    assert len(fake.bernoulli_calls) == 1
    call = fake.bernoulli_calls[0]
    assert call["name"] == "observed"
    assert len(call["observed"]) == len(spec.prepared.data)
    assert set(np.unique(call["observed"])).issubset({0, 1})

    sample = fake.sample_calls[0]
    assert sample["draws"] == 100
    assert sample["tune"] == 20
    assert sample["chains"] == 2
    assert sample["cores"] == 1
    assert sample["random_seed"] == 2501
    assert sample["progressbar"] is True
    assert sample["compute_convergence_checks"] is False
    assert sample["return_inferencedata"] is True
    assert sample["idata_kwargs"] == {"log_likelihood": True}
    assert sample["nuts"] == {
        "target_accept": 0.91,
        "max_treedepth": 9,
    }

    normal_names = {row["name"] for row in fake.normal_calls}
    assert "b_Intercept" in normal_names
    assert "b" in normal_names

    if random_slope:
        assert fake.lkj_calls
        assert "participant_z" in normal_names
        assert "participant_re" in fake.deterministic_calls
        assert not any(row["name"] == "sd_participant" for row in fake.half_student_calls)
    else:
        assert not fake.lkj_calls
        assert any(row["name"] == "sd_participant" for row in fake.half_student_calls)

    if include_item:
        assert any(row["name"] == "sd_item" for row in fake.half_student_calls)
        assert "item_z" in normal_names
    else:
        assert not any(row["name"] == "sd_item" for row in fake.half_student_calls)


@pytest.mark.parametrize(
    ("random_slope", "include_item"),
    [
        (False, True),
        (True, False),
    ],
)
def test_duration_pymc_kernel_with_deterministic_backend_double(
    monkeypatch,
    random_slope,
    include_item,
):
    spec = _duration_spec(
        random_slope=random_slope,
        include_item=include_item,
        seed=2530 + int(random_slope) + 10 * int(include_item),
    )
    fake = _FakePM()
    monkeypatch.setattr(duration, "_load_pymc", lambda: fake)

    model, idata = duration._run_duration_pymc(spec, _controls())

    assert isinstance(model, _FakeModel)
    assert idata == {
        "kind": "fake_inferencedata",
        "draws": 100,
        "tune": 20,
    }
    assert fake.context_entries == fake.context_exits == 1
    assert len(fake.lognormal_calls) == 1
    call = fake.lognormal_calls[0]
    assert call["name"] == "observed"
    assert len(call["observed"]) == len(spec.prepared.data)
    assert np.all(call["observed"] > 0)
    assert call["sigma"] == 1.0

    sample = fake.sample_calls[0]
    assert sample["draws"] == 100
    assert sample["tune"] == 20
    assert sample["chains"] == 2
    assert sample["cores"] == 1
    assert sample["random_seed"] == 2501
    assert sample["progressbar"] is True
    assert sample["compute_convergence_checks"] is False
    assert sample["return_inferencedata"] is True
    assert sample["idata_kwargs"] == {"log_likelihood": True}
    assert sample["nuts"] == {
        "target_accept": 0.91,
        "max_treedepth": 9,
    }

    normal_names = {row["name"] for row in fake.normal_calls}
    assert "b_Intercept" in normal_names
    assert "b" in normal_names
    assert any(row["name"] == "sigma" for row in fake.half_student_calls)

    if random_slope:
        assert fake.lkj_calls
        assert "participant_z" in normal_names
        assert "participant_re" in fake.deterministic_calls
        assert not any(row["name"] == "sd_participant" for row in fake.half_student_calls)
    else:
        assert not fake.lkj_calls
        assert any(row["name"] == "sd_participant" for row in fake.half_student_calls)

    if include_item:
        assert any(row["name"] == "sd_item" for row in fake.half_student_calls)
        assert "item_z" in normal_names
    else:
        assert not any(row["name"] == "sd_item" for row in fake.half_student_calls)
