from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

import gp3bayespy.backends as b
import gp3bayespy.postfit_exploration as postfit
from gp3bayespy.exceptions import GP3BayesError


@dataclass
class Demo:
    x: int
    nested: dict[str, object]


def test_backend_capabilities_environment_aliases_and_strict(monkeypatch):
    monkeypatch.setattr(b, "_available", lambda package: False)
    monkeypatch.setattr(b, "_version", lambda package: None)
    caps = b.backend_capabilities()
    assert list(caps["backend"]) == ["pymc", "cmdstanpy"]
    assert not caps["ready_for_package_interface"].any()

    monkeypatch.setattr(b, "backend_capabilities", lambda: caps)
    env = b.validate_backend_environment("pymc")
    assert env.status == "fail"
    assert env.to_frame().equals(env.checks)
    alias = b.validate_backend_environment("rstan")
    assert alias.backend == "pymc"
    cmd_alias = b.validate_backend_environment("cmdstanr")
    assert cmd_alias.backend == "cmdstanpy"

    prereq = b.validate_backend_environment("cmdstanpy", compile_test=True)
    compiler = prereq.checks.loc[prereq.checks["check"] == "compiler_smoke_test"].iloc[0]
    assert compiler["status"] == "not_assessed"
    assert compiler["detail"] == "prerequisite check failed"

    with pytest.raises(GP3BayesError):
        b.validate_backend_environment("bad")
    with pytest.raises(GP3BayesError):
        b.validate_backend_environment("pymc", strict=True)


def test_draw_summary_and_backend_parity_success_review_and_errors(monkeypatch):
    left = pd.DataFrame(
        {
            "variable": ["a", "b"],
            "mean": [0.0, 1.0],
            "sd": [1.0, 2.0],
            "mcse_mean": [0.1, 0.2],
        }
    )
    right = left.copy()
    right["mean"] += [0.05, -0.05]
    right["sd"] *= [1.02, 0.99]

    summary = b._draw_summary(left)
    assert list(summary.columns) == ["variable", "mean", "sd", "mcse_mean"]

    no_mcse = b._draw_summary(left.drop(columns="mcse_mean"))
    assert no_mcse["mcse_mean"].isna().all()

    parity = b.audit_backend_parity(left, right)
    assert parity.status == "pass"
    assert parity.to_frame().equals(parity.table)
    assert not parity.identical_draws_expected
    assert not parity.model_adequacy_established

    review_right = right.copy()
    review_right.loc[0, "mean"] = 10.0
    review = b.audit_backend_parity(left, review_right)
    assert review.status == "review"

    missing = pd.concat(
        [
            right,
            pd.DataFrame({"variable": ["c"], "mean": [0.0], "sd": [1.0], "mcse_mean": [0.1]}),
        ],
        ignore_index=True,
    )
    missing_audit = b.audit_backend_parity(left, missing)
    assert missing_audit.status == "review"
    assert missing_audit.missing_from_left == ("c",)

    for kwargs in (
        {"mcse_multiplier": -1},
        {"absolute_tolerance": -1},
        {"relative_sd_tolerance": -1},
    ):
        with pytest.raises(GP3BayesError):
            b.audit_backend_parity(left, right, **kwargs)

    with pytest.raises(GP3BayesError):
        b.audit_backend_parity(
            left.loc[left["variable"] == "a"],
            right.loc[right["variable"] == "b"],
        )

    draws = pd.DataFrame(
        {
            "chain": [1, 1, 2, 2],
            "draw": [1, 2, 1, 2],
            "a": [0.0, 0.2, -0.1, 0.1],
            "b": [1.0, 1.2, 0.8, 1.1],
            "empty": [np.nan] * 4,
        }
    )
    monkeypatch.setattr(
        postfit,
        "extract_posterior_draws",
        lambda x, variables=None, format="dataframe": draws,
    )
    derived = b._draw_summary(object())
    assert set(derived["variable"]) == {"a", "b"}

    monkeypatch.setattr(
        postfit,
        "extract_posterior_draws",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad")),
    )
    with pytest.raises(GP3BayesError):
        b._draw_summary(object())


def test_schema_node_capture_compare_validate_freeze_and_read(tmp_path):
    obj = Demo(1, {"a": [1, 2], "frame": pd.DataFrame({"x": [1, 2]})})
    schema = b.capture_gp3bayes_schema(obj, max_depth=4)
    assert schema.max_depth == 4
    assert schema.fields.iloc[0]["typeof"] == "dataclass"
    assert schema.to_dict()["object_class"][0] == "Demo"

    mapping_schema = b.capture_gp3bayes_schema(
        {"a": (1, 2), "b": [3], "frame": pd.DataFrame({"x": [1]})},
        max_depth=3,
    )
    kinds = set(mapping_schema.fields["typeof"])
    assert {"mapping", "tuple", "list", "DataFrame"}.issubset(kinds)

    for depth in (-1, True, 1.5):
        with pytest.raises(GP3BayesError):
            b.capture_gp3bayes_schema(obj, max_depth=depth)  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        b.capture_gp3bayes_schema(3)

    same = b.compare_gp3bayes_schemas(schema, schema, compare_lengths=True)
    assert same.status == "pass"
    assert same.to_frame().equals(same.table)

    changed = b.compare_gp3bayes_schemas(
        schema,
        b.capture_gp3bayes_schema(Demo(1, {"a": [1, 2, 3]}), max_depth=4),
        compare_lengths=True,
    )
    assert changed.status == "review"

    valid = b.validate_gp3bayes_schema(obj, schema, strict=True, compare_lengths=True)
    assert valid.status == "pass"
    with pytest.raises(GP3BayesError):
        b.validate_gp3bayes_schema(obj, object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        b.validate_gp3bayes_schema(
            Demo(2, {"other": [1]}),
            schema,
            strict=True,
            compare_lengths=True,
        )

    frozen_memory = b.freeze_gp3bayes_schema(schema)
    assert frozen_memory.frozen
    assert frozen_memory.frozen_at is not None

    path = tmp_path / "schema.json"
    frozen = b.freeze_gp3bayes_schema(schema, path)
    assert path.is_file()
    loaded = b.read_gp3bayes_schema(path)
    assert loaded.frozen
    assert loaded.object_class == frozen.object_class

    with pytest.raises(GP3BayesError):
        b.freeze_gp3bayes_schema(schema, path)
    overwrite = b.freeze_gp3bayes_schema(schema, path, overwrite=True)
    assert overwrite.frozen
    with pytest.raises(GP3BayesError):
        b.freeze_gp3bayes_schema(schema, tmp_path / "missing" / "schema.json")
    with pytest.raises(GP3BayesError):
        b.read_gp3bayes_schema(tmp_path / "absent.json")


def test_schema_comparison_presence_type_and_length_review_branches():
    left = b.capture_gp3bayes_schema({"a": [1, 2], "b": {"x": 1}}, max_depth=3)
    right = b.capture_gp3bayes_schema({"a": (1, 2, 3), "c": {"x": 1}}, max_depth=3)

    no_lengths = b.compare_gp3bayes_schemas(left, right, compare_lengths=False)
    with_lengths = b.compare_gp3bayes_schemas(left, right, compare_lengths=True)
    assert no_lengths.status == "review"
    assert with_lengths.status == "review"
    assert (
        ~with_lengths.table["reference_present"] | ~with_lengths.table["candidate_present"]
    ).any()
    assert (~with_lengths.table["same_type"]).any()
