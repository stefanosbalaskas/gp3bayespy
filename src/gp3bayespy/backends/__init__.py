"""Backend discovery, parity auditing, and structural schema governance.

This module adapts the frozen gp3bayes 0.5.0 backend-reliability contract to
Python.  PyMC is the primary optional sampling backend and CmdStanPy is the
portable CmdStan interface.  Capability checks never compile or fit a model
unless explicitly requested.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import GP3BayesError


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _available(package: str) -> bool:
    return find_spec(package) is not None


def backend_capabilities() -> pd.DataFrame:
    """Report approved optional Python backend capabilities without importing them."""
    pymc = _available("pymc")
    cmdstanpy = _available("cmdstanpy")
    cmdstan_ready = False
    cmdstan_version: str | None = None
    cmdstan_path: str | None = None
    if cmdstanpy:
        try:
            import cmdstanpy as csp  # type: ignore[import-untyped]

            cmdstan_path = csp.cmdstan_path()
            cmdstan_version = ".".join(map(str, csp.cmdstan_version()))  # type: ignore[arg-type]
            cmdstan_ready = bool(cmdstan_path)
        except Exception:
            pass
    return pd.DataFrame(
        {
            "backend": ["pymc", "cmdstanpy"],
            "backend_package_available": [pymc, cmdstanpy],
            "backend_package_version": [_version("pymc"), _version("cmdstanpy")],
            "external_runtime_available": [True, cmdstan_ready],
            "external_runtime_version": [None, cmdstan_version],
            "external_runtime_path": [None, cmdstan_path],
            "ready_for_package_interface": [pymc, cmdstanpy and cmdstan_ready],
            "algorithm": ["NUTS", "CmdStan NUTS"],
            "model_family_scope": [
                "Bernoulli-logit, positive lognormal duration, and governed pupil models"
            ]
            * 2,
            "unrestricted_modeling": [False, False],
        }
    )


@dataclass(slots=True)
class BackendEnvironment:
    validation_version: str
    backend: str
    status: str
    checks: pd.DataFrame
    capabilities: pd.DataFrame
    compile_test: bool
    model_fitted: bool = False

    def to_frame(self) -> pd.DataFrame:
        return self.checks.copy()


@dataclass(slots=True)
class BackendParityAudit:
    parity_version: str
    status: str
    table: pd.DataFrame
    missing_from_left: tuple[str, ...]
    missing_from_right: tuple[str, ...]
    settings: Mapping[str, float]
    identical_draws_expected: bool = False
    model_adequacy_established: bool = False

    def to_frame(self) -> pd.DataFrame:
        return self.table.copy()


@dataclass(slots=True)
class ObjectSchema:
    schema_version: str
    object_class: tuple[str, ...]
    max_depth: int
    fields: pd.DataFrame
    values_recorded: bool = False
    frozen: bool = False
    frozen_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "object_class": list(self.object_class),
            "max_depth": self.max_depth,
            "fields": self.fields.to_dict(orient="records"),
            "values_recorded": self.values_recorded,
            "frozen": self.frozen,
            "frozen_at": self.frozen_at,
        }


@dataclass(slots=True)
class SchemaComparison:
    comparison_version: str
    status: str
    table: pd.DataFrame
    reference_class: tuple[str, ...]
    candidate_class: tuple[str, ...]
    compare_lengths: bool
    value_equivalence_tested: bool = False

    def to_frame(self) -> pd.DataFrame:
        return self.table.copy()


@dataclass(slots=True)
class SchemaValidation:
    validation_version: str
    status: str
    schema: ObjectSchema
    candidate: ObjectSchema
    comparison: SchemaComparison
    schema_compatibility_only: bool = True


def validate_backend_environment(
    backend: str = "pymc",
    compile_test: bool = False,
    strict: bool = False,
) -> BackendEnvironment:
    """Validate one approved backend environment without fitting a model."""
    aliases = {"rstan": "pymc", "cmdstanr": "cmdstanpy"}
    backend = aliases.get(backend, backend)
    if backend not in {"pymc", "cmdstanpy"}:
        raise GP3BayesError("`backend` must be 'pymc' or 'cmdstanpy'.")
    caps = backend_capabilities()
    row = caps.loc[caps["backend"] == backend].reset_index(drop=True)
    package_ok = bool(row.loc[0, "backend_package_available"])
    runtime_ok = bool(row.loc[0, "external_runtime_available"])
    checks = [
        {
            "check": "backend_package",
            "status": "pass" if package_ok else "fail",
            "detail": row.loc[0, "backend_package_version"],
        },
        {
            "check": "external_runtime",
            "status": "pass" if runtime_ok else "fail",
            "detail": row.loc[0, "external_runtime_version"] or "managed by Python package",
        },
    ]
    compile_status = "not_assessed"
    compile_detail = "not requested"
    if compile_test:
        if not package_ok or not runtime_ok:
            compile_detail = "prerequisite check failed"
        elif backend == "cmdstanpy":
            try:
                import cmdstanpy as csp  # type: ignore[import-untyped]

                csp.cmdstan_path()
                compile_status = "pass"
                compile_detail = "CmdStan runtime path resolved"
            except Exception as exc:
                compile_status = "fail"
                compile_detail = str(exc)
        else:
            # Deliberately no compiler/model build. Importing PyMC is the Python
            # equivalent of a lightweight backend smoke for this contract.
            try:
                __import__("pymc")
                compile_status = "pass"
                compile_detail = "PyMC import smoke passed; no model was fitted"
            except Exception as exc:
                compile_status = "fail"
                compile_detail = str(exc)
    checks.append(
        {
            "check": "compiler_smoke_test",
            "status": compile_status,
            "detail": compile_detail,
        }
    )
    table = pd.DataFrame(checks)
    status = (
        "fail"
        if (table["status"] == "fail").any()
        else ("ready" if (table["status"] == "not_assessed").any() else "pass")
    )
    result = BackendEnvironment("0.2", backend, status, table, row, compile_test)
    if strict and status == "fail":
        raise GP3BayesError(f"Backend environment for {backend!r} is not ready.")
    return result


def _draw_summary(x: Any, variables: Sequence[str] | str | None = None) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame) and {"variable", "mean", "sd"}.issubset(x.columns):
        out = x.copy()
        if "mcse_mean" not in out:
            out["mcse_mean"] = np.nan
        return out[["variable", "mean", "sd", "mcse_mean"]]
    try:
        from ..postfit_exploration import extract_posterior_draws

        draws = extract_posterior_draws(x, variables=variables, format="dataframe")
    except Exception as exc:
        raise GP3BayesError(
            "Backend parity inputs must be approved fits or posterior-summary tables."
        ) from exc
    if not isinstance(draws, pd.DataFrame) or draws.empty:
        raise GP3BayesError("No posterior draws were available for backend parity auditing.")
    numeric = draws.select_dtypes(include=[np.number]).copy()
    numeric = numeric.drop(
        columns=[c for c in ("chain", "draw", ".chain", ".iteration") if c in numeric],
        errors="ignore",
    )
    rows: list[dict[str, Any]] = []
    for name in numeric.columns:
        z = numeric[name].to_numpy(dtype=float)
        z = z[np.isfinite(z)]
        if z.size == 0:
            continue
        sd = float(np.std(z, ddof=1)) if z.size > 1 else 0.0
        rows.append(
            {
                "variable": str(name),
                "mean": float(np.mean(z)),
                "sd": sd,
                "mcse_mean": sd / math.sqrt(z.size) if z.size else math.nan,
            }
        )
    return pd.DataFrame(rows)


def audit_backend_parity(
    rstan_fit: Any,
    cmdstanr_fit: Any,
    variables: Sequence[str] | str | None = None,
    mcse_multiplier: float = 3,
    absolute_tolerance: float = 0,
    relative_sd_tolerance: float = 0.10,
) -> BackendParityAudit:
    """Compare posterior summaries relative to Monte Carlo uncertainty."""
    if mcse_multiplier < 0 or absolute_tolerance < 0 or relative_sd_tolerance < 0:
        raise GP3BayesError("Backend parity tolerances must be non-negative.")
    left = _draw_summary(rstan_fit, variables)
    right = _draw_summary(cmdstanr_fit, variables)
    lv = list(dict.fromkeys(left["variable"].astype(str)))
    rv = list(dict.fromkeys(right["variable"].astype(str)))
    common = [v for v in lv if v in set(rv)]
    if not common:
        raise GP3BayesError("No common posterior variables were available for comparison.")
    left = left.set_index("variable").loc[common]
    right = right.set_index("variable").loc[common]
    combined = np.sqrt(
        left["mcse_mean"].to_numpy(float) ** 2 + right["mcse_mean"].to_numpy(float) ** 2
    )
    mean_diff = left["mean"].to_numpy(float) - right["mean"].to_numpy(float)
    allowed = np.maximum(float(absolute_tolerance), float(mcse_multiplier) * combined)
    mean_ok = np.isfinite(combined) & (np.abs(mean_diff) <= allowed)
    lsd = left["sd"].to_numpy(float)
    rsd = right["sd"].to_numpy(float)
    scale = np.maximum.reduce([np.abs(lsd), np.abs(rsd), np.full_like(lsd, np.finfo(float).eps)])
    rel_sd = np.abs(lsd - rsd) / scale
    sd_ok = np.isfinite(rel_sd) & (rel_sd <= float(relative_sd_tolerance))
    table = pd.DataFrame(
        {
            "variable": common,
            "left_mean": left["mean"].to_numpy(float),
            "right_mean": right["mean"].to_numpy(float),
            "mean_difference": mean_diff,
            "combined_mcse": combined,
            "allowed_mean_difference": allowed,
            "mean_within_mcse": mean_ok,
            "left_sd": lsd,
            "right_sd": rsd,
            "relative_sd_difference": rel_sd,
            "sd_within_tolerance": sd_ok,
        }
    )
    table["status"] = np.where(mean_ok & sd_ok, "pass", "review")
    missing_left = tuple(v for v in rv if v not in set(lv))
    missing_right = tuple(v for v in lv if v not in set(rv))
    status = (
        "pass"
        if (table["status"] == "pass").all() and not missing_left and not missing_right
        else "review"
    )
    return BackendParityAudit(
        "0.2",
        status,
        table,
        missing_left,
        missing_right,
        {
            "mcse_multiplier": float(mcse_multiplier),
            "absolute_tolerance": float(absolute_tolerance),
            "relative_sd_tolerance": float(relative_sd_tolerance),
        },
    )


def _schema_class(x: Any) -> tuple[str, ...]:
    return tuple(cls.__name__ for cls in type(x).mro() if cls is not object)


def _schema_node(x: Any, path: str, depth: int, max_depth: int) -> list[dict[str, Any]]:
    if dataclasses.is_dataclass(x):
        names = [f.name for f in dataclasses.fields(x)]
        children = [(name, getattr(x, name)) for name in names]
        kind = "dataclass"
    elif isinstance(x, Mapping):
        names = [str(k) for k in x]
        children = [(str(k), v) for k, v in x.items()]
        kind = "mapping"
    elif isinstance(x, (list, tuple)):
        names = [str(i + 1) for i in range(len(x))]
        children = [(str(i + 1), v) for i, v in enumerate(x)]
        kind = type(x).__name__
    elif isinstance(x, pd.DataFrame):
        names = [str(c) for c in x.columns]
        children = []
        kind = "DataFrame"
    else:
        names = []
        children = []
        kind = type(x).__name__
    try:
        length = len(x)  # type: ignore[arg-type]
    except Exception:
        length = 1
    row = {
        "path": path,
        "class": "/".join(_schema_class(x)),
        "typeof": kind,
        "length": int(length),
        "names": "|".join(names),
    }
    rows = [row]
    if depth < max_depth:
        for name, child in children:
            rows.extend(_schema_node(child, f"{path}${name}", depth + 1, max_depth))
    return rows


def capture_gp3bayes_schema(x: Any, max_depth: int = 3) -> ObjectSchema:
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 0:
        raise GP3BayesError("`max_depth` must be one non-negative integer.")
    # Python adaptation accepts any gp3bayespy dataclass/result or mapping used
    # by this package, while rejecting scalar primitives.
    module = type(x).__module__
    if not (
        module.startswith("gp3bayespy") or dataclasses.is_dataclass(x) or isinstance(x, Mapping)
    ):
        raise GP3BayesError("`x` must be a gp3bayespy object.")
    fields = pd.DataFrame(_schema_node(x, "root", 0, max_depth))
    return ObjectSchema("0.2", _schema_class(x), max_depth, fields)


def compare_gp3bayes_schemas(
    x: Any,
    y: Any,
    compare_lengths: bool = False,
) -> SchemaComparison:
    left = x if isinstance(x, ObjectSchema) else capture_gp3bayes_schema(x)
    right = y if isinstance(y, ObjectSchema) else capture_gp3bayes_schema(y)
    lf = left.fields.set_index("path")
    rf = right.fields.set_index("path")
    paths = list(dict.fromkeys([*lf.index.tolist(), *rf.index.tolist()]))
    rows: list[dict[str, Any]] = []
    for path in paths:
        lp = path in lf.index
        rp = path in rf.index
        lrow = lf.loc[path] if lp else None
        rrow = rf.loc[path] if rp else None
        same_class = bool(lp and rp and lrow["class"] == rrow["class"])  # type: ignore[index]
        same_type = bool(lp and rp and lrow["typeof"] == rrow["typeof"])  # type: ignore[index]
        same_names = bool(lp and rp and lrow["names"] == rrow["names"])  # type: ignore[index]
        same_length = bool(lp and rp and int(lrow["length"]) == int(rrow["length"]))  # type: ignore[index]
        ok = (
            lp
            and rp
            and same_class
            and same_type
            and same_names
            and (same_length or not compare_lengths)
        )
        rows.append(
            {
                "path": path,
                "reference_present": lp,
                "candidate_present": rp,
                "same_class": same_class,
                "same_type": same_type,
                "same_names": same_names,
                "same_length": same_length,
                "status": "pass" if ok else "review",
            }
        )
    table = pd.DataFrame(rows)
    status = "pass" if (table["status"] == "pass").all() else "review"
    return SchemaComparison(
        "0.2", status, table, left.object_class, right.object_class, compare_lengths
    )


def validate_gp3bayes_schema(
    x: Any,
    schema: ObjectSchema,
    strict: bool = False,
    compare_lengths: bool = False,
) -> SchemaValidation:
    if not isinstance(schema, ObjectSchema):
        raise GP3BayesError("`schema` must be a gp3bayespy ObjectSchema.")
    candidate = capture_gp3bayes_schema(x, max_depth=schema.max_depth)
    comparison = compare_gp3bayes_schemas(schema, candidate, compare_lengths)
    result = SchemaValidation("0.2", comparison.status, schema, candidate, comparison)
    if strict and result.status == "review":
        raise GP3BayesError("The object structure differs from the supplied schema.")
    return result


def freeze_gp3bayes_schema(
    schema: Any,
    file: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
) -> ObjectSchema:
    obj = schema if isinstance(schema, ObjectSchema) else capture_gp3bayes_schema(schema)
    obj = ObjectSchema(
        obj.schema_version,
        obj.object_class,
        obj.max_depth,
        obj.fields.copy(),
        obj.values_recorded,
        True,
        datetime.now(UTC).isoformat(),
    )
    if file is None:
        return obj
    path = Path(file)
    if path.exists() and not overwrite:
        raise GP3BayesError("The schema file already exists; use `overwrite=True` to replace it.")
    if not path.parent.exists():
        raise GP3BayesError("The parent directory of `file` does not exist.")
    path.write_text(json.dumps(obj.to_dict(), indent=2), encoding="utf-8")
    return obj


def read_gp3bayes_schema(file: str | os.PathLike[str]) -> ObjectSchema:
    path = Path(file)
    if not path.is_file():
        raise GP3BayesError("`file` must identify an existing schema file.")
    data = json.loads(path.read_text(encoding="utf-8"))
    return ObjectSchema(
        str(data["schema_version"]),
        tuple(data["object_class"]),
        int(data["max_depth"]),
        pd.DataFrame(data["fields"]),
        bool(data.get("values_recorded", False)),
        bool(data.get("frozen", False)),
        data.get("frozen_at"),
    )


__all__ = [
    "BackendEnvironment",
    "BackendParityAudit",
    "ObjectSchema",
    "SchemaComparison",
    "SchemaValidation",
    "audit_backend_parity",
    "backend_capabilities",
    "capture_gp3bayes_schema",
    "compare_gp3bayes_schemas",
    "freeze_gp3bayes_schema",
    "read_gp3bayes_schema",
    "validate_backend_environment",
    "validate_gp3bayes_schema",
]
