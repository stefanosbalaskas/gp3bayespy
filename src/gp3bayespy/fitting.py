"""Shared restricted-fitting infrastructure.

This module holds Python-native helpers used by the binary and duration fitting
ports.  The frozen R reference routes approved models through brms/rstan.  The
Python port preserves the same restricted-control and conservative-status
contracts while adapting execution to the optional PyMC NUTS backend.
"""

from __future__ import annotations

import math
import numbers
import os
import warnings
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from typing import Any, cast

import pandas as pd

from .exceptions import BackendUnavailableError, GP3BayesError
from .specification import PriorSpecification, validate_prior_specification


def _numeric_scalar(
    value: object,
    name: str,
    *,
    lower: float = -math.inf,
    upper: float = math.inf,
    lower_open: bool = False,
    upper_open: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(float(value))
    ):
        raise GP3BayesError(f"`{name}` must be one finite numeric value.")
    number = float(cast(Any, value))
    lower_ok = number > lower if lower_open else number >= lower
    upper_ok = number < upper if upper_open else number <= upper
    if not lower_ok or not upper_ok:
        left = "(" if lower_open else "["
        right = ")" if upper_open else "]"
        raise GP3BayesError(f"`{name}` must lie in {left}{lower:g}, {upper:g}{right}.")
    return number


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    number = _numeric_scalar(value, name, lower=float(minimum))
    if number != math.floor(number):
        raise GP3BayesError(f"`{name}` must be integer-valued.")
    return int(number)


def _default_cores(chains: int) -> int:
    detected = os.cpu_count() or 1
    return min(int(chains), int(detected), 2)


@dataclass(frozen=True, slots=True)
class _SamplingControls:
    """Validated restricted MCMC controls shared by binary and duration fits."""

    chains: int
    iter: int
    warmup: int
    post_warmup_iterations: int
    cores: int
    seed: int
    adapt_delta: float
    max_treedepth: int
    refresh: int

    def as_dict(self) -> dict[str, int | float]:
        return {
            "chains": self.chains,
            "iter": self.iter,
            "warmup": self.warmup,
            "post_warmup_iterations": self.post_warmup_iterations,
            "cores": self.cores,
            "seed": self.seed,
            "adapt_delta": self.adapt_delta,
            "max_treedepth": self.max_treedepth,
            "refresh": self.refresh,
        }


def _validate_sampling_controls(
    *,
    chains: object,
    iter: object,
    warmup: object,
    cores: object | None,
    seed: object,
    adapt_delta: object,
    max_treedepth: object,
    refresh: object,
) -> _SamplingControls:
    chains_value = _integer(chains, "chains", minimum=1)
    iter_value = _integer(iter, "iter", minimum=100)
    warmup_value = _integer(warmup, "warmup", minimum=0)
    if warmup_value >= iter_value:
        raise GP3BayesError("`warmup` must be smaller than `iter`.")

    cores_value = (
        _default_cores(chains_value) if cores is None else _integer(cores, "cores", minimum=1)
    )
    if cores_value > chains_value:
        raise GP3BayesError("`cores` cannot exceed `chains`.")

    seed_value = _integer(seed, "seed", minimum=0)
    adapt_value = _numeric_scalar(
        adapt_delta,
        "adapt_delta",
        lower=0,
        upper=1,
        lower_open=True,
        upper_open=True,
    )
    tree_value = _integer(max_treedepth, "max_treedepth", minimum=5)
    refresh_value = _integer(refresh, "refresh", minimum=0)

    return _SamplingControls(
        chains=chains_value,
        iter=iter_value,
        warmup=warmup_value,
        post_warmup_iterations=iter_value - warmup_value,
        cores=cores_value,
        seed=seed_value,
        adapt_delta=adapt_value,
        max_treedepth=tree_value,
        refresh=refresh_value,
    )


def _prior_number(value: object) -> str:
    number = float(cast(Any, value))
    if not math.isfinite(number):
        raise GP3BayesError("Prior text cannot contain non-finite numeric values.")
    text = f"{number:.15f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _prior_row(priors: PriorSpecification, parameter_class: str) -> pd.Series:
    validate_prior_specification(priors)
    mask = priors.table["parameter_class"].eq(parameter_class)
    if int(mask.sum()) != 1:
        raise GP3BayesError(
            f"The prior specification must contain exactly one `{parameter_class}` row."
        )
    row = priors.table.loc[mask].iloc[0]
    return cast(pd.Series, row)


def _normal_prior_text(row: pd.Series) -> str:
    return f"normal({_prior_number(row['location'])}, {_prior_number(row['scale'])})"


def _student_t_prior_text(row: pd.Series) -> str:
    return (
        f"student_t({_prior_number(row['df'])}, {_prior_number(row['location'])}, "
        f"{_prior_number(row['scale'])})"
    )


def _lkj_prior_text(row: pd.Series) -> str:
    return f"lkj({_prior_number(row['shape'])})"


def _translation_parameter_table(
    priors: PriorSpecification,
    *,
    include_sigma: bool,
    random_slope: bool,
) -> pd.DataFrame:
    required = ["Intercept", "b", "sd"]
    if include_sigma:
        required.append("sigma")
    if random_slope:
        required.append("cor")

    rows: list[dict[str, Any]] = []
    for parameter_class in required:
        row = _prior_row(priors, parameter_class)
        if parameter_class in {"Intercept", "b"}:
            prior_text = _normal_prior_text(row)
            backend_class = parameter_class
        elif parameter_class in {"sd", "sigma"}:
            prior_text = _student_t_prior_text(row)
            backend_class = parameter_class
        else:
            prior_text = _lkj_prior_text(row)
            # brms validates a correlation prior against its Cholesky factor
            # class `L`; keeping that compatibility class makes the R-derived
            # parity fixture directly auditable while execution is PyMC-native.
            backend_class = "L"
        rows.append(
            {
                "parameter_class": parameter_class,
                "class": backend_class,
                "source": "user",
                "prior": prior_text,
            }
        )
    return pd.DataFrame(rows)


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not_installed"


@cache
def _importable(package: str) -> bool:
    if find_spec(package) is None:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import_module(package)
    except Exception:
        return False
    return True


def _pymc_available() -> bool:
    return _importable("pymc")


def _require_pymc(purpose: str) -> None:
    if not _pymc_available():
        raise BackendUnavailableError(
            f"Optional package `pymc` is required to {purpose} and must import "
            "successfully. Install or repair the `gp3bayespy[bayes]` extra."
        )


def _load_pymc() -> Any:
    """Import PyMC only after the optional-backend gate has passed."""
    return import_module("pymc")


def _backend_versions() -> dict[str, str]:
    return {
        "pymc": _package_version("pymc"),
        "arviz": _package_version("arviz"),
    }
