"""Backend-neutral post-fit exploration entry points."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .posterior import extract_draws


def extract_posterior_draws(
    fit: Any,
    variables: Sequence[str] | str | None = None,
    regex: str | None = None,
    format: str = "array",
) -> Any:
    """Extract posterior draws from an approved gp3bayespy fit.

    ``rvars`` is intentionally adapted from R posterior rvars to a mapping from
    variable names to ``chain × draw`` NumPy arrays.
    """
    return extract_draws(fit, variables=variables, regex=regex, format=format)
