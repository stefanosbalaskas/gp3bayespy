"""Optional Bayesian backend discovery."""

from __future__ import annotations

from importlib.util import find_spec


def backend_capabilities() -> dict[str, bool | str]:
    """Report optional Python backend availability without importing them."""
    return {
        "pymc": find_spec("pymc") is not None,
        "arviz": find_spec("arviz") is not None,
        "cmdstanpy": find_spec("cmdstanpy") is not None,
        "numpyro": find_spec("numpyro") is not None,
        "backend_policy": "optional; core contracts remain backend-independent",
    }
