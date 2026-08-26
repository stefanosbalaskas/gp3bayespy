"""Opt-in real PyMC/ArviZ smoke for posterior extraction and diagnostics."""

from __future__ import annotations

import gp3bayespy as gp
from smoke_fitting_backend import _binary_specification, _duration_specification


def _fit_pair():
    binary = gp.fit_binary_model(
        _binary_specification(),
        chains=2,
        iter=100,
        warmup=50,
        cores=1,
        seed=4501,
    )
    duration = gp.fit_duration_model(
        _duration_specification(),
        chains=2,
        iter=100,
        warmup=50,
        cores=1,
        seed=4502,
    )
    return binary, duration


def main() -> None:
    binary, duration = _fit_pair()

    binary_draws = gp.extract_posterior_draws(binary, format="matrix")
    duration_draws = gp.extract_posterior_draws(duration, format="matrix")
    if "b_Intercept" not in binary_draws.columns or "b_condition" not in binary_draws.columns:
        raise RuntimeError("Binary posterior canonical variable mapping failed.")
    if "sigma" not in duration_draws.columns:
        raise RuntimeError("Duration posterior did not retain sigma.")

    binary_summary = gp.summarise_binary_posterior(binary, probability=0.90)
    duration_summary = gp.summarise_duration_posterior(duration, probability=0.90)
    binary_diagnostics = gp.diagnose_binary_fit(binary)
    duration_diagnostics = gp.diagnose_duration_fit(duration)

    for summary in (binary_summary, duration_summary):
        if not summary.posterior_summarised:
            raise RuntimeError("Posterior summary did not record execution.")
        if summary.convergence_claim or summary.posterior_adequacy_established:
            raise RuntimeError("Posterior summary must not imply convergence or adequacy.")

    for diagnostics in (binary_diagnostics, duration_diagnostics):
        if not diagnostics.diagnostics_assessed:
            raise RuntimeError("Sampling diagnostics were not assessed.")
        if diagnostics.convergence_claim or diagnostics.posterior_adequacy_established:
            raise RuntimeError("Diagnostics must not imply convergence or adequacy.")

    print("GPB-PY-05 real PyMC/ArviZ posterior smoke: PASS")
    print("Binary diagnostic threshold status:", binary_diagnostics.status)
    print("Duration diagnostic threshold status:", duration_diagnostics.status)


if __name__ == "__main__":
    main()
