"""Opt-in real PyMC smoke for binary and duration posterior-predictive checks."""

from __future__ import annotations

import numpy as np
from smoke_fitting_backend import _binary_specification, _duration_specification

import gp3bayespy as gp


def main() -> None:
    binary = gp.fit_binary_model(
        _binary_specification(),
        chains=1,
        iter=100,
        warmup=50,
        cores=1,
        seed=4701,
    )
    duration = gp.fit_duration_model(
        _duration_specification(),
        chains=1,
        iter=100,
        warmup=50,
        cores=1,
        seed=4702,
    )

    binary_ppc = gp.check_binary_posterior_predictive(
        binary,
        draws=50,
        seed=4703,
    )
    duration_ppc = gp.check_duration_posterior_predictive(
        duration,
        draws=50,
        seed=4704,
    )

    if binary_ppc.draws != 50 or duration_ppc.draws != 50:
        raise RuntimeError("Posterior-predictive smoke returned the wrong draw count.")
    if not np.isfinite(binary_ppc.brier_score):
        raise RuntimeError("Binary posterior-predictive Brier score must be finite.")
    if not np.isfinite(duration_ppc.log_scale_rmse):
        raise RuntimeError("Duration posterior-predictive log-RMSE must be finite.")
    if not np.isfinite(duration_ppc.replicated.to_numpy(dtype=float)).all():
        raise RuntimeError("Duration posterior-predictive summaries must be finite.")
    if binary_ppc.adequacy_established or duration_ppc.adequacy_established:
        raise RuntimeError("Posterior-predictive checks must not establish global adequacy.")

    print("GPB-PY-07 real PyMC posterior-predictive smoke: PASS")
    print("Binary PPC status:", binary_ppc.status)
    print("Duration PPC status:", duration_ppc.status)
    print("Binary Brier score:", binary_ppc.brier_score)
    print("Duration log-RMSE:", duration_ppc.log_scale_rmse)
    print("Global adequacy established: False")


if __name__ == "__main__":
    main()
