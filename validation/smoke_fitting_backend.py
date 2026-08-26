"""Opt-in real PyMC/NUTS smoke test for the restricted fitting bridge."""

from __future__ import annotations

import gp3bayespy as gp


def _binary_specification() -> gp.BinaryModelSpecification:
    simulation = gp.simulate_hierarchical_binary_data(
        n_participants=6,
        trials_per_participant=6,
        n_items=3,
        random_slope_sd=0.0,
        seed=4401,
    )
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        random_slope=False,
    )
    prepared = gp.prepare_hierarchical_binary_data(
        simulation.data,
        contract,
        condition_levels=["control", "treatment"],
    )
    return gp.specify_binary_model(prepared, baseline=0.35)


def _duration_specification() -> gp.DurationModelSpecification:
    simulation = gp.simulate_hierarchical_duration_data(
        n_participants=6,
        trials_per_participant=6,
        n_items=3,
        random_slope_sd=0.0,
        seed=4402,
    )
    contract = gp.create_model_contract(
        family="duration",
        outcome_col="duration",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        random_slope=False,
        outcome_unit="milliseconds",
    )
    prepared = gp.prepare_hierarchical_duration_data(
        simulation.data,
        contract,
        condition_levels=["control", "treatment"],
    )
    return gp.specify_duration_model(prepared, baseline=500.0)


def main() -> None:
    binary = gp.fit_binary_model(
        _binary_specification(),
        chains=1,
        iter=100,
        warmup=50,
        cores=1,
        seed=4403,
    )
    duration = gp.fit_duration_model(
        _duration_specification(),
        chains=1,
        iter=100,
        warmup=50,
        cores=1,
        seed=4404,
    )

    for fit in (binary, duration):
        if fit.backend_fit is None or fit.backend_model is None:
            raise RuntimeError("Backend smoke fit did not retain model and draws.")
        if not fit.fit_performed:
            raise RuntimeError("Backend smoke fit did not record sampling.")
        if fit.diagnostics_assessed or fit.posterior_adequacy_established:
            raise RuntimeError("Fitting must not imply diagnostics or adequacy.")

    print("GPB-PY-04 real PyMC/NUTS smoke: PASS")
    print("Binary backend versions:", dict(binary.package_versions))
    print("Duration backend versions:", dict(duration.package_versions))


if __name__ == "__main__":
    main()
