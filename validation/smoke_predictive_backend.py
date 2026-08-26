"""Opt-in real PyMC smoke for governed predictive reconstruction."""

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
        seed=4601,
    )
    duration = gp.fit_duration_model(
        _duration_specification(),
        chains=1,
        iter=100,
        warmup=50,
        cores=1,
        seed=4602,
    )

    binary_grid = gp.create_prediction_grid(
        binary,
        variables="condition",
        at={"condition": [-0.5, 0.5]},
    )
    duration_grid = gp.create_prediction_grid(
        duration,
        variables="condition",
        at={"condition": [-0.5, 0.5]},
    )

    binary_expected = gp.predict_binary_probability(
        binary,
        binary_grid,
        ndraws=25,
    )
    binary_predictive = gp.extract_posterior_predictions(
        binary,
        binary_grid,
        ndraws=25,
        seed=4603,
    )
    binary_linear = gp.extract_linear_predictions(
        binary,
        binary_grid,
        ndraws=25,
    )

    duration_median = gp.predict_duration(
        duration,
        duration_grid,
        type="median",
        ndraws=25,
    )
    duration_expected = gp.predict_duration(
        duration,
        duration_grid,
        type="expected",
        ndraws=25,
    )
    duration_predictive = gp.predict_duration(
        duration,
        duration_grid,
        type="predictive",
        ndraws=25,
        seed=4604,
    )

    arrays = [
        binary_expected.draws,
        binary_predictive,
        binary_linear,
        duration_median.draws,
        duration_expected.draws,
        duration_predictive.draws,
    ]
    if any(array.shape != (25, 2) for array in arrays):
        raise RuntimeError("Predictive smoke returned an unexpected draw shape.")
    if any(not np.isfinite(array).all() for array in arrays):
        raise RuntimeError("Predictive smoke returned non-finite draws.")
    if not set(np.unique(binary_predictive)).issubset({0.0, 1.0}):
        raise RuntimeError("Binary posterior predictive draws must be Bernoulli.")
    if not np.all(duration_predictive.draws > 0):
        raise RuntimeError("Duration posterior predictive draws must remain positive.")
    if not np.all(duration_expected.draws >= duration_median.draws):
        raise RuntimeError("Lognormal expected predictions must not fall below medians.")
    if binary_expected.support.automatic_rejection:
        raise RuntimeError("Support auditing must not reject prediction rows automatically.")
    if (
        binary_expected.causal_effect_established
        or binary_expected.out_of_sample_adequacy_established
    ):
        raise RuntimeError("Prediction must not imply causal or adequacy claims.")

    conditional = gp.predict_model(
        binary,
        binary_grid,
        type="expected",
        include_group_effects=True,
        ndraws=25,
        seed=4605,
    )
    if np.allclose(conditional.draws, binary_expected.draws):
        raise RuntimeError("Conditional and population-level predictions unexpectedly match.")

    print("GPB-PY-06 real PyMC predictive smoke: PASS")
    print("Binary prediction shape:", binary_expected.draws.shape)
    print("Duration prediction shape:", duration_expected.draws.shape)
    print("Automatic support rejection:", binary_expected.support.automatic_rejection)
    print("Causal claim:", binary_expected.causal_effect_established)
    print(
        "Out-of-sample adequacy claim:",
        binary_expected.out_of_sample_adequacy_established,
    )


if __name__ == "__main__":
    main()
