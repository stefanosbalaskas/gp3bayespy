# Governance

`gp3bayespy` is intentionally conservative about what software output can establish. The package separates statistical evidence from analyst interpretation and keeps automatic decision-making disabled where the frozen gp3bayes contract requires that boundary.

## Package-wide rules

<div class="grid cards" markdown>

-   **No automatic model selection**

    Predictive scores, ELPD differences, and model weights are evidence. The package does not automatically declare a preferred model.

-   **No automatic exclusion**

    Readiness, influence, missingness, and measurement audits do not automatically remove participants, groups, trials, or observations.

-   **No automatic adequacy claim**

    Passing diagnostics does not itself establish that a model is substantively adequate.

-   **No automatic causal claim**

    Association estimates are not described as causal effects unless the study design and target estimand justify that interpretation.

-   **No latent-state inference**

    Behavioural, gaze, pupil, or physiological measurements are not automatically mapped to cognition, emotion, stress, comprehension, intention, personality, diagnosis, or deception.

-   **Explicit unsupported uses**

    Restricted model families and workflow contracts expose unsupported uses rather than silently widening the analysis scope.

</div>

## What diagnostics mean

A diagnostic result such as `pass`, `review`, or `fail` describes evidence under a declared rule. It does not substitute for a scientific conclusion.

The same principle applies to:

- convergence diagnostics;
- posterior predictive checks;
- calibration;
- PSIS-LOO and Pareto-*k*;
- sensitivity analyses;
- recovery and SBC;
- pupil temporal audits;
- backend parity checks.

## Reproducibility

Analysis manifests, frozen specifications, transformation replay, model cards, and publication bundles are designed to make decisions inspectable. They do not certify that those decisions are scientifically correct.

## Reference

These boundaries are inherited from the frozen **gp3bayes 0.5.0** reference and are part of Python parity rather than optional documentation language.
