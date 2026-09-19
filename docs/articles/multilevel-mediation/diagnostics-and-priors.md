# Diagnostics and priors

## Priors are explicit

`create_mediation_prior_specification()` exposes all default scales. The defaults are weakly informative, not silently data-adaptive. Use prior predictive simulation to determine whether those priors imply plausible mediator and outcome distributions in the declared design.

```python
spec = specify_multilevel_gaze_mediation(
    prepared,
    mediator_family="gaussian",
    outcome_family="bernoulli",
    priors=create_mediation_prior_specification(coefficient_sd=0.75),
    missingness_policy="quality_eligible",
)

prior_predictive_check_mediation(spec, draws=200, seed=2026)
```

## Required sampler diagnostics

Before interpreting an indirect effect, inspect:

- maximum R-hat;
- minimum bulk ESS;
- divergences;
- maximum-tree-depth hits;
- prior predictive behavior;
- posterior predictive behavior.

The standard extraction functions treat critical convergence failure as a blocker rather than a footnote.

## Prior sensitivity

Refit the same declared model under at least one defensible narrower and wider coefficient prior. Compare the indirect-effect distribution, direct paths, posterior predictive behavior, and convergence diagnostics. A result that changes materially under modest prior changes should be reported as prior-sensitive.

## Random slopes

Random slopes are optional and should reflect the repeated-measures design rather than a maximal-by-default rule. For the stable **simple** mediation model, the Python implementation recognizes `mediator_x`, `outcome_x`, and `outcome_m`. If both participant-specific `a` and `b` slopes are estimated, participant-specific indirect effects can be derived; otherwise the package does not fabricate them. Serial and moderated extensions currently use participant random intercepts only; non-empty random-slope requests are rejected explicitly rather than ignored.


## Worked sensitivity example

See [Worked example: prior sensitivity](prior-sensitivity-worked-example.md) for a reproducible narrow/default/wide coefficient-prior workflow and reporting language.
