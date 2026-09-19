# Serial and moderated extensions

## Serial mediation

Serial mediation is appropriate only when the design supports the ordering, for example:

```text
AI error → source inspection → trust → override decision
```

Prepare the second mediator in eyeprocesspy first:

```python
prepared = add_multilevel_mediation_component(
    prepared,
    value_col="trust",
    semantic="mediator2",
    within_col="M2_within",
    between_col="M2_between",
)

fit = fit_multilevel_serial_gaze_mediation(
    prepared,
    mediator_family="lognormal",
    mediator2_family="gaussian",
    outcome_family="bernoulli",
    missingness_policy="quality_eligible",
)
```

The serial within-person estimand is the posterior product `a1_W × d_W × b2_W`. Specific indirect paths through the first or second mediator should be reported separately when substantively meaningful.

## Moderated mediation

Prepare the moderator decomposition in eyeprocesspy rather than inside gp3bayespy:

```python
prepared = add_multilevel_mediation_component(
    prepared,
    value_col="warning_salience",
    semantic="moderator",
    within_col="Z_within",
    between_col="Z_between",
)

fit = fit_multilevel_moderated_gaze_mediation(
    prepared,
    moderation_path="a",
    moderator_component="within",
)

low = posterior_conditional_indirect_effect(fit, moderator_value=-1)
high = posterior_conditional_indirect_effect(fit, moderator_value=1)
```

The conditional indirect effect depends on the chosen moderator scale. Report the exact moderator values used and whether the moderator component is within- or between-participant.

## Guardrail

Do not add serial links or moderators because they improve fit. The package requires explicit model structure; it does not automatically search over mediation chains or interaction placements.


## Current random-effects scope

The simple mediation model supports declared participant random slopes when the corresponding within-participant predictor varies sufficiently. The current serial and moderated extensions deliberately use participant random intercepts only. In Python, non-empty `random_slopes` requests for these extensions fail explicitly; the R implementation follows the same scientific limitation. This prevents an API argument from suggesting heterogeneity that the fitted model does not actually estimate.

## Between-person estimability

Serial and moderated builders construct between-person paths only when the observed prepared components vary between participants. A balanced within-subject exposure can therefore yield a valid within serial/conditional indirect effect while leaving the corresponding between-X paths unestimated. This is a design property, not a convergence failure.
