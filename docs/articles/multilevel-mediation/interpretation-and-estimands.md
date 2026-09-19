# Interpretation and estimands

## Within-participant indirect effect

The within effect is the posterior distribution of `a_W × b_W` after participant-mean decomposition. It asks whether trial-to-trial changes in the exposure relative to a participant's own mean are associated with trial-to-trial changes in the mediator, which are in turn associated with the outcome conditional on the exposure path.

## Between-participant indirect effect

The between effect is `a_B × b_B`. It describes participant-level differences in mean exposure and mean mediator. It is not interchangeable with the within mechanism, even when both products have the same sign.

## Direct and total effects

`posterior_direct_effect()` returns the relevant `c'` path. `posterior_total_effect()` adds the coefficient-product indirect path on the same linear-predictor scale. For nonlinear models, do not describe this as a probability-scale decomposition unless a separate predictive/counterfactual estimand has been implemented.

## Participant-specific indirect effects

Participant-specific effects are available only when both participant-specific `a` and `b` slopes are estimated. The package deliberately refuses to manufacture participant indirect effects from a model that contains only one of those random slopes.

## Causal scope

Hierarchical mediation does not by itself identify a causal indirect effect. Causal language depends on design, temporal ordering, intervention structure, confounding assumptions, treatment-induced mediator-outcome confounding, and the estimand being used. When those assumptions are not defended, report the paths as multilevel mediation/association parameters rather than causal natural indirect effects.
