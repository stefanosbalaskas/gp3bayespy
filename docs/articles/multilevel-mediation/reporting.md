# Reporting multilevel gaze mediation

A reproducible report should state the observational level, decomposition convention, mediator and outcome families, random effects, explicit missingness policy, quality rule, priors, sampler settings, convergence diagnostics, posterior predictive checks, and the scale of the indirect effect.

## Minimum reporting template

> Trial-level observations were modeled with participant-specific hierarchical effects. The exposure and gaze mediator were decomposed into within- and between-participant components before modeling. The mediator used a [family] likelihood and the outcome used a [family] likelihood. Missing or quality-ineligible trials were handled using the predeclared [policy] rule; absent gaze was not coded as zero. Weakly informative priors were [describe]. Four chains were sampled for [draws] post-warmup draws after [warmup] warmup iterations. Maximum R-hat was [x], minimum bulk ESS was [x], with [x] divergences and [x] maximum-tree-depth hits. The within-participant indirect effect on the linear-predictor product scale was [estimate, interval].

## For nonlinear outcome families

Add an explicit sentence that a coefficient-product indirect effect is not a probability-scale natural indirect effect. Do not translate a logit-scale product into a percentage-point mediated effect without a separate predictive/counterfactual calculation.

## Failure reporting

If convergence fails, report the failure and the attempted remedial specifications. Do not publish the blocked indirect-effect summary as though it were valid. If missingness exclusions materially change the result, report the sensitivity rather than presenting only the preferred policy.
