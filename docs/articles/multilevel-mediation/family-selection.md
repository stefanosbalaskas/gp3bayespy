# Choosing mediator and outcome families

Do not choose a Gaussian likelihood merely because it is convenient. Match the family to the support and measurement process of the variable.

| Variable | Candidate family | Requirements / cautions |
| --- | --- | --- |
| approximately symmetric continuous dwell/score | Gaussian | inspect scale and residual behavior |
| strictly positive duration | lognormal or Gamma | zeros are invalid under these families |
| proportion strictly inside `(0, 1)` | beta | exact 0/1 values require another strategy |
| inspected vs not inspected | Bernoulli | values must be 0/1 |
| fixation/visit count | Poisson | equidispersion may be unrealistic |
| overdispersed count | negative binomial | non-negative integer counts |
| ordered response | ordinal | contiguous ordered integer categories |

## Zero-heavy gaze variables

A dwell variable with many genuine zeros is not compatible with the current lognormal or Gamma mediator implementation. Do not add a tiny constant silently. Either model a scientifically defensible transformed/continuous representation, use a binary inspection mediator, or treat a hurdle/mixture extension as a separate model-development task.

## Indirect-effect scale

For nonlinear likelihoods, the default path-product estimand is on the model's linear-predictor scale. `a_W × b_W` remains a useful coefficient-product summary, but it is not automatically a probability-scale natural indirect effect. Report the stored estimand scale explicitly.


## Worked family-contract example

See [Worked example: non-Gaussian mediator contracts](non-gaussian-family-contracts.md) for a deterministic support-validation workflow across lognormal, Gamma, beta, Bernoulli, Poisson, and negative-binomial mediators.
