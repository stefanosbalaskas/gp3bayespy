# Assumptions and limitations

Multilevel mediation preserves the repeated-measures structure, but it does not make causal mediation assumptions disappear. This page separates what the implementation estimates from what the research design must justify.

## Statistical structure

The simple model separates within-participant and between-participant paths. That separation prevents one common form of level conflation, but it still assumes that the declared likelihoods and hierarchical structure are reasonable representations of the observed data.

Random slopes should be added only where the design contains enough within-participant information to estimate them. A maximal random-effects structure is not automatically preferable when trial counts or predictor variation are weak.

## Causal identification is not automatic

A posterior distribution for `a_W × b_W` is not, by itself, a causal natural indirect effect. Causal interpretation additionally requires defensible temporal ordering and assumptions about confounding of the exposure→mediator and mediator→outcome relations. Treatment-induced mediator–outcome confounding is a particular concern in mediation settings.

When those assumptions are not defended, describe the result as a **multilevel indirect association/pathway parameter** rather than a causal mediated effect.

## Link-scale limitation for nonlinear outcomes

For Bernoulli, ordinal, Poisson, and negative-binomial outcomes, the implemented coefficient-product indirect effect is on the model's **linear-predictor product scale**. It is not a percentage-point effect and must not be labeled a probability-scale natural indirect effect.

A probability-scale or counterfactual indirect estimand would require an additional predictive/g-computation implementation plus explicitly stated identification assumptions. The current API deliberately does not manufacture that interpretation from a logit- or log-link coefficient product.

## Missingness and quality

The fitting API requires an explicit missingness policy. `complete_case` and `quality_eligible` analyses are conditional on the declared inclusion rule; neither magically solves informative missingness. If gaze loss or response missingness differs systematically by condition, participant, stimulus, or outcome tendency, report that pattern and perform sensitivity analyses where scientifically justified.

Absent gaze is never automatically equivalent to zero gaze.

## Measurement limitation

Dwell time, fixation count, first-entry latency, inspection indicators, and related metrics are process measurements. They do not directly establish attention, comprehension, trust, cognitive load, or another latent psychological construct. The substantive label for the mediator requires independent measurement validity evidence.

## Between-person paths in balanced within-subject designs

When every participant receives exactly the same exposure proportion, `X_between` can be constant. In that design, between-person exposure paths are not estimable from the observed data. The package omits those paths rather than allowing prior information to masquerade as data-supported estimation.

## Serial mediation

A serial chain such as `X → gaze → trust → decision` should be fit only when the temporal/design ordering supports that chain. Adding available variables into a longer path does not establish the ordering. Serial indirect products also compound uncertainty across multiple paths and should be interpreted accordingly.

## Moderated mediation

A moderator must be identified at its correct level (trial-varying or participant-level). Conditional indirect effects depend on the scale and coding of the moderator. Report the moderator values at which effects are summarized; do not present a single conditional contrast as universally representative.

## Backend and convergence limitations

The package exposes R-hat, ESS, divergence, and tree-depth checks where the backend provides them. Passing those diagnostics does not prove model adequacy, while failing critical diagnostics blocks the default effect extractors. Posterior predictive checking, prior sensitivity, and design-support review remain necessary.

## Generalization

Participant-level random effects account for dependence and heterogeneity in the sampled data. They do not by themselves demonstrate that the mediated pathway generalizes to new populations, devices, tasks, stimuli, or acquisition settings. Generalization requires an external or held-out design appropriate to that claim.

## Recommended sensitivity set

At minimum, consider sensitivity to:

1. defensible narrower/wider coefficient priors;
2. missingness policy where more than one policy is scientifically plausible;
3. mediator likelihood when distributional choice is uncertain;
4. random-slope structure where supported by the design;
5. influential participants/trials and posterior predictive misfit;
6. alternative operationalizations of the gaze mediator when construct validity is uncertain.

Do not select the sensitivity result that gives the preferred indirect effect. Report material changes across defensible specifications.
