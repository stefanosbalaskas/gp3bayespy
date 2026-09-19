# Analysis plan and preregistration template

A multilevel mediation analysis is easier to interpret when the estimand, data levels, missingness rule, likelihoods, random effects, priors, diagnostics, and sensitivity analyses are declared **before** fitting the observed data. This template turns those choices into a compact analysis contract.

## 1. Research question and temporal order

Write the proposed mechanism in temporal order and identify the observational unit.

> **Example:** On each trial, AI correctness (`X`) precedes source inspection (`M`), which precedes the override decision (`Y`). Trials are nested within participants. The primary target is the within-participant pathway linking trial-to-trial variation in AI correctness to source inspection and then to the decision outcome.

If the mediator is measured after the outcome, or if the proposed ordering cannot be justified from the task design, preregister the analysis as associational rather than causal mediation.

## 2. Primary estimand

Declare the level and scale of the primary indirect effect.

> **Primary estimand:** within-participant coefficient-product indirect effect, `a_within × b_within`, on the outcome model's linear-predictor product scale.

When the outcome is Bernoulli, ordinal, Poisson, or negative binomial, do not preregister the coefficient product as a probability-scale natural indirect effect. A probability-scale/counterfactual estimand requires a separate implementation and additional identification assumptions.

If a between-participant indirect effect is planned, state that it will be reported only when the observed design supports both between-person paths.

## 3. Preparation contract

Declare the columns and preparation rules before model fitting.

```python
prepared = prepare_multilevel_mediation_data(
    trials,
    participant_col="participant_id",
    trial_col="trial_id",
    x_col="ai_correct",
    mediator_col="source_dwell",
    outcome_col="correct_override",
    quality_col="valid_fraction",
    minimum_quality=0.80,
    quality_action="flag",
    warn=False,
)
```

Predeclare:

- the participant and trial identifiers;
- exposure, mediator, and outcome variables;
- within/between decomposition convention;
- whether a genuine observed zero is possible for the mediator;
- quality variable and threshold;
- whether quality problems are flagged or explicitly masked;
- provenance fields to retain.

Do not silently delete incomplete or poor-quality rows during preparation.

## 4. Missingness policy

Choose the inferential missingness policy after reviewing the preparation audits, but before inspecting the fitted indirect effect.

> **Example:** Primary analysis uses `missingness_policy="quality_eligible"`. Missing mediator values are not recoded as zero. A complete-case analysis is reported as a sensitivity specification only if scientifically defensible.

If gaze loss or response missingness may depend on condition, participant, stimulus, or outcome tendency, preregister how those patterns will be described and which sensitivity analyses will be considered.

## 5. Likelihoods

Declare the mediator and outcome families based on measurement support, not convenience.

| Variable type | Candidate family | Key support rule |
| --- | --- | --- |
| approximately symmetric continuous | Gaussian | inspect tails and scale |
| strictly positive duration | lognormal or Gamma | no exact zero support |
| proportion strictly inside (0,1) | beta | exact 0/1 require another model or justified handling |
| binary inspection indicator | Bernoulli | define success explicitly |
| count | Poisson / negative binomial | inspect overdispersion |
| binary decision | Bernoulli-logit | indirect product is on link scale |

Record the chosen family and the reason for rejecting plausible alternatives. See [Choosing mediator and outcome families](family-selection.md) and [Worked non-Gaussian family contracts](non-gaussian-family-contracts.md).

## 6. Random-effects structure

Declare the participant random intercepts and any random slopes requested by the design.

> **Example:** participant random intercepts in both submodels and a participant-varying exposure→mediator slope (`mediator_x`). A participant-specific indirect effect will only be reported when both participant-specific `a` and `b` slopes are actually estimated.

Do not preregister a maximal random-effects structure solely as a convention. The requested slopes must be supported by repeated-measures variation and trial counts. Serial and moderated extensions should retain the currently implemented random-effects boundary rather than silently requesting unsupported slopes.

## 7. Priors

Declare the default prior specification and at least one defensible narrower/wider sensitivity scenario.

```python
primary_priors = create_mediation_prior_specification(
    intercept_sd=2.5,
    coefficient_sd=1.0,
    group_sd_scale=1.0,
)
```

> **Sensitivity scenarios:** coefficient `sd = 0.5` and `sd = 1.5`, with the same data, likelihoods, missingness policy, and random-effects structure.

Use prior predictive checks to determine whether the declared priors generate scientifically plausible mediator/outcome values. See [Diagnostics and priors](diagnostics-and-priors.md) and [Worked example: prior sensitivity](prior-sensitivity-worked-example.md).

## 8. Sampling plan and convergence gate

Record sampler settings prospectively.

```python
fit = fit_multilevel_gaze_mediation(
    prepared,
    mediator_family="gaussian",
    outcome_family="bernoulli",
    priors=primary_priors,
    random_slopes=("mediator_x",),
    missingness_policy="quality_eligible",
    chains=4,
    draws=1000,
    tune=1000,
    seed=2026,
)
```

Predeclare that substantive effect extraction requires the convergence gate to pass. At minimum report:

- maximum R-hat;
- minimum bulk ESS;
- divergences;
- maximum-tree-depth hits;
- prior predictive checks;
- posterior predictive checks.

Do not redefine the convergence threshold after seeing the preferred effect estimate.

## 9. Simulation/recovery validation

Before interpreting the observed-data fit, evaluate the planned workflow under synthetic weak, moderate, and strong known-truth scenarios that resemble the participant/trial structure of the intended study.

Summarize bias, credible-interval coverage, convergence failures, predictive behavior, and the frequency with which requested paths are estimable. See [Simulation and recovery planning](simulation-and-recovery.md).

Good recovery supports the analysis pipeline under the simulated conditions; it does not establish the real-data causal mechanism.

## 10. Model comparison

If candidate models will be compared, state the candidates and keep the pointwise observations identical.

> **Example:** compare random-intercept and supported random-slope specifications fitted to the same ordered analysis rows.

`compare_multilevel_mediation_models()` refuses comparisons when mediator/outcome observations or participant-trial ordering differ. Do not use PSIS-LOO to compare models created from different exclusion or missingness sets. See [Model comparison for multilevel mediation](model-comparison.md).

## 11. Sensitivity analyses

Separate sensitivity dimensions rather than changing several decisions at once. A useful preregistered set may include:

1. narrower and wider coefficient priors;
2. an alternative mediator family when measurement support is genuinely ambiguous;
3. supported random-slope alternatives;
4. an alternative missingness policy only when both policies are scientifically defensible;
5. alternative mediator operationalizations chosen independently of the observed indirect effect;
6. influential-participant or trial diagnostics without automatic deletion.

Report material changes across all defensible specifications rather than selecting the specification yielding the largest or most favorable indirect effect.

## 12. Reporting plan

Predeclare the minimum reporting set:

- participant and analyzed-trial counts;
- preparation and quality rules;
- missingness policy and excluded-row count;
- within/between decomposition;
- mediator/outcome families;
- random-effects structure;
- prior scales;
- sampler settings and seed;
- convergence diagnostics;
- prior and posterior predictive checks;
- direct, indirect, and total effects on their correct scales;
- non-estimable paths;
- sensitivity results;
- limitations and causal-identification boundaries;
- software versions and provenance.

Use [Reporting guidance](reporting.md) and the [Worked reporting example](reporting-example.md) as manuscript templates.

## Filled preregistration example

A concise preregistered statement could read:

> The primary analysis will estimate a within-participant trial-level mediation pathway from AI correctness to source dwell to binary correct override. Trial observations will be prepared with explicit participant-mean decomposition; genuine zero dwell will remain distinct from missing or quality-ineligible gaze. The mediator will use a Gaussian likelihood and the outcome a Bernoulli-logit likelihood. The primary model will include participant random intercepts and a supported exposure→mediator random slope. The primary missingness policy will be `quality_eligible`. Coefficient priors will use `sd = 1.0`, with `sd = 0.5` and `sd = 1.5` sensitivity specifications. Indirect effects will be reported on the linear-predictor product scale and extracted only after the declared convergence gate passes. Simulation/recovery, prior/posterior predictive checks, and prespecified sensitivity analyses will accompany the substantive fit. Between-person paths will be reported only when the observed design supports them, and PSIS-LOO comparisons will use identical ordered trial observations.

## Related API

- `prepare_multilevel_mediation_data()`
- `validate_multilevel_mediation_data()`
- `audit_mediation_missingness()`
- `check_mediation_trial_counts()`
- `create_mediation_prior_specification()`
- `specify_multilevel_gaze_mediation()`
- `fit_multilevel_gaze_mediation()`
- `check_mediation_convergence()`
- `prior_predictive_check_mediation()`
- `posterior_predictive_check_mediation()`
- `estimate_within_indirect_effect()`
- `compare_multilevel_mediation_models()`
- `report_multilevel_gaze_mediation()`
