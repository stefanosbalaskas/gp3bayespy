# Simulation and recovery planning

Simulation-based recovery asks whether a declared mediation workflow can recover known generating parameters under a design that resembles the intended study. It is a validation exercise for the **analysis pipeline and design**, not evidence that the real-data mediation mechanism is true.

## Why simulate before fitting the real study?

A hierarchical mediation model can be mathematically identifiable yet practically weak when there are too few participants, too few repeated trials, little within-person exposure variation, a rare outcome, substantial mediator noise, or a random-effects structure that the data cannot support. Simulation makes those weaknesses visible before substantive interpretation.

Use `simulate_multilevel_gaze_mediation()` to generate repeated trial-level data with declared path truths, then pass the result through the same `eyeprocesspy` preparation and `gp3bayespy` specification/fitting steps planned for the study.

## CI-small design-contract example

The repository ships:

```bash
python examples/multilevel_gaze_mediation_simulation_recovery_contract.py
```

It creates weak, moderate, and strong within-person mediation scenarios while keeping the data-preparation and model contract explicit:

```python
scenarios = {
    "weak": {"a_within": 0.10, "b_within": 0.10},
    "moderate": {"a_within": 0.45, "b_within": 0.55},
    "strong": {"a_within": 0.90, "b_within": 1.00},
}
```

The intended coefficient-product truths are therefore 0.01, 0.2475, and 0.90 on the simulator's linear-predictor product scale.

The CI-small example deliberately stops before sampling. Its purpose is to verify that the same trial-level preparation, within/between decomposition, likelihood declarations, random-slope request, missingness policy, and estimability checks can be materialized reproducibly across known generating scenarios.

## Full recovery workflow

When a compatible Bayesian backend is available, repeat each scenario over many seeds and fit the **same model you plan to use for the study**.

```python
truth = {"a_within": 0.45, "b_within": 0.55}

raw = simulate_multilevel_gaze_mediation(
    n_participants=100,
    trials_per_participant=20,
    a_within=truth["a_within"],
    b_within=truth["b_within"],
    cprime_within=0.20,
    seed=2026,
)

prepared = prepare_multilevel_mediation_data(
    raw,
    x_col="ai_correct",
    mediator_col="source_dwell",
    outcome_col="correct_override",
    quality_col="valid_fraction",
    minimum_quality=0.80,
    warn=False,
)

fit = fit_multilevel_gaze_mediation(
    prepared,
    mediator_family="gaussian",
    outcome_family="bernoulli",
    random_slopes=("mediator_x",),
    missingness_policy="error",
    chains=4,
    draws=1000,
    tune=1000,
    seed=2026,
)
```

Do not judge recovery from one favorable simulation. Repeat over many seeds and summarize performance across replications.

## What to evaluate

For each generating parameter and the within indirect effect, summarize at least:

- bias or median error relative to the known truth;
- interval coverage across replications;
- interval width / posterior uncertainty;
- convergence-failure frequency;
- divergence and tree-depth frequency;
- frequency with which the requested paths are not estimable from realized data;
- posterior predictive behavior;
- sensitivity to defensible prior scales and random-effects structures.

For a binary outcome, remember that the implemented indirect coefficient product is on the linear-predictor product scale. Recovery must be judged against the generating truth on that same scale.

## Strong, moderate, and weak truths

Recovery should be evaluated across more than one effect size. A workflow that recovers only a very strong indirect path can still be poorly calibrated for the smaller pathways expected in practice.

The focused test suite therefore includes a strong-vs-weak synthetic-signal check, while backend-enabled recovery should extend this to repeated fits and formal coverage summaries.

## Design sensitivity

Useful design scenarios include:

1. fewer/more participants;
2. fewer/more trials per participant;
3. balanced versus heterogeneous exposure proportions;
4. stronger mediator noise;
5. rare versus common binary outcomes;
6. missing mediator values or quality-ineligible trials generated under an explicit mechanism;
7. alternative supported random-slope structures;
8. alternative likelihoods when mediator distributional assumptions are uncertain.

Change one design dimension at a time when the goal is to understand its effect on recovery.

## Interpretation

Good recovery supports the statement:

> Under the simulated data-generating conditions and declared model specification, the analysis workflow can recover the target model-scale parameters with acceptable bias, interval coverage, and computational diagnostics.

It does **not** support the statement:

> The real study has a causal mediated mechanism.

Recovery validates the estimator/workflow under assumed conditions; it does not validate the assumptions themselves or establish construct validity for the gaze mediator.

## Limitations

Simulation results are only as relevant as the generating scenarios. Overly clean simulations can create false reassurance. Include realistic trial imbalance, participant heterogeneity, mediator noise, missingness/quality loss, and outcome prevalence whenever those features are expected in the real study.

Do not tune the simulator until it reproduces a preferred empirical result. Predeclare or justify the scenario grid independently of the observed indirect-effect estimate.

## Reporting example

A concise methods statement could read:

> Before substantive analysis, we evaluated the multilevel mediation workflow in repeated synthetic datasets generated with the planned participant/trial structure. Recovery scenarios varied the within exposure-to-mediator and mediator-to-outcome paths across weak, moderate, and strong values while preserving the intended likelihoods, random-effects structure, and missingness policy. We summarized parameter bias, 95% credible-interval coverage, convergence failures, and posterior predictive behavior across replications. Simulation recovery was used to assess the analysis pipeline under known conditions and was not treated as evidence for the real-data causal mechanism.

## Related API

- `simulate_multilevel_gaze_mediation()` — known-truth repeated-measures generator.
- `prepare_multilevel_mediation_data()` — trial-level preparation and decomposition in `eyeprocesspy`.
- `specify_multilevel_gaze_mediation()` — model contract without fitting.
- `fit_multilevel_gaze_mediation()` — backend fitting.
- `check_mediation_convergence()` — convergence gate.
- `estimate_within_indirect_effect()` — target within-person coefficient-product estimand.
- `posterior_predictive_check_mediation()` — predictive adequacy check.
- `report_multilevel_gaze_mediation()` — structured reporting helper.

For prior robustness after recovery, continue to [Worked example: prior sensitivity](prior-sensitivity-worked-example.md). For candidate-model predictive comparison, see [Model comparison for multilevel mediation](model-comparison.md).
