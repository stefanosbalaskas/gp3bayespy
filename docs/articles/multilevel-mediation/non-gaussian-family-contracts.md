# Worked example: non-Gaussian mediator contracts

The mediation API supports several mediator likelihoods because gaze/process variables do not all share the same support. This example is deliberately backend-independent: it validates the scientific **family contract** before sampling rather than pretending a sampler run is necessary to check support.

## Supported mediator families

| Family | Required observed support | Typical process variable |
| --- | --- | --- |
| `gaussian` | finite continuous values | approximately symmetric score/dwell measure |
| `lognormal` | strictly positive values | positive duration |
| `gamma` | strictly positive values | positive right-skewed duration |
| `beta` | strictly inside `(0, 1)` | bounded proportion without exact endpoints |
| `bernoulli` | exactly `0/1` | inspected vs not inspected |
| `poisson` | non-negative integers | event/fixation counts when equidispersion is defensible |
| `negative_binomial` | non-negative integers | overdispersed count |

The package does not silently transform values to make them compatible with a requested likelihood.

## Runnable contract example

The repository ships:

```bash
python examples/multilevel_gaze_mediation_family_contracts.py
```

It creates one synthetic repeated-measures dataset and derives six mediator representations with valid support. Each representation is prepared with `eyeprocesspy` and then passed to `specify_multilevel_gaze_mediation()` without sampling.

```python
spec = specify_multilevel_gaze_mediation(
    prepared,
    mediator_family="negative_binomial",
    outcome_family="bernoulli",
    random_slopes=("mediator_x",),
    missingness_policy="error",
)
```

The example verifies the same analysis-row count and exposure/outcome structure while changing only the mediator measurement scale and declared likelihood.

## Invalid shortcuts

Do not:

- add a tiny constant to zero-valued durations merely to make `lognormal` or `gamma` fit;
- round a continuous mediator to integers solely to access a count likelihood;
- squeeze exact 0/1 proportions into `(0, 1)` without a documented measurement rationale;
- recode missing gaze as Bernoulli zero;
- choose the family by whichever one produces the preferred indirect effect.

Those operations change the measurement model or estimand and should be treated as explicit analysis decisions, not preprocessing conveniences.

## Zero-heavy gaze variables

A duration with many genuine zeros is not supported by the current lognormal/Gamma mediator contract. Scientifically defensible alternatives can include a binary inspection mediator, a continuous conditional-on-positive duration analysis with an explicitly changed estimand, or a future hurdle/mixture model developed as a separate method. The current API intentionally refuses to imply that one of those choices is automatic.

## Interpretation with nonlinear mediator families

The mediator path coefficient lives on that mediator model's linear-predictor scale. For a Bernoulli mediator, for example, `a_W` is a log-odds-scale coefficient. The downstream within indirect product `a_W × b_W` remains a model-scale coefficient product and should be reported with its stored estimand scale rather than translated into an unsupported probability-scale natural indirect effect.

## Reporting checklist

Report:

1. the observed support of the mediator;
2. the selected family and why it matches the measurement process;
3. whether genuine zeros or exact endpoints were present;
4. any transformation performed before preparation;
5. the outcome family;
6. the indirect-effect scale;
7. prior predictive and posterior predictive evidence after fitting.

See [Choosing mediator and outcome families](family-selection.md), [Interpretation and estimands](interpretation-and-estimands.md), and [Assumptions and limitations](assumptions-and-limitations.md).
