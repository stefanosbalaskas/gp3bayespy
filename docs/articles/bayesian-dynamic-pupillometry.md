# Bayesian dynamic pupillometry: governed foundation

> Python-facing port of `bayesian-dynamic-pupillometry.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

## Scope

Development version 0.4.0.9000 adds a restricted Gaussian hierarchical
pupil-timecourse family. The family is designed for pupil series that have
already been produced by a documented preprocessing workflow. It does not
detect blinks, interpolate samples, correct pupil foreshortening error (PFE),
or infer psychological constructs from pupil change.

The workflow is deliberately staged:

1. declare a pupil contract;
2. prepare data only through explicit deterministic transformations;
3. inspect readiness and measurement-context audits;
4. create a closed model specification;
5. fit through an approved `rstan` or `cmdstanr` backend;
6. estimate declared pupil trajectories and window estimands;
7. inspect posterior predictive and temporal diagnostics;
8. validate against an explicit prediction target; and
9. compare prespecified sensitivity scenarios.

## Deterministic synthetic example

The simulator uses one convenient smooth response waveform for software
testing. It is not a claim that all biological pupil responses have this
shape.

Readiness output is evidence about the observed series. A warning is not an
automatic exclusion rule, and a pass is not evidence that the measurement or
scientific interpretation is valid.

## Restricted specification

The specification constructs the approved formula internally. There is no
user-facing arbitrary formula or arbitrary family argument.

## Interpretation boundary

Reported quantities are pupil diameter, pupil change, trajectories, and
posterior contrasts in those measurements. They are not automatically
cognitive load, attention, arousal, stress, emotion, surprise, or effort.
Those interpretations require a separate scientific argument and appropriate
experimental design.

## Python API mapping

- `gp3bayespy.audit_pupil_readiness`
- `gp3bayespy.create_pupil_contract`
- `gp3bayespy.prepare_pupil_timecourse`
- `gp3bayespy.pupil_readiness_table`
- `gp3bayespy.pupil_specification_table`
- `gp3bayespy.simulate_pupil_timecourse`
- `gp3bayespy.specify_pupil_timecourse_model`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/pupil_workflow.py`.
