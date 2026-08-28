# Transformation Replay and Detailed Posterior Predictive Checks

> Python-facing port of `transformation-replay-and-detailed-ppc.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

## Replay recorded transformations

The recipe stores the already-approved mapping, condition coding, scaling
centres/scales, formula, and model-matrix columns. It does not learn a new
transformation from new data.

For prediction data, replay is explicit:

Unseen condition values, missing required transformed predictors, or a duration
unit inconsistent with the stored source unit produce errors rather than silent
recoding.

## Detailed binary PPC

The detailed binary object exposes calibration gaps, participant and item event
rates, participant-condition sparsity, and replicated all-zero/all-one
participant patterns. These are descriptive discrepancy checks, not a single
pass/fail goodness-of-fit test.

## Detailed duration PPC

Raw- and log-scale distributions are both retained because a lognormal model
can appear reasonable on one scale while still missing substantively important
tail or grouping structure. Persistent discrepancies request model-contract
review and never trigger an automatic likelihood switch.

## Python API mapping

- `gp3bayespy.apply_transformation_recipe`
- `gp3bayespy.check_binary_ppc_details`
- `gp3bayespy.check_duration_ppc_details`
- `gp3bayespy.create_model_contract`
- `gp3bayespy.create_transformation_recipe`
- `gp3bayespy.invert_transformation_recipe`
- `gp3bayespy.prepare_hierarchical_binary_data`
- `gp3bayespy.simulate_hierarchical_binary_data`
- `gp3bayespy.validate_transformation_replay`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/predictive_diagnostics.py`.
