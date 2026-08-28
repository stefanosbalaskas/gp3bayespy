# First-Class Estimands and Sensitivity Workflows

> Python-facing port of `estimands-and-sensitivity.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

## Why estimands are first-class

gp3bayes distinguishes model coefficients from substantive quantities. Binary
workflows can report a design-standardised probability contrast. Duration
workflows can report conditional-median differences and ratios and a declared
posterior predictive upper quantile. None is automatically interpreted as a
causal effect.

## Binary probability standardisation

The fitting code below is not executed while building the article.

The target rows define the covariate distribution over which expected
probabilities are averaged. With `include_group_effects = FALSE`, predictions
are population-level rather than conditioned on observed group effects.

## Duration median and predictive-tail estimands

The exponentiated lognormal location contrast is treated as a conditional
median ratio, not an arithmetic-mean ratio. Predictive quantiles include
residual predictive variation.

## Structural sensitivity

No structure is selected automatically. The workflow asks whether the declared
estimand materially changes under the approved random-slope alternative.

## Participant and item deletion

Omission is a sensitivity analysis, not an exclusion rule. For designs with
many groups, units must be supplied explicitly rather than launching an
unbounded sequence of refits.

## Parameterisation sensitivity

Alternative codings and scales require explicit prior choices where prior
meaning changes. Unit conversion is handled separately because ratios should be
unit-free while absolute duration quantities must scale by the declared factor.

## Exact K-fold validation

Exact K-fold is deliberately an optional, expensive predictive-validation
adapter. It complements PSIS-LOO when refitting is scientifically appropriate;
it never becomes an automatic best-model selector.

## Python API mapping

- `gp3bayespy.compute_kfold_cv`
- `gp3bayespy.create_contrast_coding_sensitivity_specification`
- `gp3bayespy.create_duration_unit_sensitivity_specification`
- `gp3bayespy.create_group_deletion_sensitivity_plan`
- `gp3bayespy.create_predictor_scaling_sensitivity_specification`
- `gp3bayespy.create_random_slope_sensitivity_plan`
- `gp3bayespy.estimate_standardized_duration_estimands`
- `gp3bayespy.estimate_standardized_probability_contrast`
- `gp3bayespy.fit_binary_model_backend`
- `gp3bayespy.fit_duration_model_backend`
- `gp3bayespy.run_group_deletion_sensitivity`
- `gp3bayespy.run_random_slope_sensitivity`
- `gp3bayespy.summarise_estimand_draws`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/sensitivity_workflow.py`.
