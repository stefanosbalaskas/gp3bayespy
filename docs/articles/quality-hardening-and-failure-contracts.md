# Quality Hardening and Failure Contracts

> Python-facing port of `quality-hardening-and-failure-contracts.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

The development API is deliberately frozen during this hardening phase. The
purpose is to deepen integration and maintenance guarantees rather than add new
analytical surface area.

## Public API contract

A machine-readable manifest records all exported function names and formal
argument names. Tests compare the installed namespace against this manifest so
accidental public additions, removals, or signature changes become explicit
failures rather than silent drift.

The relevant governance interfaces include `validate_gp3bayes_object()`,
`capture_gp3bayes_schema()`, `validate_gp3bayes_schema()`,
`create_analysis_manifest()`, and `compare_analysis_manifests()`.

## Lightweight post-fit adapters

Small adapters are useful because they let downstream reports use stable data
frames rather than inspect internal object fields. Examples include
`backend_environment_table()`, `loo_influence_atlas_table()`,
`prediction_profile_table()`, `prediction_surface_table()`,
`prediction_draws_long()`, and `prior_posterior_draws_long()`.

## Explicit failure boundaries

Fit-dependent extraction helpers such as `extract_expected_predictions()`,
`extract_posterior_predictions()`, `extract_linear_predictions()`,
`extract_log_likelihood()`, and `extract_sampler_diagnostics()` reject malformed
inputs rather than guessing.

Prediction-comparison helpers retain explicit bounds. In particular,
`prediction_pairwise_contrasts()` and `prediction_rank_probabilities()` require
the analyst to opt into larger comparison sets rather than expanding
combinatorially without review.

## Prediction diagnostics

The diagnostic layer separates descriptive posterior evidence from automatic
decisions. `binary_group_calibration()`,
`posterior_predictive_summary_table()`, `predictive_coverage_table()`,
`duration_pit_table()`, and `loo_group_influence_table()` return evidence for
review; none automatically certifies adequacy or excludes observations/groups.

## Output safety

Writers such as `write_model_card()`, `write_publication_registry()`,
`write_diagnostic_dashboard_report()`, `write_analysis_bundle_report()`, and
`save_publication_registry_figures()` remain explicit-output operations. The
package does not use the current working directory as an implicit reporting
destination.

## Why examples are selective

Not every exported wrapper has a runnable Rd example. Many functions require a
fitted Bayesian backend object, and duplicating expensive fits across hundreds
of help topics would make checks slower without improving the underlying API.
The package therefore combines short deterministic Rd examples for lightweight
functions with articles, unit tests, integration tests, and complete reference
documentation for fit-dependent workflows.

## Python API mapping

- `gp3bayespy.backend_environment_table`
- `gp3bayespy.binary_group_calibration`
- `gp3bayespy.capture_gp3bayes_schema`
- `gp3bayespy.compare_analysis_manifests`
- `gp3bayespy.create_analysis_manifest`
- `gp3bayespy.duration_pit_table`
- `gp3bayespy.extract_expected_predictions`
- `gp3bayespy.extract_linear_predictions`
- `gp3bayespy.extract_log_likelihood`
- `gp3bayespy.extract_posterior_predictions`
- `gp3bayespy.extract_sampler_diagnostics`
- `gp3bayespy.loo_group_influence_table`
- `gp3bayespy.loo_influence_atlas_table`
- `gp3bayespy.posterior_predictive_summary_table`
- `gp3bayespy.prediction_draws_long`
- `gp3bayespy.prediction_pairwise_contrasts`
- `gp3bayespy.prediction_profile_table`
- `gp3bayespy.prediction_rank_probabilities`
- `gp3bayespy.prediction_surface_table`
- `gp3bayespy.predictive_coverage_table`
- `gp3bayespy.prior_posterior_draws_long`
- `gp3bayespy.save_publication_registry_figures`
- `gp3bayespy.validate_gp3bayes_object`
- `gp3bayespy.validate_gp3bayes_schema`
- `gp3bayespy.write_analysis_bundle_report`
- `gp3bayespy.write_diagnostic_dashboard_report`
- `gp3bayespy.write_model_card`
- `gp3bayespy.write_publication_registry`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```
