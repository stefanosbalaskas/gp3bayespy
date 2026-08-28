# A Reproducible 0.2.0 Release Case Study

> Python-facing port of `release-case-study.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

## Purpose

This case study exercises the stable 0.2.0 workflow on deterministic synthetic
data. The vignette evaluates every backend-independent stage and leaves the
optional Stan fits unevaluated so package documentation remains portable.

## 1. Simulate known data

## 2. Declare the analysis contract

## 3. Preflight the design

## 4. Prepare and specify

## 5. Freeze analysis provenance

## 6. Optional dual-backend fitting

## 7. Unified posterior review

## 8. Cross-backend consistency

## 9. Evidence and compatibility

The end product is an inspectable chain from design contract to evidence
inventory. At no stage does the package infer emotion, cognition, diagnosis,
causality, model adequacy, robustness, or a preferred model automatically.

## Python API mapping

- `gp3bayespy.audit_backend_parity`
- `gp3bayespy.audit_design_support`
- `gp3bayespy.audit_model_readiness`
- `gp3bayespy.capture_gp3bayes_schema`
- `gp3bayespy.check_binary_prior_predictive`
- `gp3bayespy.check_model_ppc`
- `gp3bayespy.collect_model_evidence`
- `gp3bayespy.compute_psis_loo`
- `gp3bayespy.create_analysis_manifest`
- `gp3bayespy.create_model_contract`
- `gp3bayespy.create_sensitivity_suite_plan`
- `gp3bayespy.diagnose_model_fit`
- `gp3bayespy.estimate_model_estimands`
- `gp3bayespy.fit_binary_model_backend`
- `gp3bayespy.freeze_analysis_manifest`
- `gp3bayespy.freeze_gp3bayes_schema`
- `gp3bayespy.model_workflow_status`
- `gp3bayespy.prepare_hierarchical_binary_data`
- `gp3bayespy.run_sensitivity_suite`
- `gp3bayespy.simulate_hierarchical_binary_data`
- `gp3bayespy.specify_binary_model`
- `gp3bayespy.summarise_model_posterior`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/reproducibility_workflow.py`.
