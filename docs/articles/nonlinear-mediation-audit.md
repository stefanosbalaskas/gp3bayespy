# Experimental nonlinear mediation functional-form audit

This is a **frequentist diagnostic** alongside, not inside, the existing Bayesian multilevel mediation models. It addresses the September 2026 BriDGE methods briefing while avoiding causal-discovery and Bayesian-posterior claims.

```python
from gp3bayespy.nonlinear_mediation_audit import audit_mediation_functional_form

evidence = audit_mediation_functional_form(
    trial_data,
    treatment_col="treatment", mediator_col="pupil_response",
    outcome_col="trust", participant_col="participant_id",
    bootstrap_replicates=999, seed=2026,
)
print(evidence["estimates"])
```

**Models:** both include a treatment × mediator term; the nonlinear sensitivity also includes mediator squared and treatment × mediator squared. The mediation model is Gaussian with binary treatment and a continuous mediator and outcome. No GAM/spline causal-search algorithm, generalized outcome family, or longitudinal mediator trajectory is fitted.

**Contrast:** model-predicted outcome at treatment=1, comparing mediator levels generated under treatment=1 vs 0 using the shared fitted mediator residual. This is a *model-based indirect mediator shift*, not an identified natural indirect effect without sequential ignorability and suitable treatment assignment assumptions.

**Uncertainty:** participant-cluster bootstrap fully refits both models and records each failed replicate explicitly. Percentile limits are reported as sensitivity evidence, not Bayesian credible intervals. Prefer more resamples and robust variance design when publication is intended.

**Next qualification:** simulation under nonlinear mechanisms, measured mediator confounding, heteroscedastic errors, treatment-induced confounding, sparse trial counts, and external method-comparison reference code. Work with the existing [trial-level mediation guide](multilevel-mediation/index.md) rather than silently replacing it.
