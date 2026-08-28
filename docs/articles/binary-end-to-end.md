# End-to-End Hierarchical Binary Workflow

> Python-facing port of `binary-end-to-end.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

## Scope

This article presents the approved Bernoulli-logit workflow for binary
trial-level outcomes with repeated observations, participant effects, and
optional crossed item effects. The interface is deliberately restricted:
users do not supply an arbitrary formula, likelihood, backend, Stan program,
or automatic model-selection rule.

The core contract, simulation, preparation, model specification, and prior
predictive check do not require a Bayesian backend.

## Simulate a known data-generating process

The simulation object retains fixed effects, grouping scales, the condition
coding, random effects, and the random-number seed. Stored truth enables
later recovery assessment without a private or external data set.

## Declare the model contract

Contract creation records the intended outcome, grouping structure,
likelihood, link, supported interaction, assumptions, diagnostics, limitations,
and interpretation boundaries. It does not validate the data or establish
model adequacy.

## Prepare and audit the data

Outcome mapping, condition coding, missing-value decisions, and requested
scaling are explicit and recorded. No variable is silently transformed.

## Specify priors and inspect prior implications

A prior-predictive failure requests substantive review. The function does not
automatically change priors or select a different model.

## Translate and fit with the optional backend

The fitting route is fixed to `brms`, `rstan`, and full MCMC sampling. The
following code requires the optional backend and a working C++ toolchain.

A returned fit confirms that sampling completed. It does not by itself
establish convergence or posterior adequacy.

## Diagnose, interpret, and validate

The diagnostic object reports R-hat, bulk and tail ESS, divergent transitions,
maximum-treedepth saturation, and chain-level energy diagnostics. Its
pass/review/fail status is a threshold report, not an automatic declaration
that the model converged.

Population-level coefficients are reported on the log-odds and odds-ratio
scales. Posterior probabilities and intervals are not frequentist significance
tests and are not automatically causal.

## Prior sensitivity and simulation recovery

A small recovery run is only a smoke test. A larger run assesses the declared
synthetic design and does not validate every future use of the package.

## Structured reporting

The report keeps fitting, diagnostics, predictive checks, sensitivity, and
recovery as separate evidence layers. It does not automatically claim
convergence, predictive validity, causal identification, or substantive
validity.

## Python API mapping

- `gp3bayespy.assess_binary_prior_sensitivity`
- `gp3bayespy.check_binary_posterior_predictive`
- `gp3bayespy.check_binary_prior_predictive`
- `gp3bayespy.create_binary_model_report`
- `gp3bayespy.create_model_contract`
- `gp3bayespy.diagnose_binary_fit`
- `gp3bayespy.fit_binary_model`
- `gp3bayespy.prepare_hierarchical_binary_data`
- `gp3bayespy.run_binary_recovery`
- `gp3bayespy.simulate_hierarchical_binary_data`
- `gp3bayespy.specify_binary_model`
- `gp3bayespy.summarise_binary_posterior`
- `gp3bayespy.translate_binary_model_to_brms`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/binary_workflow.py`.
