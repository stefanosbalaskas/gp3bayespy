# Optional Bayesian Backend Installation

> Python-facing port of `optional-backend-installation.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

## Core installation

The package is independently useful without a Bayesian backend. Model
contracts, readiness audits, deterministic simulation, preparation,
specification, and prior predictive checks do not require `brms`, `rstan`,
`posterior`, `bayesplot`, or a compiler.

## Optional fitting and validation dependencies

Full-MCMC fitting uses `brms` with either `rstan` or `cmdstanr`. Posterior
diagnostics and visualisation use `posterior` and `bayesplot`.

For the `rstan` route:

For the `cmdstanr` route, install the common packages first:

Then install CmdStanR from the Stan R-universe repository:

The supported fitting interface remains restricted to `brms` and full MCMC.
`gp3bayespy` allows `rstan` or `cmdstanr` as implementation backends but does not
expose variational inference, Pathfinder, Laplace approximation, arbitrary Stan
programs, arbitrary model families, or arbitrary backend arguments.

## Windows toolchain check

On Windows, source compilation requires the Rtools version compatible with the
installed R version. After installing Rtools, start a clean R session and run:

The result should be `TRUE`.

For `cmdstanr`, additionally run:

## Backend preflight

For `rstan`:

For `cmdstanr`:

## Minimal compilation smoke test

Compilation should be tested with a deliberately small synthetic model before a
large analysis. Short chains may produce low effective-sample-size warnings;
those warnings must not be interpreted as adequate posterior inference.

A successful smoke fit confirms compilation and sampling execution only.
Production analyses require adequate iterations, sampling diagnostics,
posterior predictive checks, sensitivity assessment, and transparent
reporting.

## Clean-process package checks

After a Stan fit on Windows, run package checks and pkgdown builds in separate
clean R processes. This avoids accidental inheritance of model-compilation
flags from the interactive session.

```text
Rscript --vanilla -e "devtools::check()"
Rscript --vanilla -e "pkgdown::check_pkgdown(); pkgdown::build_site(preview = FALSE)"
```

## Python API mapping

- `gp3bayespy.bayesian_backend_capabilities`
- `gp3bayespy.check_cmdstan_backend`
- `gp3bayespy.create_model_contract`
- `gp3bayespy.fit_binary_model_backend`
- `gp3bayespy.prepare_hierarchical_binary_data`
- `gp3bayespy.simulate_hierarchical_binary_data`
- `gp3bayespy.specify_binary_model`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/backend_status.py`.
