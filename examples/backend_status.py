"""Inspect available Bayesian backends without compiling or fitting."""

import gp3bayespy as gp

print(gp.backend_capabilities())
print(gp.validate_backend_environment())
