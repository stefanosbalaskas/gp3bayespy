"""Direct PSIS-LOO from pointwise log-likelihood draws."""

import numpy as np

import gp3bayespy as gp

rng = np.random.default_rng(21)
log_lik_a = rng.normal(-1.1, 0.25, size=(400, 60))
log_lik_b = rng.normal(-1.2, 0.30, size=(400, 60))
loo_a = gp.compute_psis_loo_from_log_lik(log_lik_a)
loo_b = gp.compute_psis_loo_from_log_lik(log_lik_b)
print(gp.loo_pointwise_table(loo_a).head())
print(gp.compare_psis_loo({"model_a": loo_a, "model_b": loo_b}))
