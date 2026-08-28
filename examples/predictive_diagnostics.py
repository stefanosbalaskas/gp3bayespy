"""Backend-free binary predictive diagnostics and graphics."""

import numpy as np

import gp3bayespy as gp

probability = np.array([0.05, 0.20, 0.75, 0.90])
observed = np.array([0, 0, 1, 1])
roc = gp.binary_roc_curve(probability, observed)
pr = gp.binary_precision_recall_curve(probability, observed)
calibration = gp.binary_calibration_error(probability, observed, bins=2)
print(roc)
print(pr)
print(calibration)
