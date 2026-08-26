"""gp3bayespy: contract-first Bayesian workflows in Python.

Frozen R reference: gp3bayes 0.5.0.
"""

from .backends import backend_capabilities
from .contracts import ModelContract, create_model_contract
from .exceptions import BackendUnavailableError, GP3BayesError
from .parity import parity_counts, read_parity_manifest, reference_metadata
from .readiness import AuditCheck, ReadinessAudit, audit_model_readiness
from .specification import (
    ModelSpecification,
    PriorSpecification,
    build_model_formula,
    create_model_specification,
    create_prior_specification,
    validate_prior_specification,
)

__version__ = "0.1.0.dev0"
__r_reference_version__ = "0.5.0"
__r_reference_sha256__ = (
    "537eb05f949de1bcc1d6f8234066f064597951ecfa9cbbdf938d0a895ce5dd8a"
)

__all__ = [
    "AuditCheck",
    "BackendUnavailableError",
    "GP3BayesError",
    "ModelContract",
    "ModelSpecification",
    "PriorSpecification",
    "ReadinessAudit",
    "audit_model_readiness",
    "backend_capabilities",
    "build_model_formula",
    "create_model_contract",
    "create_model_specification",
    "create_prior_specification",
    "parity_counts",
    "read_parity_manifest",
    "reference_metadata",
    "validate_prior_specification",
]
