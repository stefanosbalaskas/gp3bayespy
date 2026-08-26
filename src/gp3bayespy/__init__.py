"""gp3bayespy: contract-first Bayesian workflows in Python.

Frozen R reference: gp3bayes 0.5.0.
"""
from .backends import backend_capabilities
from .binary import (
    BinaryModelSpecification,
    BinaryPrepared,
    BinaryPriorPredictiveCheck,
    BinarySimulation,
    check_binary_prior_predictive,
    prepare_hierarchical_binary_data,
    simulate_hierarchical_binary_data,
    specify_binary_model,
)
from .contracts import ModelContract, create_model_contract
from .duration import (
    DurationModelSpecification,
    DurationPrepared,
    DurationPriorPredictiveCheck,
    DurationSimulation,
    check_duration_prior_predictive,
    prepare_hierarchical_duration_data,
    simulate_hierarchical_duration_data,
    specify_duration_model,
)
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
__r_reference_sha256__ = "537eb05f949de1bcc1d6f8234066f064597951ecfa9cbbdf938d0a895ce5dd8a"

__all__ = [
    "AuditCheck",
    "BackendUnavailableError",
    "BinaryModelSpecification",
    "BinaryPrepared",
    "BinaryPriorPredictiveCheck",
    "BinarySimulation",
    "DurationModelSpecification",
    "DurationPrepared",
    "DurationPriorPredictiveCheck",
    "DurationSimulation",
    "GP3BayesError",
    "ModelContract",
    "ModelSpecification",
    "PriorSpecification",
    "ReadinessAudit",
    "audit_model_readiness",
    "backend_capabilities",
    "build_model_formula",
    "check_binary_prior_predictive",
    "check_duration_prior_predictive",
    "create_model_contract",
    "create_model_specification",
    "create_prior_specification",
    "parity_counts",
    "prepare_hierarchical_binary_data",
    "prepare_hierarchical_duration_data",
    "read_parity_manifest",
    "reference_metadata",
    "simulate_hierarchical_binary_data",
    "simulate_hierarchical_duration_data",
    "specify_binary_model",
    "specify_duration_model",
    "validate_prior_specification",
]
