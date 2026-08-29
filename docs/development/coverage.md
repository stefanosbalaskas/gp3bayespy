---
title: Coverage and visual QA
description: Branch-aware coverage and visual documentation QA for gp3bayespy.
---

# Coverage and visual QA

Coverage is reported separately for the immutable 0.5.0 release and the protected current development line.

| Metric | Value |
| --- | ---: |
| Released v0.5.0 branch-aware coverage | **47.9482%** |
| Visual-docs phase-1 checkpoint | **52.8043%** |
| Current branch-aware coverage | **68.0637%** |
| Current measured tests | **350 / 350 passing** |
| Frozen R exports | **458 / 458 implemented** |
| Canonical articles | **59 / 59 ported** |
| Deterministic documentation figures | **30** |
| Supplementary visual galleries | **6** |

## Coverage policy

The current CI suite measures all package modules with branch coverage and does not remove low-coverage modules from the denominator. The protected development line fails below **65% branch-aware coverage**.

Coverage is an execution metric, not a certificate of convergence, model adequacy, substantive validity, causal identification, or correctness of a scientific interpretation.

## Module census

| Module | Coverage | Statements | Branches |
| --- | ---: | ---: | ---: |
| `advanced_optional_workflows.py` | 35.3% | 416 | 134 |
| `postfit_exploration.py` | 51.8% | 356 | 154 |
| `reproducibility.py` | 57.6% | 290 | 120 |
| `predictive.py` | 62.1% | 1416 | 504 |
| `reporting.py` | 62.6% | 532 | 136 |
| `design_support_diagnostics.py` | 65.1% | 258 | 80 |
| `pupil.py` | 66.5% | 3185 | 1052 |
| `specification_closure.py` | 66.9% | 939 | 276 |
| `binary.py` | 68.0% | 754 | 216 |
| `duration.py` | 68.2% | 680 | 178 |
| `sensitivity.py` | 68.7% | 520 | 170 |
| `readiness.py` | 74.9% | 407 | 186 |
| `backends/__init__.py` | 75.0% | 268 | 56 |
| `loo.py` | 79.6% | 143 | 48 |
| `posterior.py` | 82.0% | 350 | 110 |
| `hierarchical_effects_advanced.py` | 84.4% | 197 | 60 |
| `unified_workflow_api.py` | 85.2% | 154 | 56 |
| `prior_posterior_bridge.py` | 86.8% | 202 | 64 |
| `fitting.py` | 87.4% | 123 | 28 |
| `posterior_validation_core.py` | 89.1% | 78 | 32 |
| `specification.py` | 89.6% | 272 | 130 |
| `evidence_graphics_gg.py` | 90.2% | 105 | 18 |
| `ppc.py` | 93.0% | 140 | 32 |
| `contracts.py` | 95.9% | 103 | 42 |
| `__init__.py` | 100.0% | 27 | 0 |
| `_reference/__init__.py` | 100.0% | 0 | 0 |
| `exceptions.py` | 100.0% | 2 | 0 |
| `parity.py` | 100.0% | 18 | 2 |

## Visual documentation QA

The visual-documentation tranche executes plotting adapters, regenerates deterministic package figures, supplies six gallery-style articles, and places figures into canonical articles where a visual materially improves interpretation.

The tranche also regression-tests the advanced-pupil derivative, dynamic-contrast, and posterior-trajectory adapters exposed during coverage expansion.
