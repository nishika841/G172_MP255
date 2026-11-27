
"""Unit tests for Stage 4 monitoring, retraining, and holdout artifacts."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from src.mp255.monitoring import compute_psi


def test_psi_computation_detects_drift():
    """Verify that compute_psi detects significant drift when distribution shifts."""

    np.random.seed(42)

    expected = np.random.normal(loc=10.0, scale=2.0, size=500)

    # Undrifted distribution
    same_dist = np.random.normal(loc=10.0, scale=2.0, size=500)

    psi_low = compute_psi(expected, same_dist)

    assert psi_low < 0.1, f"Expected low PSI for identical distributions, got {psi_low}"

    # Drastically shifted distribution
    drifted = np.random.normal(loc=30.0, scale=2.0, size=500)

    psi_high = compute_psi(expected, drifted)

    assert psi_high > 0.25, f"Expected high PSI for shifted distribution, got {psi_high}"
