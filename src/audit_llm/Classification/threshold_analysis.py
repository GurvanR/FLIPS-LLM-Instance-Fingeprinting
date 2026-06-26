"""
Threshold and error bar calculation utilities.

Provides functions for calculating constrained error bars that respect non-negative
constraints, supporting various methods like clipping, SEM, and confidence intervals.
"""

import numpy as np


def calculate_constrained_error_bars(means, stds, method="clip", n_samples=None):
    """
    Calculate error bars that respect non-negative constraints.

    Args:
        means: Array of mean values
        stds: Array of standard deviations
        method: Method for handling negative lower bounds
        n_samples: Number of samples (required for SEM and confidence intervals)

    Returns:
        tuple: (lower_errors, upper_errors) for asymmetric error bars
    """
    if method == "clip":
        lower_errors = np.minimum(stds, means)
        upper_errors = stds

    elif method == "sem" and n_samples is not None:
        sem = stds / np.sqrt(n_samples)
        lower_errors = np.minimum(sem, means)
        upper_errors = sem

    elif method == "confidence" and n_samples is not None:
        from scipy import stats

        sem = stds / np.sqrt(n_samples)
        t_val = stats.t.ppf(0.975, n_samples - 1) if n_samples > 1 else 1.96
        ci = t_val * sem
        lower_errors = np.minimum(ci, means)
        upper_errors = ci

    else:
        # Default to clipping
        lower_errors = np.minimum(stds, means)
        upper_errors = stds

    return lower_errors, upper_errors
