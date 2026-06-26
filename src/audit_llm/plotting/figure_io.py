"""Figure I/O and numerical utilities for plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_fig_and_show(save_path: str | Path, show: bool = False, fig_name: str = "") -> None:
    """Save the current matplotlib figure and optionally show it."""
    if save_path:
        out_path = Path(save_path) / fig_name
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
    if show:
        plt.show()
    plt.close()


def clip_std(
    mean: np.ndarray | float,
    std: np.ndarray | float,
    lower: float = 0.0,
    upper: float = 1.0,
) -> np.ndarray | float:
    """Clip std so error bars don't exceed [lower, upper]."""
    max_below = mean - lower
    max_above = upper - mean
    return np.minimum(std, np.minimum(max_below, max_above))
