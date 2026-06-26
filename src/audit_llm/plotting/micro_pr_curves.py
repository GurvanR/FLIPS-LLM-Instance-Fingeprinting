"""Reusable plotting of micro-averaged Precision/Recall vs confidence threshold."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from audit_llm.plot_configs import (
    GRID_CONFIG,
    LEGEND_CONFIG,
    XLABEL_CONFIG,
    XTICKS_CONFIG,
    YLABEL_CONFIG,
)

LINE_BLUE = "tab:blue"

SATURATION_TARGETS: tuple[tuple[float, str, str, str], ...] = (
    (0.99, "lightsteelblue",
     "Recall@P.99={paired:.0%}",
     "Recall @ Precision=0.99 = {paired:.0%}"),
    (0.999, "cornflowerblue",
     "Recall@P.999={paired:.0%}",
     "Recall @ Precision=0.999 = {paired:.0%}"),
    (0.9999, "steelblue",
     "Recall@P.9999={paired:.0%}",
     "Recall @ Precision=0.9999 = {paired:.0%}"),
)


def _scaled(cfg: dict[str, Any], scale: float, *keys: str) -> dict[str, Any]:
    out = dict(cfg)
    for k in keys:
        if k in out and isinstance(out[k], (int, float)):
            out[k] = out[k] * scale
    return out


def plot_micro_pr_curve_on_ax(
    ax,
    thresholds,
    mean_prec,
    std_prec,
    mean_rec,
    std_rec,
    *,
    title: str | None = None,
    show_legend: bool = True,
    show_annotations: bool = True,
    show_xylabels: bool = True,
    xlabel: str = "Confidence threshold",
    ylabel: str = "(Micro-averaged) Precision / Recall",
    xylabel_pad: float | None = None,
    fontsize_scale: float = 1.0,
    legend_fontsize_scale: float | None = None,
    saturation_targets_filter: tuple[float, ...] | None = None,
    include_saturation_in_legend: bool = True,
    annot_y: float = 0.3,
) -> None:
    """Draw the standard micro P/R vs confidence curve on a given matplotlib Axes.

    Visual elements mirror the standalone PDF: dashed precision / solid recall in
    blue, ±1-std variance bands, and vertical saturation markers at
    P=0.99/0.999/0.9999 with rotated annotations.

    `fontsize_scale` shrinks label/tick/legend/annotation fonts for inset use.
    """
    thresholds = np.asarray(thresholds)
    mean_prec = np.asarray(mean_prec)
    std_prec = np.asarray(std_prec)
    mean_rec = np.asarray(mean_rec)
    std_rec = np.asarray(std_rec)

    ax.plot(thresholds, mean_prec, color=LINE_BLUE, linestyle="--",
            linewidth=2.0, label="Precision")
    ax.fill_between(thresholds,
                    np.clip(mean_prec - std_prec, 0.0, 1.0),
                    np.clip(mean_prec + std_prec, 0.0, 1.0),
                    color=LINE_BLUE, alpha=0.10)
    ax.plot(thresholds, mean_rec, color=LINE_BLUE, linestyle="-",
            linewidth=2.0, label="Recall")
    ax.fill_between(thresholds,
                    np.clip(mean_rec - std_rec, 0.0, 1.0),
                    np.clip(mean_rec + std_rec, 0.0, 1.0),
                    color=LINE_BLUE, alpha=0.10)

    annot_fontsize = 8 * fontsize_scale

    targets = (
        SATURATION_TARGETS
        if saturation_targets_filter is None
        else tuple(t for t in SATURATION_TARGETS if t[0] in saturation_targets_filter)
    )

    for target, color, annot_template, legend_template in targets:
        mask = mean_prec >= target
        if mask.any():
            idx = int(mask.argmax())
            paired_value = float(mean_rec[idx])
            cross_thr = float(thresholds[idx])
            vline_kwargs = {"color": color, "linestyle": "-", "linewidth": 1.0}
            if include_saturation_in_legend:
                vline_kwargs["label"] = legend_template.format(paired=paired_value)
            ax.axvline(cross_thr, **vline_kwargs)
            if show_annotations:
                ax.text(
                    cross_thr + 0.005, annot_y,
                    annot_template.format(paired=paired_value),
                    color=color, rotation=-90,
                    ha="left", va="center",
                    fontsize=annot_fontsize, fontweight="bold",
                )

    if show_xylabels:
        xlabel_kwargs = _scaled(XLABEL_CONFIG, fontsize_scale, "fontsize")
        ylabel_kwargs = _scaled(YLABEL_CONFIG, fontsize_scale, "fontsize")
        if xylabel_pad is not None:
            xlabel_kwargs["labelpad"] = xylabel_pad
            ylabel_kwargs["labelpad"] = xylabel_pad
        ax.set_xlabel(xlabel, **xlabel_kwargs)
        ax.set_ylabel(ylabel, **ylabel_kwargs)
    ax.tick_params(
        axis="both", which="major",
        **_scaled(XTICKS_CONFIG, fontsize_scale, "labelsize"),
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(**GRID_CONFIG)
    if title is not None:
        ax.set_title(title, fontsize=11 * fontsize_scale)
    if show_legend:
        legend_scale = legend_fontsize_scale if legend_fontsize_scale is not None else fontsize_scale
        ax.legend(**_scaled(LEGEND_CONFIG, legend_scale, "fontsize"))


def save_micro_pr_curve_figure(pr_curve_data_bs: dict, save_path, n_splits: int) -> Path:
    """Render and save one standalone micro P/R-vs-confidence PDF for a single batch size.

    ``pr_curve_data_bs`` is one entry of the ``pr_curve_data`` dict returned by
    ``compute_micro_pr_curve`` (keys: ``thresholds``, ``mean_prec``, ``std_prec``,
    ``mean_rec``, ``std_rec``, ``tp_group_name``, ``bs``). Kept here so the live run
    and the cached preview script draw the exact same figure.
    """
    import matplotlib
    matplotlib.use("Agg")  # safe for headless HPC runs
    import matplotlib.pyplot as plt

    tp_group_name = pr_curve_data_bs["tp_group_name"]
    bs = pr_curve_data_bs["bs"]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    plot_micro_pr_curve_on_ax(
        ax,
        pr_curve_data_bs["thresholds"],
        pr_curve_data_bs["mean_prec"],
        pr_curve_data_bs["std_prec"],
        pr_curve_data_bs["mean_rec"],
        pr_curve_data_bs["std_rec"],
        legend_fontsize_scale=1.3,
    )
    plt.tight_layout()

    pdf_out = Path(save_path) / f"{tp_group_name}_{n_splits}_splits_micro_pr_curve_bs_{bs}.pdf"
    fig.savefig(pdf_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_out
