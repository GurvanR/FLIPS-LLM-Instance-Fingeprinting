"""Threshold analysis plots — alpha trade-off, threshold distributions, extraction."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np

from audit_llm.plot_configs import (
    GRID_CONFIG,
    LEGEND_CONFIG,
    XLABEL_CONFIG,
    XTICKS_CONFIG,
    YLABEL_CONFIG,
)


def plot_alpha_tradeoff(
    alpha_results: dict,
    fig_save_path: str | Path,
    batch_prediction_sizes: list,
) -> None:
    """Plot accuracy vs alpha-quantile threshold, one plot per batch size."""
    alphas = sorted(alpha_results.keys())

    # Create one plot per batch size
    for bs_idx, bs in enumerate(batch_prediction_sizes):
        # Extract metrics for each alpha for this specific batch size
        global_acc_means = []
        global_acc_stds = []
        unseen_acc_means = []
        unseen_acc_stds = []
        known_acc_means = []
        known_acc_stds = []

        for alpha in alphas:
            # Access the specific batch size index
            global_acc_means.append(alpha_results[alpha]["global_accuracy_mean"][bs_idx])
            global_acc_stds.append(alpha_results[alpha]["global_accuracy_std"][bs_idx])
            unseen_acc_means.append(alpha_results[alpha]["unseen_accuracy_mean"][bs_idx])
            unseen_acc_stds.append(alpha_results[alpha]["unseen_accuracy_std"][bs_idx])
            known_acc_means.append(alpha_results[alpha]["known_accuracy_mean"][bs_idx])
            known_acc_stds.append(alpha_results[alpha]["known_accuracy_std"][bs_idx])

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot three curves with error bars
        ax.errorbar(alphas, global_acc_means, yerr=global_acc_stds, marker="o", label="Global Accuracy", capsize=5)
        ax.errorbar(
            alphas, unseen_acc_means, yerr=unseen_acc_stds, marker="o", label="Unseen Models Accuracy", capsize=5
        )
        ax.errorbar(alphas, known_acc_means, yerr=known_acc_stds, marker="o", label="Known Models Accuracy", capsize=5)

        # Applying the style configurations
        ax.set_xlabel(r"$\alpha$-Quantile Threshold", **XLABEL_CONFIG)
        ax.set_ylabel("Accuracy", **YLABEL_CONFIG)

        # Applying tick parameters to both axes
        ax.tick_params(axis="both", which="major", **XTICKS_CONFIG)
        ax.legend(**LEGEND_CONFIG)
        ax.grid(**GRID_CONFIG)

        # Layout and Saving
        plt.tight_layout()
        plot_path = Path(fig_save_path) / f"alpha_tradeoff_accuracy_bs_{bs}.pdf"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Alpha trade-off plot for batch size %s saved to %s", bs, plot_path)


def plot_thresholds_distribution(
    thresholds: Dict[str, List[Dict]],
    fig_path: Optional[Path] = None,
    bins: int = 20,
    show_individual_ds: bool = False,
) -> None:
    """Plot mean threshold histogram (with +/-1 std band) across datasets.

    Parameters
    ----------
    thresholds : dict
        ``{dataset: [{clf: {bs: threshold_float}}, ...]}``.
    fig_path : Path or None
        Directory for saving figures.
    bins : int
        Number of histogram bins.
    show_individual_ds : bool
        If True, overlay individual dataset histograms faintly.
    """
    if not thresholds:
        raise ValueError("thresholds dict is empty")

    datasets = list(thresholds.keys())
    if not datasets:
        raise ValueError("no datasets found in thresholds")

    # Union grid across all (tp, run): supports mix_tp_at_pred where each
    # uplet (= dataset key) only carries data at its single bs.
    clfs_set: set = set()
    bs_set: set = set()
    for tp in datasets:
        for run in thresholds[tp]:
            for clf, bs_dict in run.items():
                clfs_set.add(clf)
                bs_set.update(bs_dict.keys())
    if not clfs_set:
        raise ValueError("No classifiers found in any threshold run")
    if not bs_set:
        raise ValueError("No batch sizes found in any threshold run")
    clfs = sorted(clfs_set)
    batch_sizes = sorted(bs_set)

    for clf in clfs:
        for bs in batch_sizes:
            # Per (clf, bs), only keep token-pairs whose runs have this key.
            filtered_tp = []
            per_tp_values = []
            for tp in datasets:
                vals = [
                    float(run[clf][bs])
                    for run in thresholds[tp]
                    if clf in run and bs in run[clf]
                ]
                if vals:
                    filtered_tp.append(tp)
                    per_tp_values.append(np.array(vals))

            if not per_tp_values:
                logger.debug(
                    "plot_thresholds_distribution: no data for clf=%s bs=%s, skipping figure.",
                    clf, bs,
                )
                continue

            # Compute global min/max for binning
            all_mins = [vals.min() for vals in per_tp_values]
            all_maxs = [vals.max() for vals in per_tp_values]
            vmin, vmax = min(all_mins), max(all_maxs)
            if vmin == vmax:
                pad = 1e-6 if vmin == 0 else abs(vmin) * 1e-3
                vmin -= pad
                vmax += pad

            bin_edges = np.linspace(vmin, vmax, bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            hist_matrix = np.zeros((len(filtered_tp), bins), dtype=float)

            # Fill histogram matrix
            for i, vals in enumerate(per_tp_values):
                hist, _ = np.histogram(vals, bins=bin_edges, density=True)
                hist_matrix[i, :] = hist

            mean_hist = hist_matrix.mean(axis=0)
            std_hist = hist_matrix.std(axis=0)

            # Plotting
            if fig_path is not None:
                fig_bs_path = Path(fig_path) / "Thresholds" / clf
                fig_bs_path.mkdir(parents=True, exist_ok=True)

                plt.figure(figsize=(8, 6))

                if show_individual_ds:
                    for i in range(len(filtered_tp)):
                        plt.plot(bin_centers, hist_matrix[i, :], alpha=0.25, linewidth=1)

                width = bin_edges[1] - bin_edges[0]
                plt.bar(bin_centers, mean_hist, width=width, alpha=0.6, align="center", label="Mean histogram")

                lower = np.clip(mean_hist - std_hist, 0, None)
                upper = mean_hist + std_hist

                plt.xlabel("Thresholds", **XLABEL_CONFIG)
                plt.ylabel("Density", **YLABEL_CONFIG)
                plt.tick_params(axis="both", which="major", **XTICKS_CONFIG)
                plt.legend(**LEGEND_CONFIG)
                plt.grid(**GRID_CONFIG)
                plt.tight_layout()

                fname = fig_bs_path / f"Threshold_{clf}_bs{bs}_mean_hist.pdf"
                plt.savefig(fname)
                plt.close()


def plot_openset_roc_curves(
    roc_data: Dict[int, Dict[str, Dict[str, np.ndarray]]],
    fig_save_path: str | Path,
    batch_prediction_sizes: List[int],
    show_std_band: bool = True,
) -> None:
    """Plot ROC curves for unseen-vs-known binary detection, one figure per batch size.

    Parameters
    ----------
    roc_data : dict
        ``{bs: {clf_name: {"mean_fpr": array, "mean_tpr": array,
        "std_tpr": array, "mean_auroc": float, "std_auroc": float}}}``.
    fig_save_path : str or Path
        Directory for saving figures.
    batch_prediction_sizes : list[int]
        Batch sizes to iterate over.
    show_std_band : bool
        If True, draw +/- 1 std shaded band around mean TPR.
    """
    from audit_llm.plotting.constants import COLORBLIND_COLORS

    for bs in batch_prediction_sizes:
        if bs not in roc_data or not roc_data[bs]:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        for color_idx, (clf_name, clf_roc) in enumerate(roc_data[bs].items()):
            color = COLORBLIND_COLORS[color_idx % len(COLORBLIND_COLORS)]
            mean_fpr = clf_roc["mean_fpr"]
            mean_tpr = clf_roc["mean_tpr"]
            std_tpr = clf_roc["std_tpr"]
            mean_auroc = clf_roc["mean_auroc"]
            std_auroc = clf_roc["std_auroc"]

            label = f"{clf_name} (AUROC = {mean_auroc:.3f} ± {std_auroc:.3f})"
            ax.plot(mean_fpr, mean_tpr, color=color, label=label, linewidth=2)

            if show_std_band:
                tpr_upper = np.minimum(mean_tpr + std_tpr, 1.0)
                tpr_lower = np.maximum(mean_tpr - std_tpr, 0.0)
                ax.fill_between(mean_fpr, tpr_lower, tpr_upper, color=color, alpha=0.15)

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random")

        ax.set_xlabel("False Positive Rate", **XLABEL_CONFIG)
        ax.set_ylabel("True Positive Rate", **YLABEL_CONFIG)
        ax.tick_params(axis="both", which="major", **XTICKS_CONFIG)
        ax.legend(**LEGEND_CONFIG)
        ax.grid(**GRID_CONFIG)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        plt.tight_layout()
        plot_path = Path(fig_save_path) / f"roc_curve_unseen_vs_known_bs_{bs}.pdf"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("ROC curve for batch size %s saved to %s", bs, plot_path)


def save_openset_metrics_table(
    roc_data: Dict[int, Dict[str, Dict]],
    fig_save_path: str | Path,
    batch_prediction_sizes: List[int],
) -> None:
    """Save Precision/Recall/F1/AUROC per classifier as a Markdown table, one file per batch size.

    Parameters
    ----------
    roc_data : dict
        ``{bs: {clf_name: {"mean_precision": float, "std_precision": float,
        "mean_recall": float, "std_recall": float, "mean_f1": float,
        "std_f1": float, "mean_auroc": float, "std_auroc": float, ...}}}``.
    fig_save_path : str or Path
        Directory where the Markdown files are written.
    batch_prediction_sizes : list[int]
        Batch sizes to iterate over.
    """
    for bs in batch_prediction_sizes:
        if bs not in roc_data or not roc_data[bs]:
            continue
        lines = [
            f"## Open-set Detection Metrics — Batch size {bs}",
            "Threshold: alpha-quantile | Class: Unknown (pos_label=0)",
            "",
            "| Classifier | Precision (Unseen) | Recall (Unseen) | F1 (Unseen) | AUROC |",
            "|:---:|:---:|:---:|:---:|:---:|",
        ]
        for clf_name, d in roc_data[bs].items():
            lines.append(
                f"| {clf_name} "
                f"| {d['mean_precision']:.3f} \u00b1 {d['std_precision']:.3f} "
                f"| {d['mean_recall']:.3f} \u00b1 {d['std_recall']:.3f} "
                f"| {d['mean_f1']:.3f} \u00b1 {d['std_f1']:.3f} "
                f"| {d['mean_auroc']:.3f} \u00b1 {d['std_auroc']:.3f} |"
            )
        table_path = Path(fig_save_path) / f"openset_metrics_bs_{bs}.md"
        table_path.write_text("\n".join(lines) + "\n")
        logger.info("Metrics table saved to %s", table_path)

        # JSON sidecar so downstream consumers (e.g. NxM heatmap subtitle) can
        # pick up precision/recall without re-running the openset pipeline.
        json_payload = {
            clf_name: {
                "mean_precision": d["mean_precision"],
                "std_precision": d["std_precision"],
                "mean_recall": d["mean_recall"],
                "std_recall": d["std_recall"],
                "mean_f1": d["mean_f1"],
                "std_f1": d["std_f1"],
                "mean_auroc": d["mean_auroc"],
                "std_auroc": d["std_auroc"],
            }
            for clf_name, d in roc_data[bs].items()
        }
        json_path = Path(fig_save_path) / f"openset_metrics_bs_{bs}.json"
        json_path.write_text(json.dumps(json_payload, indent=2))


def extract_thresholds(
    MaxProbas: Dict,
    alpha: float = 0.05,
    prioritize: str = "Known",
    fig_path: Optional[Path] = None,
) -> dict:
    """Extract rejection thresholds from known/unknown probability distributions.

    Parameters
    ----------
    MaxProbas : dict
        ``{'Known'/'Unknown': [dict[clf][bs]['wrong'/'correct'] -> 1d array]}``.
    alpha : float
        Quantile threshold.
    prioritize : str
        ``'Known'`` or ``'Unknown'``.
    fig_path : Path or None
        Unused (reserved for future plotting).

    Returns
    -------
    dict
        ``{clf: {bs: threshold_float}}``.
    """
    # Get classifiers
    clfs = list(MaxProbas["Known"][0].keys())
    # Get batch sizes
    batch_sizes = list(MaxProbas["Known"][0][clfs[0]].keys())

    thresholds = defaultdict(dict)

    for clf in clfs:
        for bs in batch_sizes:
            # Only correct Known predictions matter
            Known_vals = np.concatenate([d[clf][bs]["correct"] for d in MaxProbas["Known"]])
            # All Unknown predictions are "wrong" effectively
            Unknown_wrong = np.concatenate([d[clf][bs]["wrong"] for d in MaxProbas["Unknown"]])
            Unknown_correct = np.concatenate([d[clf][bs]["correct"] for d in MaxProbas["Unknown"]])
            Unknown_vals = np.concatenate([Unknown_wrong, Unknown_correct])

            MinKnown = np.min(Known_vals)
            MaxUnknown = np.max(Unknown_vals)

            if MinKnown < MaxUnknown:
                # overlapping distributions
                if prioritize == "Known":
                    thresholds[clf][bs] = np.percentile(Known_vals, 100 * alpha)
                elif prioritize == "Unknown":
                    thresholds[clf][bs] = np.percentile(Unknown_vals, 100 * (1 - alpha))
            else:
                # non-overlapping distributions
                if prioritize == "Known":
                    thresholds[clf][bs] = MaxUnknown
                elif prioritize == "Unknown":
                    thresholds[clf][bs] = MinKnown

    return thresholds


def _compute_tpr_fpr_single_iter(
    kc: np.ndarray,
    kw: np.ndarray,
    u: np.ndarray,
    thresholds: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Vectorised TPR / FPR (pos_label=Known=1) for one outer iteration.

    TPR = fraction of Known samples (kc ∪ kw) predicted as Known (score ≥ t).
    FPR = fraction of Unknown samples (u) falsely predicted as Known (score ≥ t).
    Convention matches the existing ROC code (``pos_label=1`` = Known).
    """
    t = thresholds[None, :]

    kc_ge = (kc[:, None] >= t).sum(axis=0).astype(float) if kc.size else np.zeros_like(thresholds)
    kw_ge = (kw[:, None] >= t).sum(axis=0).astype(float) if kw.size else np.zeros_like(thresholds)
    u_ge = (u[:, None] >= t).sum(axis=0).astype(float) if u.size else np.zeros_like(thresholds)

    n_known = float(kc.size + kw.size)
    n_unknown = float(u.size)

    tpr = (kc_ge + kw_ge) / max(n_known, 1.0)
    fpr = u_ge / max(n_unknown, 1.0)
    return {"tpr": tpr, "fpr": fpr}


def _compute_pr_curves_single_iter(
    kc: np.ndarray,
    kw: np.ndarray,
    u: np.ndarray,
    thresholds: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Vectorised per-threshold P/R for one outer iteration.

    Returns dict with keys ``unseen_p``, ``unseen_r``, ``global_p``, ``global_r``,
    each a 1-D array indexed by ``thresholds``.

    Conventions (see plan): Unseen is the binary unseen-vs-known detection
    with ``pos_label=Unknown``; Global is the closed-set abstention rule of
    ``compute_micro_pr_curve`` restricted to Known samples.
    """
    t = thresholds[None, :]  # (1, T)

    kc_ge = (kc[:, None] >= t).sum(axis=0).astype(float) if kc.size else np.zeros_like(thresholds)
    kw_ge = (kw[:, None] >= t).sum(axis=0).astype(float) if kw.size else np.zeros_like(thresholds)
    u_ge = (u[:, None] >= t).sum(axis=0).astype(float) if u.size else np.zeros_like(thresholds)

    n_kc, n_kw, n_u = float(kc.size), float(kw.size), float(u.size)
    kc_lt = n_kc - kc_ge
    kw_lt = n_kw - kw_ge
    u_lt = n_u - u_ge

    # Unseen (pos_label = Unknown)
    TP_u = u_lt
    FP_u = kc_lt + kw_lt
    FN_u = u_ge
    denom_pu = TP_u + FP_u
    denom_ru = TP_u + FN_u
    unseen_p = np.where(denom_pu > 0, TP_u / np.maximum(denom_pu, 1.0), 1.0)
    unseen_r = np.where(denom_ru > 0, TP_u / np.maximum(denom_ru, 1.0), 0.0)

    # Global (Known-only, closed-set abstention)
    TP = kc_ge
    FP = kw_ge
    FN = FP + kc_lt + kw_lt
    denom_pg = TP + FP
    denom_rg = TP + FN
    global_p = np.where(denom_pg > 0, TP / np.maximum(denom_pg, 1.0), 1.0)
    global_r = np.where(denom_rg > 0, TP / np.maximum(denom_rg, 1.0), 0.0)

    return {
        "unseen_p": unseen_p,
        "unseen_r": unseen_r,
        "global_p": global_p,
        "global_r": global_r,
    }


def plot_unseen_and_global_pr_vs_confidence(
    pr_data: Dict[int, Dict[str, List[Dict[str, np.ndarray]]]],
    fig_save_path: str | Path,
    batch_prediction_sizes: List[int],
    n_thresholds: int = 101,
    unseen_prevalence: float | None = None,
) -> None:
    """Plot Precision/Recall vs raw confidence for Unseen detection and global Known-class
    classification, on a single axes per batch size.

    Only ``bs=1`` and ``bs=max(batch_prediction_sizes)`` are produced.

    Parameters
    ----------
    pr_data : dict
        ``{bs: {clf_name: [iteration_dict, ...]}}`` where each iteration dict has
        keys ``"kc"`` (Known-correct max scores), ``"kw"`` (Known-wrong),
        ``"u"`` (Unknown samples, all pooled).
    fig_save_path : str or Path
        Directory for saving figures.
    batch_prediction_sizes : list[int]
        All batch sizes the run uses (only ``min`` and ``max`` of this list are plotted).
    n_thresholds : int
        Resolution of the threshold sweep.
    unseen_prevalence : float or None
        If provided, overrides the empirical sample-level prevalence on the
        horizontal reference line. Use this to display the configured
        ``m_test_size`` (design prevalence) instead of the realised one, which
        can drift slightly due to integer flooring of unknown-class count and
        per-class sample truncation.
    """
    if not batch_prediction_sizes:
        return
    bs_to_plot = sorted({1, max(batch_prediction_sizes)})
    thresholds = np.linspace(0.0, 1.0, n_thresholds)

    for bs in bs_to_plot:
        if bs not in pr_data or not pr_data[bs]:
            continue

        fig, ax = plt.subplots(figsize=(8.25, 5.25))

        for clf_name, iterations in pr_data[bs].items():
            if not iterations:
                continue

            per_iter = [
                _compute_pr_curves_single_iter(
                    np.asarray(it["kc"], dtype=float),
                    np.asarray(it["kw"], dtype=float),
                    np.asarray(it["u"], dtype=float),
                    thresholds,
                )
                for it in iterations
            ]
            curves = {
                k: np.stack([d[k] for d in per_iter], axis=0)
                for k in ("unseen_p", "unseen_r", "global_p", "global_r")
            }

            for key, color, linestyle, label in (
                ("global_p", "tab:blue", "--", "Global Precision"),
                ("global_r", "tab:blue", "-", "Global Recall"),
                ("unseen_p", "tab:orange", "--", "Unseen Precision"),
                ("unseen_r", "tab:orange", "-", "Unseen Recall"),
            ):
                mean = curves[key].mean(axis=0)
                std = curves[key].std(axis=0)
                ax.plot(
                    thresholds, mean,
                    color=color, linestyle=linestyle, linewidth=2.0,
                    label=label,
                )
                ax.fill_between(
                    thresholds,
                    np.clip(mean - std, 0.0, 1.0),
                    np.clip(mean + std, 0.0, 1.0),
                    color=color, alpha=0.10,
                )

            # Saturation markers: thresholds where mean curves first cross a
            # per-line target. Legend + in-figure annotation report the *paired*
            # metric value at that same threshold (the cost paid on the other axis).
            annot_y = 0.62  # ~13% lower than 3/4 of the [0, 1.05] y-range
            annot_fontsize = 8  # ~15% smaller than the previous size 9
            for curve_key, paired_key, target, color, annot_template, legend_template in (
                ("global_p", "global_r", 0.999, "tab:blue",
                 "Recall@P.999={paired:.0%}",
                 "Recall @ Precision=0.999 = {paired:.0%}"),
                ("unseen_r", "unseen_p", 0.90, "tab:orange",
                 "Precision@R.90={paired:.0%}",
                 "Precision @ Recall=0.90 = {paired:.0%}"),
            ):
                mean_curve = curves[curve_key].mean(axis=0)
                mask = mean_curve >= target
                if mask.any():
                    idx = int(mask.argmax())
                    paired_value = float(curves[paired_key].mean(axis=0)[idx])
                    cross_thr = float(thresholds[idx])
                    ax.axvline(
                        cross_thr, color=color, linestyle="-", linewidth=0.6,
                        label=legend_template.format(paired=paired_value),
                    )
                    ax.text(
                        cross_thr + 0.005, annot_y,
                        annot_template.format(paired=paired_value),
                        color=color, rotation=-90,
                        ha="left", va="center", fontsize=annot_fontsize,
                        fontweight="bold",
                    )

            # Horizontal: Unseen prevalence. Prefer the caller-provided design
            # value (``m_test_size``); otherwise fall back to empirical
            # ``n_u / (n_kc + n_kw + n_u)``, which can drift slightly from the
            # configured prevalence due to integer flooring of unknown-class
            # count and per-class truncation.
            if unseen_prevalence is not None:
                mean_prev = float(unseen_prevalence)
            else:
                prevalences = []
                for it in iterations:
                    n_kc = int(np.asarray(it["kc"]).size)
                    n_kw = int(np.asarray(it["kw"]).size)
                    n_u = int(np.asarray(it["u"]).size)
                    total = n_kc + n_kw + n_u
                    if total > 0:
                        prevalences.append(n_u / total)
                mean_prev = float(np.mean(prevalences)) if prevalences else None
            if mean_prev is not None:
                prev_label = f"Unseen prevalence={mean_prev:.0%}"
                ax.axhline(
                    mean_prev, color="grey", linestyle="-", linewidth=1.5,
                    label=prev_label,
                )
                ax.text(
                    0.25, mean_prev + 0.01, prev_label,
                    color="grey",
                    ha="center", va="bottom", fontsize=annot_fontsize,
                    fontweight="bold",
                )

        ax.set_xlabel("Confidence threshold", **XLABEL_CONFIG)
        ax.set_ylabel("(Micro-averaged) Precision / Recall", **YLABEL_CONFIG)
        ax.tick_params(axis="both", which="major", **XTICKS_CONFIG)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(**GRID_CONFIG)
        ax.legend(**{**LEGEND_CONFIG, "fontsize": 12}, ncol=1)
        plt.tight_layout()

        plot_path = Path(fig_save_path) / f"pr_vs_confidence_unseen_and_global_bs_{bs}.pdf"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Unseen+Global P/R curve for bs=%s saved to %s", bs, plot_path)


def plot_unseen_and_global_pr_vs_alpha(
    pr_data: Dict[int, Dict[str, List[Dict[str, np.ndarray]]]],
    fig_save_path: str | Path,
    batch_prediction_sizes: List[int],
    n_alphas: int = 101,
) -> None:
    """P/R vs α-quantile threshold for Unseen detection and Global (Known) classification.

    Mirrors :func:`plot_unseen_and_global_pr_vs_confidence` in styling and bs scope
    (only ``bs=1`` and ``bs=max(batch_prediction_sizes)``), but the x-axis is the
    α-quantile threshold parameter instead of the raw confidence. For each outer
    iteration the applied threshold is ``np.quantile(kc, α)`` (correct-known
    scores only — matches :func:`extract_thresholds`), so α is the false-rejection
    rate among correctly-classified Knowns. Per-iteration thresholds → error
    bands reflect calibration variance under the deployable rule.

    Parameters
    ----------
    pr_data : dict
        ``{bs: {clf_name: [iteration_dict, ...]}}`` where each iteration dict has
        keys ``"kc"`` (Known-correct max scores), ``"kw"`` (Known-wrong),
        ``"u"`` (Unknown samples, all pooled).
    fig_save_path : str or Path
        Directory for saving figures.
    batch_prediction_sizes : list[int]
        All batch sizes the run uses (only ``min`` and ``max`` of this list are plotted).
    n_alphas : int
        Resolution of the α sweep.
    """
    if not batch_prediction_sizes:
        return
    bs_to_plot = sorted({1, max(batch_prediction_sizes)})
    alphas = np.linspace(0.0, 1.0, n_alphas)

    for bs in bs_to_plot:
        if bs not in pr_data or not pr_data[bs]:
            continue

        fig, ax = plt.subplots(figsize=(9, 6))

        for clf_name, iterations in pr_data[bs].items():
            if not iterations:
                continue

            per_iter = []
            for it in iterations:
                kc = np.asarray(it["kc"], dtype=float)
                if kc.size == 0:
                    continue
                kw = np.asarray(it["kw"], dtype=float)
                u = np.asarray(it["u"], dtype=float)
                t_per_alpha = np.quantile(kc, alphas)
                per_iter.append(_compute_pr_curves_single_iter(kc, kw, u, t_per_alpha))

            if not per_iter:
                continue

            curves = {
                k: np.stack([d[k] for d in per_iter], axis=0)
                for k in ("unseen_p", "unseen_r", "global_p", "global_r")
            }

            for key, color, linestyle, label in (
                ("global_p", "tab:blue", "--", f"{clf_name} — Global P"),
                ("global_r", "tab:blue", "-", f"{clf_name} — Global R"),
                ("unseen_p", "tab:orange", "--", f"{clf_name} — Unseen P"),
                ("unseen_r", "tab:orange", "-", f"{clf_name} — Unseen R"),
            ):
                mean = curves[key].mean(axis=0)
                std = curves[key].std(axis=0)
                ax.plot(
                    alphas, mean,
                    color=color, linestyle=linestyle, linewidth=2.0,
                    label=label,
                )
                ax.fill_between(
                    alphas,
                    np.clip(mean - std, 0.0, 1.0),
                    np.clip(mean + std, 0.0, 1.0),
                    color=color, alpha=0.10,
                )

        ax.set_xlabel(r"$\alpha$-Quantile Threshold", **XLABEL_CONFIG)
        ax.set_ylabel("(Micro-averaged) Precision / Recall", **YLABEL_CONFIG)
        ax.tick_params(axis="both", which="major", **XTICKS_CONFIG)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(**GRID_CONFIG)
        ax.set_title(f"P/R vs α-quantile — Unseen detection vs Global (Known) — bs={bs}")
        ax.legend(**LEGEND_CONFIG, ncol=2)
        plt.tight_layout()

        plot_path = Path(fig_save_path) / f"alpha_tradeoff_pr_unseen_and_global_bs_{bs}.pdf"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Unseen+Global P/R-vs-α curve for bs=%s saved to %s", bs, plot_path)


def _compute_alpha_roc_per_clf(
    iterations: List[Dict[str, np.ndarray]],
    alphas: np.ndarray,
) -> Optional[Dict[str, np.ndarray]]:
    """Compute mean (FPR(α), TPR(α)) curve across outer iterations for one classifier.

    Returns ``None`` if no valid iteration exists. Otherwise returns a dict with
    ``mean_fpr``, ``mean_tpr``, and ``auroc``.
    """
    tpr_stack, fpr_stack = [], []
    for it in iterations:
        kc = np.asarray(it["kc"], dtype=float)
        if kc.size == 0:
            continue
        kw = np.asarray(it["kw"], dtype=float)
        u = np.asarray(it["u"], dtype=float)
        t_per_alpha = np.quantile(kc, alphas)
        res = _compute_tpr_fpr_single_iter(kc, kw, u, t_per_alpha)
        tpr_stack.append(res["tpr"])
        fpr_stack.append(res["fpr"])

    if not tpr_stack:
        return None

    mean_tpr = np.stack(tpr_stack).mean(axis=0)
    mean_fpr = np.stack(fpr_stack).mean(axis=0)
    # Curve runs from (1,1) at α=0 to (0,0) at α=1; reverse for trapz orientation.
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auroc = float(np.abs(_trapz(mean_tpr, mean_fpr)))
    return {"mean_fpr": mean_fpr, "mean_tpr": mean_tpr, "auroc": auroc}


def plot_alpha_roc_curves(
    pr_data: Dict[int, Dict[str, List[Dict[str, np.ndarray]]]],
    fig_save_path: str | Path,
    batch_prediction_sizes: List[int],
    n_alphas: int = 101,
) -> None:
    """Plot ROC curve (FPR vs TPR) parameterised by α-quantile threshold.

    For each outer iteration the threshold at quantile α of the correct-Known
    scores (``kc``) is applied; TPR and FPR are then averaged across iterations.
    One figure per batch size (all batch sizes, unlike the P/R plots).

    The α-ROC traces the same (FPR, TPR) locus as the raw-confidence ROC but
    parameterised by the *deployable* α rule instead of an oracle threshold sweep.
    Compare with ``roc_curve_unseen_vs_known_bs_{bs}.pdf`` (oracle).
    """
    from audit_llm.plotting.constants import COLORBLIND_COLORS

    if not batch_prediction_sizes:
        return
    alphas = np.linspace(0.0, 1.0, n_alphas)

    for bs in batch_prediction_sizes:
        if bs not in pr_data or not pr_data[bs]:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        for color_idx, (clf_name, iterations) in enumerate(pr_data[bs].items()):
            result = _compute_alpha_roc_per_clf(iterations, alphas)
            if result is None:
                continue
            color = COLORBLIND_COLORS[color_idx % len(COLORBLIND_COLORS)]
            label = f"{clf_name} (AUROC = {result['auroc']:.3f})"
            ax.plot(result["mean_fpr"], result["mean_tpr"], color=color, linewidth=2, label=label)

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random")
        ax.set_xlabel("False Positive Rate (Unknown predicted as Known)", **XLABEL_CONFIG)
        ax.set_ylabel("True Positive Rate (Known correctly predicted)", **YLABEL_CONFIG)
        ax.set_title(f"ROC — α-quantile threshold — bs={bs}")
        ax.tick_params(axis="both", which="major", **XTICKS_CONFIG)
        ax.legend(**LEGEND_CONFIG)
        ax.grid(**GRID_CONFIG)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        plt.tight_layout()
        plot_path = Path(fig_save_path) / f"roc_curve_alpha_bs_{bs}.pdf"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("α-ROC curve for bs=%s saved to %s", bs, plot_path)


def plot_roc_curves_overlay(
    pr_data: Dict[int, Dict[str, List[Dict[str, np.ndarray]]]],
    roc_data: Dict[int, Dict[str, Dict]],
    fig_save_path: str | Path,
    batch_prediction_sizes: List[int],
    n_alphas: int = 101,
) -> None:
    """Overlay oracle ROC (raw-confidence sweep) and α-ROC on the same axes.

    Oracle ROC curves (from ``roc_data``) are drawn as solid lines with ±1 std
    shaded bands.  The α-ROC curve for each classifier is overlaid as a dashed
    line of the same color.  This lets you see which part of the oracle ROC space
    the deployable α-rule actually navigates.

    Parameters
    ----------
    pr_data : dict
        ``{bs: {clf_name: [iteration_dict, ...]}}`` — raw per-iteration scores.
    roc_data : dict
        ``{bs: {clf_name: {mean_fpr, mean_tpr, std_tpr, mean_auroc, std_auroc}}}``
        — pre-aggregated oracle ROC from ``_compute_and_plot_roc_curves``.
    """
    from audit_llm.plotting.constants import COLORBLIND_COLORS

    if not batch_prediction_sizes:
        return
    alphas = np.linspace(0.0, 1.0, n_alphas)

    for bs in batch_prediction_sizes:
        if (bs not in pr_data or not pr_data[bs]) and (bs not in roc_data or not roc_data[bs]):
            continue

        fig, ax = plt.subplots(figsize=(5.5, 3.5))

        # Collect classifier union from both dicts to keep colors consistent.
        clf_names = list(dict.fromkeys(
            list(roc_data.get(bs, {}).keys()) + list(pr_data.get(bs, {}).keys())
        ))

        for color_idx, clf_name in enumerate(clf_names):
            color = COLORBLIND_COLORS[color_idx % len(COLORBLIND_COLORS)]

            # --- Oracle ROC (solid) ---
            if clf_name in roc_data.get(bs, {}):
                d = roc_data[bs][clf_name]
                label_oracle = f"{clf_name} oracle (AUROC={d['mean_auroc']:.3f}±{d['std_auroc']:.3f})"
                ax.plot(d["mean_fpr"], d["mean_tpr"], color=color, linestyle="-",
                        linewidth=2, label=label_oracle)

            # --- α-ROC (dashed, same color) ---
            if clf_name in pr_data.get(bs, {}):
                result = _compute_alpha_roc_per_clf(pr_data[bs][clf_name], alphas)
                if result is not None:
                    label_alpha = f"{clf_name} α-rule (AUROC={result['auroc']:.3f})"
                    ax.plot(result["mean_fpr"], result["mean_tpr"], color=color,
                            linestyle="--", linewidth=2, label=label_alpha)

        ax.plot([0, 1], [0, 1], linestyle=":", color="gray", linewidth=1, label="Random")
        ax.set_xlabel("False Positive Rate\n(Unknown predicted as Known)", **XLABEL_CONFIG)
        ax.set_ylabel("True Positive Rate\n(Known correctly predicted)", **YLABEL_CONFIG)
        ax.tick_params(axis="both", which="major", **XTICKS_CONFIG)
        ax.legend(**{**LEGEND_CONFIG, "fontsize": 9})
        ax.grid(**GRID_CONFIG)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        plt.tight_layout()
        plot_path = Path(fig_save_path) / f"roc_curve_alpha_overlay_bs_{bs}.pdf"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("ROC overlay (oracle+α) for bs=%s saved to %s", bs, plot_path)
