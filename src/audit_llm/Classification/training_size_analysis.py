"""
Training size analysis and curve plotting functions.

Provides functions for plotting metric curves across different training sizes, creating
train-size figures, and generating trainsize-wise analysis plots.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import logging

logger = logging.getLogger(__name__)

from audit_llm.plot_configs import (
    FIG_CONFIG_TEMP_TR_SIZE,
    GRID_CONFIG,
    LEGEND_CONFIG,
    XLABEL_CONFIG,
    YLABEL_CONFIG,
    YTICKS_METRIC_CONFIG,
    XTICKS_CONFIG,
    YTICKS_CONFIG,
    get_mpl_configs,
)
from audit_llm.plotting.constants import COLOR_DELTA_TP_MAP
from audit_llm.plotting.figure_io import save_fig_and_show
from audit_llm.xp_tools.label_formatting import format_tp_group_name_label, put_uppercase_first
from audit_llm.xp_tools import (
    get_token_pairs_of_group,
    get_tp_names_of_group,
    load_train_size_dict,
)
from audit_llm.Bits_Generation.parsing_bits_tools import token_pair_name_to_items


def compute_means_stds(train_size_dict, train_sizes, temp, clf, metric, tp_names, summary_key, bs):
    means, stds = [], []
    for ts in train_sizes:
        summary = train_size_dict[ts][temp]
        tp_data = summary[summary_key][bs]
        tp_summary_mean = np.mean(
            [tp_data[tp][clf][f"{metric}_mean"] for tp in tp_names], axis=0
        )  # This value [tp_data[tp][clf][f'{metric}_mean'] is a mean over the test_splits; tp_summary_mean is a mean over the tp-uplets.
        tp_summary_std = np.std(
            [tp_data[tp][clf][f"{metric}_mean"] for tp in tp_names], axis=0
        )  # Averaging std over token pairs note that [f'{metric}_std'] exists but we want to do the std over the tp-uplets.
        means.append(tp_summary_mean)
        stds.append(tp_summary_std)
    return means, stds


def plot_train_size_curves(
    train_size_dict: Dict[float, Dict[float, Dict[Any, Any]]],
    fig_save_path: Path,
    classification_config: Dict,
    batch_sizes: List[int] = [1, 2, 3, 4, 5, 8],
    token_pairs=None,
    show: bool = False,
) -> None:
    """
    Plot metric curves across test sizes for given temperatures, classifiers, batch sizes, and batch type.

    Args:
        train_size_dict: mapping train_size -> { temperature -> pipe_summary_dict }
        metrics: list of metric names to plot (e.g. ['accuracy', 'f1'])
        batch_sizes: list of batch sizes to include in curves
        batch_type: one of 'mix_tp_at_pred', 'tp_wise', or 'across_and_tp_wise'
        fig_save_path: directory path to save figures
        show: whether to display figures interactively

    For each temperature, for each classifier, and for each metric, produces a plot
    where each batch_size in batch_sizes is a separate curve (mean ± std error bars).
    For 'across_and_tp_wise' batch_type, also creates tp_wise plots for each batch_size.
    Figures are saved via save_fig_and_show().

    Info on summary objects:
        summary[key_name][batch_size][tp][clf][f'{metric}_mean'] = np.mean(vals)
        summary[key_name][batch_size][tp][clf][f'{metric}_std'] = np.std(vals)

        where key_name in ['tp_wise', 'mix_tp_at_pred']
        where vals are values of the metrics (mean and std refers to splits)
        <metric> can be 'confusion_matrix' to get the mean confusion matrix and the std (over splits)

    """
    if classification_config.get("openset", False):
        logger.info("Skipping plot_train_size_curves for open-set mode (dedicated open-set plots are used instead).")
        return

    # Merged-wrapper mode (train_size_dict_map): the wrapper's token_pairs pool may differ
    # from each source XP's, so regenerating uplets via get_tp_names_of_group would KeyError.
    # Only F01 is meaningful in merged mode; skip this per-source train-size sweep plot.
    if classification_config.get("train_size_dict_map"):
        logger.info("Skipping plot_train_size_curves for merged-wrapper mode (train_size_dict_map is set).")
        return

    batch_types = classification_config.get("batch_types") or ["tp_wise"]
    # sorted list of test sizes and temperatures
    train_sizes = sorted(train_size_dict.keys())
    logger.debug(f"{train_sizes = }")
    temperatures = sorted({t for ts in train_size_dict.values() for t in ts.keys()})

    for temp in temperatures:
        for clf in classification_config["classifiers"]:
            for batch_type in batch_types:
                for metric in classification_config["classifier_metrics"]:
                    tr_sizes_fig_path = Path(fig_save_path) / str(temp) / str(clf) / "TrainSizes" / metric
                    tr_sizes_fig_path.mkdir(exist_ok=True, parents=True)
                    for tp_group_name in ["FLiPS", "0-1"]:
                        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(batch_sizes)))  # type:ignore
                        fig, ax = plt.subplots(**FIG_CONFIG_TEMP_TR_SIZE)
                        for bs in batch_sizes:
                            if (
                                bs == 1 or tp_group_name == "0-1" or batch_type == "tp_wise"
                            ):  # Second condition only met for 0-1 group.
                                tp_names = get_tp_names_of_group(
                                    tp_group_name, mode="tp_wise", bs=bs, token_pairs=token_pairs
                                )
                                summary_key = "tp_wise"
                            else:
                                utp = classification_config["unique_tp_in_mix"]
                                utp_default = utp[0] if isinstance(utp, list) else utp
                                tp_names = get_tp_names_of_group(
                                    tp_group_name,
                                    mode="mix_tp_at_pred",
                                    bs=bs,
                                    token_pairs=token_pairs,
                                    max_nb_of_uplet=classification_config["max_nb_of_uplet"],
                                    unique_elements=utp_default,
                                )
                                if not tp_names:
                                    # Fallback to tp_wise when unique_tp > bs
                                    tp_names = get_tp_names_of_group(
                                        tp_group_name, mode="tp_wise", bs=bs, token_pairs=token_pairs
                                    )
                                    summary_key = "tp_wise"
                                else:
                                    summary_key = batch_type

                            means, stds = compute_means_stds(
                                train_size_dict, train_sizes, temp, clf, metric, tp_names, summary_key, bs
                            )

                            ax.errorbar(
                                train_sizes,
                                means,
                                # Using other color scheme than COLORBLIND_COLORS to avoid confusion with temperature plots.
                                color=colors[batch_sizes.index(bs)],
                                yerr=stds,
                                marker="o",
                                linestyle="-",
                                label=f"n={bs}",
                            )

                        ax.set_xlabel("Training Samples", **XLABEL_CONFIG)
                        ax.set_ylabel(f"{put_uppercase_first(metric)}", **YLABEL_CONFIG)
                        ax.set_xticks(train_sizes)
                        ax.set_yticks(**YTICKS_METRIC_CONFIG(f"{metric}_{tp_group_name}"))
                        ax.tick_params(axis="x", **XTICKS_CONFIG)
                        ax.tick_params(axis="y", **YTICKS_CONFIG)
                        ax.grid(**GRID_CONFIG)
                        # Increase legend size
                        ax.legend(title="Number of Queries", loc="lower right", **LEGEND_CONFIG)
                        fig.tight_layout()  # #fig.set_size_inches() if you want do it manually for fig consistency across figures.

                        fname = f"{tp_group_name}_{metric}_{temp}_tr_size_curve.pdf"
                        save_fig_and_show(tr_sizes_fig_path, show, fname)


def plot_trainsize_wise_curves(
    train_size_dict: Dict[float, Dict[float, Dict[Any, Any]]],
    fig_save_path: Path,
    classification_config: Dict,
    datasets: Optional[list[str]],
    models_idx: Dict,
    batch_sizes: List[int] = [1, 2, 3, 4, 5, 8],
    tp_group_names: List[str] = [
        "FLiPS",
        # '0-1'
    ],
) -> None:
    """
    Plot metric curves across test sizes for given calculation_item_name, classifiers, batch sizes, and batch type.

    Args:
        train_size_dict: mapping train_size -> calculation_item_name -> summary
        metrics: list of metric names to plot (e.g. ['accuracy', 'f1'])
        batch_sizes: list of batch sizes to include in curves
        batch_type: one of 'mix_tp_at_pred', 'tp_wise', or 'across_and_tp_wise'
        fig_save_path: directory path to save figures
        show: whether to display figures interactively

    For each calculation_item_name, for each classifier, and for each metric, produces a plot
    where each batch_size in batch_sizes is a separate curve (mean ± std error bars).
    For 'across_and_tp_wise' batch_type, also creates tp_wise plots for each batch_size.
    Figures are saved via save_fig_and_show().

    Info on summary objects:
        summary[key_name][batch_size][tp][clf][f'{metric}_mean'] = np.mean(vals)
        summary[key_name][batch_size][tp][clf][f'{metric}_std'] = np.std(vals)

        where key_name in ['tp_wise', 'mix_tp_at_pred']
        where vals are values of the metrics.

        metrics can be "confusion_matrix_{mean/std}" and confusion matrices are aligned with models_idx

    """

    batch_types = classification_config.get("batch_types") or ["tp_wise"]
    metrics: List[str] = classification_config["classifier_metrics"]

    # Load additional train_size_dicts if provided
    train_size_dict_map = classification_config.get("train_size_dict_map", None)
    logger.debug(f"{train_size_dict_map = }")

    # Prepare lists to store all loaded dicts
    train_size_dicts_to_plot = []  # List of tuples: (train_size_dict, models_idx, label)

    if train_size_dict_map is not None:
        # Load all dictionaries from the map
        for path, label in train_size_dict_map.items():
            loaded_dict, loaded_models_idx = load_train_size_dict(path, label)
            train_size_dicts_to_plot.append((loaded_dict, loaded_models_idx, label))
    else:
        # Fallback to original train_size_dict
        train_size_dicts_to_plot.append((train_size_dict, models_idx, "FLiPS"))

    # Use the first entry as the main one for determining structure
    train_size_dict_main, new_models_idx_main, _ = train_size_dicts_to_plot[0]

    train_sizes = sorted(train_size_dict_main.keys())
    calculation_item_names = sorted({t for ts in train_size_dict_main.values() for t in ts.keys()})

    logger.debug(f"{train_sizes = }, {calculation_item_names =}")

    # Openset handling - apply to all loaded dicts

    for i, (tsd, mid, label) in enumerate(train_size_dicts_to_plot):
        if "Open-set" in label:
            mid[len(mid)] = "Unseen"
            train_size_dicts_to_plot[i] = (tsd, mid, label)

    for calculation_item_name in calculation_item_names:
        for clf in classification_config["classifiers"]:
            for train_size in train_sizes:
                for batch_type in batch_types:
                    summary_results = train_size_dict_main[train_size][calculation_item_name]
                    if datasets is None:
                        first_key = next((k for k in summary_results if k in batch_types), batch_types[0])
                        datasets = list(summary_results[first_key][batch_sizes[0]].keys())

                    # Making Tables
                    save_tables_path = (
                        Path(fig_save_path)
                        / str(calculation_item_name)
                        / str(clf)
                        / str(train_size)
                        / "ModelWiseTables"
                    )
                    save_tables_path.mkdir(exist_ok=True, parents=True)

                    # Collect all summary results for plotting multiple curves
                    all_summary_results = []
                    for tsd, mid, label in train_size_dicts_to_plot:
                        summary = tsd[train_size][calculation_item_name]
                        all_summary_results.append((summary, label))
                    for metric in metrics:
                        logger.debug(f"{calculation_item_name =}, {clf = }, {train_size = }, {metric = }")
                        save_curves_path = (
                            Path(fig_save_path)
                            / str(calculation_item_name)
                            / str(clf)
                            / str(train_size)
                            / "ModelWiseCurves"
                            / metric
                        )
                        save_curves_path.mkdir(exist_ok=True, parents=True)
                        plot_classifier_curves(
                            summary_results,
                            clf,
                            batch_sizes,
                            metric,
                            save_curves_path,
                            datasets,
                            batch_type,
                            train_size,
                            classification_config,
                            all_summary_results,
                            n_classes=len(new_models_idx_main) if new_models_idx_main else None,
                        )


def plot_classifier_curves(
    summary_results,
    clf,
    batch_sizes,
    metric,
    fig_save_path,
    datasets,
    batch_type,
    train_size,
    classification_config,
    all_summary_results=None,
    n_classes=None,
):
    """
    Plot curves for a single classifier.

    Args:
        all_summary_results: List of tuples (summary_results, label) for plotting multiple curves
    """
    # F01: FLiPS accuracy vs Nt (optionally vs LLMmap)
    plot_group_curves(
        summary_results,
        batch_sizes,
        clf,
        metric,
        fig_save_path,
        datasets,
        batch_type,
        tp_group_names=["FLiPS"],
        fig_name=f"F01_accuracy_vs_queries_tr{train_size}.pdf",
        classification_config=classification_config,
        all_summary_results=all_summary_results,
        n_classes=n_classes,
    )

    # Merged-wrapper mode (train_size_dict_map set): only F01 is meaningful. F02..F07 are
    # per-source analysis plots that read keys absent from a loaded checkpoint — the mix_tp uplet
    # keys (F02/F05) and, for a tp_wise-only LLMmap cache, the batch_type key itself (F03/F04/F06/F07
    # KeyError on 'mix_tp_at_pred', training_size_analysis.py:977). This holds for ANY source count:
    # both the 3-curve e3_flips_vs_llmmap config and the single-source e3_llmmap_baseline variant plot
    # only F01. (Gating on train_size_dict_map, not the old `len(all_summary_results) > 1` proxy,
    # is what fixes the 1-source LLMmap-only case.)
    if classification_config.get("train_size_dict_map"):
        return

    # F02: mix_tp_at_pred — one curve per unique_tp value
    plot_per_unique_tp_curves(
        summary_results,
        batch_sizes,
        clf,
        metric,
        fig_save_path,
        datasets,
        train_size,
        classification_config,
        summary_key_prefix="mix_tp_at_pred_utp",
        fig_name_prefix="F02_mix_pred_utp",
    )

    # F02bis: F02 + tp_wise reference curve
    plot_per_unique_tp_curves(
        summary_results,
        batch_sizes,
        clf,
        metric,
        fig_save_path,
        datasets,
        train_size,
        classification_config,
        summary_key_prefix="mix_tp_at_pred_utp",
        fig_name_prefix="F02bis_mix_pred_vs_tpwise",
        include_tp_wise_curve=True,
    )

    # F07: mix_tp_at_pred (utp='max') vs tp_wise — clean 2-curve "mixing gain" figure
    plot_mix_vs_tpwise_gain(
        summary_results,
        batch_sizes,
        clf,
        metric,
        fig_save_path,
        datasets,
        train_size,
        classification_config,
        utp_value="max",
    )

    # F05: mix_tp_at_train — one curve per unique_tp value
    plot_per_unique_tp_curves(
        summary_results,
        batch_sizes,
        clf,
        metric,
        fig_save_path,
        datasets,
        train_size,
        classification_config,
        summary_key_prefix="mix_tp_at_train_utp",
        fig_name_prefix="F05_mix_train_utp",
    )

    # F05bis: F05 + tp_wise reference curve
    plot_per_unique_tp_curves(
        summary_results,
        batch_sizes,
        clf,
        metric,
        fig_save_path,
        datasets,
        train_size,
        classification_config,
        summary_key_prefix="mix_tp_at_train_utp",
        fig_name_prefix="F05bis_mix_train_vs_tpwise",
        include_tp_wise_curve=True,
    )

    # F03: FLiPS + 0-1 accuracy curves
    plot_group_curves(
        summary_results,
        batch_sizes,
        clf,
        metric,
        fig_save_path,
        datasets,
        batch_type,
        tp_group_names=["FLiPS", "0-1"],
        fig_name=f"F03_all_groups_tr{train_size}.pdf",
        classification_config=classification_config,
    )

    # F04: Token-pair accuracy histogram
    plot_group_curves(
        summary_results,
        batch_sizes,
        clf,
        metric,
        fig_save_path,
        datasets,
        batch_type,
        tp_group_names=["FLiPS", "0-1"],
        fig_name=f"F04_histogram_tr{train_size}.pdf",
        classification_config=classification_config,
        ds_histogram=True,
    )

    # F06: Best mix_pred + best mix_train + tp_wise comparison
    plot_best_mix_comparison(
        summary_results,
        batch_sizes,
        clf,
        metric,
        fig_save_path,
        datasets,
        train_size,
        classification_config,
    )


def _parse_utp_suffix(key: str, prefix: str) -> Union[int, str]:
    suffix = key.replace(prefix, "")
    return suffix if suffix == "max" else int(suffix)


def _utp_sort_key(value: Union[int, str]) -> Tuple[int, int]:
    return (1, 0) if value == "max" else (0, value)


def plot_per_unique_tp_curves(
    summary_results: Dict,
    batch_sizes: List[int],
    clf: str,
    metric: str,
    fig_save_path,
    token_pairs: Optional[List[str]],
    train_size: int,
    classification_config: Dict,
    summary_key_prefix: str = "mix_tp_at_pred_utp",
    fig_name_prefix: str = "F02_mix_pred_utp",
    include_tp_wise_curve: bool = False,
) -> None:
    """Plot one curve per unique_tp_in_mix value.

    Collects all ``{summary_key_prefix}*`` keys from *summary_results* and plots
    an errorbar curve for each, showing how token-pair-mixing diversity
    impacts accuracy. Skipped when fewer than 2 utp entries exist.

    When *include_tp_wise_curve* is True, an additional dashed black curve
    showing pure tp_wise performance across all batch sizes is overlaid.

    Used for F02 (mix_tp_at_pred), F05 (mix_tp_at_train), and their _bis variants.
    """
    # Collect utp entries, sorted by utp value
    utp_entries = sorted(
        [(k, v) for k, v in summary_results.items() if k.startswith(summary_key_prefix)],
        key=lambda x: _utp_sort_key(_parse_utp_suffix(x[0], summary_key_prefix)),
    )
    if len(utp_entries) < 2:
        return

    fig_config = get_mpl_configs(multiplier=1.5, col_type="single_col")
    fig, ax = plt.subplots(**fig_config["fig_config"])

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(utp_entries)))  # type:ignore

    for idx, (key, utp_data) in enumerate(utp_entries):
        utp_val = _parse_utp_suffix(key, summary_key_prefix)

        plot_bs_values = []
        mean_values = []
        std_values = []

        for bs in batch_sizes:
            if bs == 1:
                tp_names = get_tp_names_of_group("FLiPS", mode="tp_wise", bs=bs, token_pairs=token_pairs)
                bs_results = summary_results.get("tp_wise", {})
            else:
                tp_names = get_tp_names_of_group(
                    "FLiPS",
                    mode="mix_tp_at_pred",
                    bs=bs,
                    token_pairs=token_pairs,
                    max_nb_of_uplet=classification_config["max_nb_of_uplet"],
                    unique_elements=utp_val,
                )
                bs_results = utp_data

            if not tp_names:
                continue  # skip this bs for this utp value (unique_tp > bs)

            tp_mean_values = []
            for tp in tp_names:
                if bs in bs_results:
                    tp_mean = bs_results[bs][tp][clf][f"{metric}_mean"]
                else:
                    tp_mean = 0
                tp_mean_values.append(tp_mean)

            tp_mean_values_arr = np.array(tp_mean_values)
            plot_bs_values.append(bs)
            mean_values.append(np.mean(tp_mean_values_arr))
            std_values.append(np.std(tp_mean_values_arr) if len(tp_names) > 1 else 0)

        if not plot_bs_values:
            continue

        ax.errorbar(
            plot_bs_values,
            mean_values,
            yerr=std_values,
            marker="o",
            linestyle="-",
            label=f"unique_tp={utp_val}",
            color=colors[idx],
        )

    # Optionally overlay a tp_wise reference curve
    if include_tp_wise_curve:
        tp_wise_data = summary_results.get("tp_wise", {})
        if tp_wise_data:
            tp_tp_names = get_tp_names_of_group("FLiPS", mode="tp_wise", bs=1, token_pairs=token_pairs)
            tp_bs_values = []
            tp_mean_values = []
            tp_std_values = []
            for bs in batch_sizes:
                if bs not in tp_wise_data:
                    continue
                ds_mean_values = []
                for tp in tp_tp_names:
                    if tp in tp_wise_data[bs]:
                        ds_mean_values.append(tp_wise_data[bs][tp][clf][f"{metric}_mean"])
                if ds_mean_values:
                    tp_bs_values.append(bs)
                    tp_mean_values.append(np.mean(ds_mean_values))
                    tp_std_values.append(np.std(ds_mean_values) if len(ds_mean_values) > 1 else 0)
            if tp_bs_values:
                ax.errorbar(
                    tp_bs_values,
                    tp_mean_values,
                    yerr=tp_std_values,
                    marker="s",
                    linestyle="--",
                    label="tp_wise",
                    color="black",
                )

    ax.set_xlabel(r"Number of queries at verification $(N_t)$", **fig_config["xlabel_config"])
    ax.set_ylabel(put_uppercase_first(metric), **fig_config["ylabel_config"])
    ax.set_xticks(batch_sizes)
    ax.legend(**{**fig_config["legend_config"], **{"loc": "lower center", "fontsize": 10.4}})
    ax.tick_params(axis="x", **fig_config["xticks_config"])
    ax.tick_params(axis="y", **fig_config["yticks_config"])
    ax.grid(**fig_config["grid_config"])
    fig.tight_layout()
    save_fig_and_show(fig_save_path, show=False, fig_name=f"{fig_name_prefix}_tr{train_size}.pdf")


def plot_mix_vs_tpwise_gain(
    summary_results: Dict,
    batch_sizes: List[int],
    clf: str,
    metric: str,
    fig_save_path,
    token_pairs: Optional[List[str]],
    train_size: int,
    classification_config: Dict,
    utp_value: Union[int, str] = "max",
    fig_name_prefix: str = "F07_mix_vs_tpwise",
) -> None:
    """Plot the accuracy gain from mix_tp_at_pred (utp=utp_value) over tp_wise (F07).

    Two curves on the same axes:
      - Solid red, circles: mix_tp_at_pred with unique_tp_in_mix=`utp_value` (default 'max').
        For bs=1, falls back to tp_wise (mixing undefined at bs=1).
      - Dashed red, squares: tp_wise across all bs.

    Mean ± std taken across the FLiPS token-pair datasets (same semantics as F02bis).
    Skipped if either `tp_wise` or `mix_tp_at_pred_utp{utp_value}` is missing.
    """
    mix_key = f"mix_tp_at_pred_utp{utp_value}"
    tp_wise_data = summary_results.get("tp_wise", {})
    mix_data = summary_results.get(mix_key, {})
    if not tp_wise_data or not mix_data:
        return

    fig_config = get_mpl_configs(multiplier=1.5, col_type="single_col")
    fig, ax = plt.subplots(**fig_config["fig_config"])

    # --- mix_tp_at_pred curve (solid red, circles); bs=1 falls back to tp_wise ---
    mix_bs_values, mix_means, mix_stds = [], [], []
    for bs in batch_sizes:
        if bs == 1:
            tp_names = get_tp_names_of_group("FLiPS", mode="tp_wise", bs=bs, token_pairs=token_pairs)
            bs_results = tp_wise_data
        else:
            tp_names = get_tp_names_of_group(
                "FLiPS",
                mode="mix_tp_at_pred",
                bs=bs,
                token_pairs=token_pairs,
                max_nb_of_uplet=classification_config["max_nb_of_uplet"],
                unique_elements=utp_value,
            )
            bs_results = mix_data
        if not tp_names or bs not in bs_results:
            continue
        tp_mean_values = [bs_results[bs][tp][clf][f"{metric}_mean"] for tp in tp_names if tp in bs_results[bs]]
        if not tp_mean_values:
            continue
        arr = np.array(tp_mean_values)
        mix_bs_values.append(bs)
        mix_means.append(float(np.mean(arr)))
        mix_stds.append(float(np.std(arr)) if len(arr) > 1 else 0.0)

    # --- tp_wise curve (dashed red, squares) ---
    tp_bs_values, tp_means, tp_stds = [], [], []
    for bs in batch_sizes:
        if bs not in tp_wise_data:
            continue
        tp_names = get_tp_names_of_group("FLiPS", mode="tp_wise", bs=bs, token_pairs=token_pairs)
        tp_mean_values = [
            tp_wise_data[bs][tp][clf][f"{metric}_mean"] for tp in tp_names if tp in tp_wise_data[bs]
        ]
        if not tp_mean_values:
            continue
        arr = np.array(tp_mean_values)
        tp_bs_values.append(bs)
        tp_means.append(float(np.mean(arr)))
        tp_stds.append(float(np.std(arr)) if len(arr) > 1 else 0.0)

    if not mix_bs_values and not tp_bs_values:
        return

    if mix_bs_values:
        ax.errorbar(
            mix_bs_values,
            mix_means,
            yerr=mix_stds,
            marker="o",
            linestyle="-",
            label="Multi Token Pair",
            color="red",
        )
    if tp_bs_values:
        ax.errorbar(
            tp_bs_values,
            tp_means,
            yerr=tp_stds,
            marker="s",
            linestyle="--",
            label="Same Token Pair",
            color="red",
        )

    ax.set_xlabel(r"Number of queries at verification $(N_t)$", **fig_config["xlabel_config"])
    ax.set_ylabel(put_uppercase_first(metric), **fig_config["ylabel_config"])
    ax.set_xticks(batch_sizes)
    ax.set_ylim(top=1.0)
    bottom = ax.get_ylim()[0]
    ax.set_yticks(np.arange(np.floor(bottom * 10) / 10, 1.001, 0.1))
    ax.legend(
        **{**fig_config["legend_config"], **{"loc": "lower center", "fontsize": 10.4, "title": "Token Pair strategy"}}
    )
    ax.tick_params(axis="x", **fig_config["xticks_config"])
    ax.tick_params(axis="y", **fig_config["yticks_config"])
    ax.grid(**fig_config["grid_config"])
    fig.tight_layout()
    save_fig_and_show(fig_save_path, show=False, fig_name=f"{fig_name_prefix}_tr{train_size}.pdf")


def _compute_utp_mean_at_bs(
    utp_data: Dict,
    bs: int,
    clf: str,
    metric: str,
    token_pairs: Optional[List[str]],
    classification_config: Dict,
    utp_val: Union[int, str],
) -> float:
    """Compute mean metric across FLiPS datasets for a given utp value at a specific batch size."""
    tp_names = get_tp_names_of_group(
        "FLiPS",
        mode="mix_tp_at_pred",
        bs=bs,
        token_pairs=token_pairs,
        max_nb_of_uplet=classification_config["max_nb_of_uplet"],
        unique_elements=utp_val,
    )
    if not tp_names or bs not in utp_data:
        return float("-inf")
    tp_mean_values = [utp_data[bs][tp][clf][f"{metric}_mean"] for tp in tp_names if tp in utp_data[bs]]
    return float(np.mean(tp_mean_values)) if tp_mean_values else float("-inf")


def plot_best_mix_comparison(
    summary_results: Dict,
    batch_sizes: List[int],
    clf: str,
    metric: str,
    fig_save_path,
    token_pairs: Optional[List[str]],
    train_size: int,
    classification_config: Dict,
) -> None:
    """Plot best mix_tp_at_pred + best mix_tp_at_train + tp_wise on one figure (F06).

    For each mix mode, selects the unique_tp value with the highest mean accuracy
    at bs=8 (or highest available bs). Skipped if no mix data is available.
    """
    # Collect utp entries for both mix modes
    mix_pred_entries = sorted(
        [(k, v) for k, v in summary_results.items() if k.startswith("mix_tp_at_pred_utp")],
        key=lambda x: _utp_sort_key(_parse_utp_suffix(x[0], "mix_tp_at_pred_utp")),
    )
    mix_train_entries = sorted(
        [(k, v) for k, v in summary_results.items() if k.startswith("mix_tp_at_train_utp")],
        key=lambda x: _utp_sort_key(_parse_utp_suffix(x[0], "mix_tp_at_train_utp")),
    )

    if not mix_pred_entries and not mix_train_entries:
        return  # No mix data at all

    # Determine the reference batch size for "best" selection: bs=8 or highest available
    ref_bs = 8 if 8 in batch_sizes else max(batch_sizes)

    fig_config = get_mpl_configs(multiplier=1.5, col_type="single_col")
    fig, ax = plt.subplots(**fig_config["fig_config"])

    curve_colors = {"tp_wise": "black", "mix_pred": "#e41a1c", "mix_train": "#377eb8"}
    curve_linestyles = {"tp_wise": "--", "mix_pred": "-", "mix_train": "-"}
    curve_markers = {"tp_wise": "s", "mix_pred": "o", "mix_train": "^"}

    # --- tp_wise curve ---
    tp_wise_data = summary_results.get("tp_wise", {})
    if tp_wise_data:
        tp_names = get_tp_names_of_group("FLiPS", mode="tp_wise", bs=1, token_pairs=token_pairs)
        tp_bs_values, tp_means, tp_stds = [], [], []
        for bs in batch_sizes:
            if bs not in tp_wise_data:
                continue
            tp_vals = [tp_wise_data[bs][tp][clf][f"{metric}_mean"] for tp in tp_names if tp in tp_wise_data[bs]]
            if tp_vals:
                tp_bs_values.append(bs)
                tp_means.append(np.mean(tp_vals))
                tp_stds.append(np.std(tp_vals) if len(tp_vals) > 1 else 0)
        if tp_bs_values:
            ax.errorbar(
                tp_bs_values, tp_means, yerr=tp_stds,
                marker=curve_markers["tp_wise"], linestyle=curve_linestyles["tp_wise"],
                label="tp_wise", color=curve_colors["tp_wise"],
            )

    # --- Helper to find best utp and plot its curve ---
    def _plot_best_utp_curve(utp_entries, prefix, curve_key):
        if len(utp_entries) < 2:
            return
        # Select best utp by metric at ref_bs
        best_utp_key, best_utp_data, best_utp_val = None, None, None
        best_score = float("-inf")
        for key, utp_data in utp_entries:
            utp_val = _parse_utp_suffix(key, prefix)
            score = _compute_utp_mean_at_bs(
                utp_data, ref_bs, clf, metric, token_pairs, classification_config, utp_val,
            )
            if score > best_score:
                best_score = score
                best_utp_key, best_utp_data, best_utp_val = key, utp_data, utp_val

        if best_utp_data is None:
            return

        # Plot the best utp's full curve
        plot_bs_values, mean_values, std_values = [], [], []
        for bs in batch_sizes:
            if bs == 1:
                tp_names = get_tp_names_of_group("FLiPS", mode="tp_wise", bs=bs, token_pairs=token_pairs)
                bs_results = summary_results.get("tp_wise", {})
            else:
                tp_names = get_tp_names_of_group(
                    "FLiPS", mode="mix_tp_at_pred", bs=bs, token_pairs=token_pairs,
                    max_nb_of_uplet=classification_config["max_nb_of_uplet"],
                    unique_elements=best_utp_val,
                )
                bs_results = best_utp_data

            if not tp_names:
                continue

            tp_vals = []
            for tp in tp_names:
                if bs in bs_results and tp in bs_results[bs]:
                    tp_vals.append(bs_results[bs][tp][clf][f"{metric}_mean"])
            if tp_vals:
                plot_bs_values.append(bs)
                mean_values.append(np.mean(tp_vals))
                std_values.append(np.std(tp_vals) if len(tp_vals) > 1 else 0)

        if plot_bs_values:
            mode_label = "mix_pred" if "pred" in curve_key else "mix_train"
            ax.errorbar(
                plot_bs_values, mean_values, yerr=std_values,
                marker=curve_markers[curve_key], linestyle=curve_linestyles[curve_key],
                label=f"{mode_label} (utp={best_utp_val})", color=curve_colors[curve_key],
            )

    _plot_best_utp_curve(mix_pred_entries, "mix_tp_at_pred_utp", "mix_pred")
    _plot_best_utp_curve(mix_train_entries, "mix_tp_at_train_utp", "mix_train")

    ax.set_xlabel(r"Number of queries at verification $(N_t)$", **fig_config["xlabel_config"])
    ax.set_ylabel(put_uppercase_first(metric), **fig_config["ylabel_config"])
    ax.set_xticks(batch_sizes)
    ax.legend(**{**fig_config["legend_config"], **{"loc": "lower center", "fontsize": 10.4}})
    ax.tick_params(axis="x", **fig_config["xticks_config"])
    ax.tick_params(axis="y", **fig_config["yticks_config"])
    ax.grid(**fig_config["grid_config"])
    fig.tight_layout()
    save_fig_and_show(fig_save_path, show=False, fig_name=f"F06_best_mix_tr{train_size}.pdf")


def plot_group_curves(
    batch_type_dict,
    batch_sizes,
    clf,
    metric,
    fig_save_path,
    token_pairs,
    batch_type,
    tp_group_names: List[str],
    fig_name: str,
    classification_config: Dict,
    ds_histogram: bool = False,
    all_summary_results=None,
    n_classes=None,
):
    """
    Plot dataset group curves with support for multiple data sources.

    Args:
        all_summary_results: List of tuples (summary_results, label) for plotting multiple curves
    """
    # Figure 1: Vitrine Figure (full tp_wise for 01 and FLiPS possibly vs LLMmap)
    fig_config = get_mpl_configs(multiplier=1.5, col_type="single_col")
    if ds_histogram:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig, ax = plt.subplots(**fig_config["fig_config"])
    bs_group_map_values = {}

    # Define colors and linestyles for curves

    all_colors = ["red", "#4daf4a", "#984ea3", "brown", "pink", "gray", "cyan", "magenta", "yellow"]

    # Plot all summary results
    if all_summary_results is not None:
        for idx, (summary_result, label) in enumerate(all_summary_results):
            if "LLMmap" in label:
                # LLMmap handling: single dataset, single classifier
                # Extract the single clf from dict
                extra_clf = list(summary_result["tp_wise"][batch_sizes[0]].values())[0]
                extra_clf = list(extra_clf.keys())[0]
                # Extract the single tp from dict
                extra_tp = list(summary_result["tp_wise"][batch_sizes[0]].keys())[0]

                # Build bs_map_values for curve
                bs_map_values_extra = {}
                for bs in batch_sizes:
                    bs_map_values_extra[bs] = (summary_result["tp_wise"], [extra_tp])

                # Plot curve with unique color/style and label
                color_idx = idx % len(all_colors)
                plot_dataset_group_curve(
                    ax,
                    extra_clf,
                    metric,
                    bs_map_values_extra,
                    label,
                    color=all_colors[color_idx],
                    ds_histogram=False,
                )
            elif "FLiPS".lower() in label.lower():
                # FLiPS handling: multiple dataset groups
                for tp_group_name in tp_group_names:
                    bs_map_values = {}
                    tp_group = get_token_pairs_of_group(tp_group_name, token_pairs=token_pairs)
                    for bs in batch_sizes:
                        if bs == 1 or len(tp_group) < bs or batch_type == "tp_wise":
                            # tp_wise: use saved keys directly — works across source XPs
                            # whose token_pairs lists differ (avoids uplet/pool mismatch).
                            tp_wise_bs = summary_result.get("tp_wise", {}).get(bs, {})
                            tp_names_of_group = list(tp_wise_bs.keys())
                            bs_map_values[bs] = (summary_result["tp_wise"], tp_names_of_group)
                        else:
                            # mix_tp_at_pred: same — use whatever uplets the source XP saved,
                            # not regenerated via random_combinations (pool-order-dependent).
                            mix_bs = summary_result.get(batch_type, {}).get(bs, {})
                            tp_names_of_group = list(mix_bs.keys())
                            if not tp_names_of_group:
                                # Fallback to tp_wise (e.g., bs==1 not applicable to mix)
                                tp_wise_bs = summary_result.get("tp_wise", {}).get(bs, {})
                                tp_names_of_group = list(tp_wise_bs.keys())
                                bs_map_values[bs] = (summary_result["tp_wise"], tp_names_of_group)
                            else:
                                bs_map_values[bs] = (summary_result[batch_type], tp_names_of_group)
                        if tp_group_name == "0-1":
                            logger.debug(f"{tp_names_of_group =}")

                    # Use unique color per FLiPS curve, use label for legend
                    color_idx = idx % len(all_colors)
                    # For FLiPS, use the label as the legend name instead of tp_group_name
                    plot_dataset_group_curve(
                        ax, clf, metric, bs_map_values, label, color=all_colors[color_idx], ds_histogram=ds_histogram
                    )
                    bs_group_map_values[tp_group_name] = bs_map_values
    else:
        # Fallback to original behavior if no all_summary_results
        for tp_group_name in tp_group_names:
            bs_map_values = {}
            tp_group = get_token_pairs_of_group(tp_group_name, token_pairs=token_pairs)
            for bs in batch_sizes:
                if bs == 1 or len(tp_group) < bs or batch_type == "tp_wise":
                    tp_names_of_group = get_tp_names_of_group(
                        tp_group_name, mode="tp_wise", bs=bs, token_pairs=token_pairs
                    )
                    bs_map_values[bs] = (batch_type_dict["tp_wise"], tp_names_of_group)
                else:
                    utp = classification_config["unique_tp_in_mix"]
                    utp_default = utp[0] if isinstance(utp, list) else utp
                    tp_names_of_group = get_tp_names_of_group(
                        tp_group_name,
                        mode="mix_tp_at_pred",
                        bs=bs,
                        token_pairs=token_pairs,
                        max_nb_of_uplet=classification_config["max_nb_of_uplet"],
                        unique_elements=utp_default,
                    )
                    if not tp_names_of_group:
                        # Fallback to tp_wise when unique_tp_in_mix > bs
                        tp_names_of_group = get_tp_names_of_group(
                            tp_group_name, mode="tp_wise", bs=bs, token_pairs=token_pairs
                        )
                        bs_map_values[bs] = (batch_type_dict["tp_wise"], tp_names_of_group)
                    else:
                        bs_map_values[bs] = (batch_type_dict[batch_type], tp_names_of_group)
                if tp_group_name == "0-1":
                    logger.debug(f"{tp_names_of_group =}")
            plot_dataset_group_curve(
                ax,
                clf,
                metric,
                bs_map_values,
                tp_group_name,
                color=COLOR_DELTA_TP_MAP.get(tp_group_name, None),
                ds_histogram=ds_histogram,
            )
            bs_group_map_values[tp_group_name] = bs_map_values

    if ds_histogram:
        tp_acc_data = plot_token_pairs_acc_histogram(ax, clf, metric, bs_group_map_values)
        if tp_acc_data and fig_save_path:
            Path(fig_save_path).mkdir(parents=True, exist_ok=True)
            md_path = Path(fig_save_path) / fig_name.replace(".pdf", ".md")
            md_path.write_text(make_token_pair_acc_table_markdown(tp_acc_data), encoding="utf-8")
            logger.info(f"Saved token-pair accuracy table to {md_path}")
        # Labels and legend
        ax.set_xlabel(f"(Single query) {put_uppercase_first(metric)}", **fig_config["xlabel_config"])
        ax.set_ylabel("Number of Token Pairs", **fig_config["ylabel_config"])
        ax.set_ylim(0, None)  # let matplotlib choose top, but not negative
        ax.tick_params(axis="x", **{**fig_config["xticks_config"], **{"rotation": 45}})
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.grid(True, color="grey", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)
        # Desired order
        legend_order = ["FLiPS", "0-1"]
        # Build legend handles in that order (skip missing groups)
        handles, labels = ax.get_legend_handles_labels()
        order = []
        for name in legend_order:
            formatted = format_tp_group_name_label(name)
            if formatted in labels:
                order.append(labels.index(formatted))
        if order:
            ax.legend(
                [handles[i] for i in order], [labels[i] for i in order], **fig_config["legend_config"], loc="upper left"
            )
        elif handles:
            ax.legend(**fig_config["legend_config"], loc="upper left")
    else:
        # F01 multi-source path: draw random-chance baseline at 1/n_classes
        if all_summary_results is not None and n_classes:
            chance = 1.0 / n_classes
            ax.axhline(
                chance,
                color="black",
                linestyle="--",
                linewidth=2.0,
                label=f"Random chance (1/{n_classes}={chance:.4f})",
            )
        ax.set_xticks(batch_sizes)
        ax.set_ylabel(put_uppercase_first(metric), **fig_config["ylabel_config"])

        # Determine figure type name based on what's being plotted
        if all_summary_results is not None and len(all_summary_results) > 1:
            # Check if we have both FLiPS and LLMmap labels
            labels_present = [label for _, label in all_summary_results]

            # Check if the substrings exist within any of the labels
            has_flips = any("FLiPS".lower() in label.lower() for label in labels_present)
            has_llmmap = any("LLMmap" in label for label in labels_present)
            if has_flips and has_llmmap:
                ftype_name = "FLiPS_vs_LLMmap"
            else:
                ftype_name = "_".join(set(labels_present))
        elif "0-1" in tp_group_names:
            ftype_name = "0-1"
        else:
            ftype_name = tp_group_names[0]

        ax.set_yticks(**YTICKS_METRIC_CONFIG(f"{metric}_{ftype_name}"))
        ax.set_xlabel(r"Number of queries at verification $(N_t)$", **fig_config["xlabel_config"])
        ax.legend(
            **{
                **fig_config["legend_config"],
                **{
                    "loc": "lower center",
                    "bbox_to_anchor": (0.5, 0.04),
                    "ncol": 2,
                    "fontsize": 10.4 / 1.3,
                },
            }
        )
        ax.tick_params(axis="x", **fig_config["xticks_config"])
        ax.tick_params(axis="y", **fig_config["yticks_config"])

    ax.grid(**fig_config["grid_config"])

    fig.tight_layout()
    save_fig_and_show(fig_save_path, show=False, fig_name=fig_name)


def plot_dataset_group_curve(
    ax,
    clf,
    metric,
    bs_map_values,
    group_name: str,
    color: str,
    ds_histogram: bool = False,
    linestyle=None,
):
    """
    Plot across-dataset quantiles with error bars.

    bs_map_values: {bs: (results, tp_names_of_group)}
    - results[bs][tp][clf][metric] is a 1d list of vals where each val correspond to a split
        from the train/test split.

    If ds_histogram is False:
    - compute mean across datasets for each batch size and plot a single curve (with std
        across dataset means as yerr), same as before.

    If ds_histogram is True:
    - rank datasets by their overall mean metric (averaged over batch sizes and splits),
        select top 5 and bottom 5, and plot an individual curve for each selected dataset.
    - For each selected dataset, each point's y is mean(arr) and the yerr is std(arr).
    """
    # Prepare sorted batch sizes
    batch_sizes = sorted(bs_map_values.keys())

    if not ds_histogram:
        # Collect mean values and std values per batch size (across datasets)
        mean_values = []
        std_values = []

        for bs in batch_sizes:
            bs_results, tp_names = bs_map_values[bs]
            tp_mean_values = []
            for tp in tp_names:
                if bs in bs_results:
                    tp_mean = bs_results[bs][tp][clf][f"{metric}_mean"]
                else:
                    tp_mean = 0  # hardcoded fallback for the moment
                tp_mean_values.append(tp_mean)
                # for 0-1 group, we have only one dataset, so std is taken over n_splits
                if len(tp_names) == 1 and bs in bs_results:  # group_name != '0-1':
                    std_value = bs_results[bs][tp][clf][f"{metric}_std"]
                    logger.debug(f"{bs =} {group_name =}, {tp =}, {std_value = }")
                else:
                    std_value = 0

            tp_mean_values = np.array(tp_mean_values)
            mean_value = np.mean(tp_mean_values)
            if len(tp_names) != 1:  # group_name != '0-1':
                std_value = np.std(tp_mean_values)

            mean_values.append(mean_value)
            std_values.append(std_value)

        mean_values = np.array(mean_values)
        std_values = np.array(std_values)

        ax.errorbar(
            batch_sizes,
            mean_values,
            yerr=std_values,
            marker="o",
            linestyle=linestyle if linestyle is not None else "-",
            label=format_tp_group_name_label(group_name),
            color=color,
        )

        logger.debug(f"{group_name =}, {mean_values =}")


def plot_token_pairs_acc_histogram(
    ax, clf, metric, bs_group_map_values, max_curves: int = 5, n_bins: int = 25, width_factor: float = 0.9
):
    """
    Plots a single side-by-side histogram comparing groups in bs_group_map_values.
    Y axis = proportion of samples in each bin (so groups with different sizes are comparable).
    Parameters:
      - ax: matplotlib Axes
      - clf, metric: as before (used to extract f"{metric}_mean")
      - bs_group_map_values: dict mapping tp_group_name -> bs_map_values (same structure you used)
      - max_curves: used only for printed top/bottom dataset lists (kept your behaviour)
      - n_bins: number of x bins (thin binning if large)
      - width_factor: how wide bars are relative to bin width (0..1)
    """
    # Collect values per group
    group_values = {}
    tp_acc_data: Dict[str, Dict[str, float]] = {}
    # Keep the printing logic (top/bottom) for each group
    for tp_group_name, bs_map_values in bs_group_map_values.items():
        batch_sizes = sorted(bs_map_values.keys())
        if not batch_sizes:
            logger.warning("Histogram mode skipped: no batch sizes available.")
            return {}
        # Prefer bs=8 (the typical reference), else fall back to the max available bs
        target_bs = 8 if 8 in batch_sizes else max(batch_sizes)
        bs_results, tp_names = bs_map_values[target_bs]

        tp_overall_means = {}
        for tp in tp_names:
            # mean is done over the splits here in your structure
            tp_overall_means[tp] = bs_results[target_bs][tp][clf][f"{metric}_mean"]

        # Sort datasets by overall mean (descending)
        sorted_tp = sorted(tp_names, key=lambda d: tp_overall_means[d], reverse=True)

        # Choose top and bottom sets for printing (keeps your original logic)
        n_top = min(max_curves, len(sorted_tp))
        top_tp = sorted_tp[:n_top]
        bottom_tp = sorted_tp[
            max(
                0,
                len(sorted_tp) - min(max_curves, len(sorted_tp) - n_top if len(sorted_tp) > n_top else len(sorted_tp)),
            ) :
        ]
        bottom_tp = [d for d in bottom_tp if d not in top_tp]
        top_tp_text = "\n ".join(f"{token_pair_name_to_items(tp)}: {tp_overall_means[tp]:.4f}" for tp in top_tp)
        bottom_tp_text = "\n ".join(f"{token_pair_name_to_items(tp)}: {tp_overall_means[tp]:.4f}" for tp in bottom_tp)
        logger.info(f"Top {n_top} datasets for {tp_group_name} by {metric}:\n {top_tp_text}")
        logger.info(f"Bottom {len(bottom_tp)} datasets for {tp_group_name} by {metric}:\n {bottom_tp_text}")

        # store numeric values for plotting
        group_values[tp_group_name] = list(tp_overall_means.values())
        tp_acc_data[tp_group_name] = {tp: tp_overall_means[tp] for tp in sorted_tp}

    if not group_values:
        raise ValueError("No group values found to plot.")

    # Flatten all values to compute shared bin edges
    all_values = np.concatenate([np.asarray(v) for v in group_values.values() if len(v) > 0])
    if all_values.size == 0:
        raise ValueError("No numeric metric values found in groups.")

    # Fixed binning: [0.30, 0.35, ..., 1.00] (step 0.05) so x-axis labels are standard.
    bin_edges = np.round(np.arange(0.30, 1.00 + 1e-9, 0.05), 2)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width = bin_edges[1] - bin_edges[0]
    # Only non-'0-1' groups render as bars ('0-1' is a vline); share bin width across them.
    bar_group_names = [g for g in group_values if g != "0-1"]
    n_bar_groups = max(len(bar_group_names), 1)
    single_bar_width = (bin_width * width_factor) / n_bar_groups

    # Actually draw bars
    for i, (tp_group_name, values) in enumerate(group_values.items()):
        values = np.asarray(values)
        if values.size == 0:
            # skip empty groups but keep legend entry
            ax.bar(
                [],
                [],
                label=format_tp_group_name_label(tp_group_name),
                color=COLOR_DELTA_TP_MAP.get(tp_group_name, None),
            )
            continue
        if tp_group_name == "0-1":
            # puts a vertical bar on the single value represented in '0-1', so that the y-axis of proportion won't get to 100%.
            # Add the value on the vline
            mean_val = values.mean()
            ax.axvline(
                mean_val,
                color=COLOR_DELTA_TP_MAP.get(tp_group_name, None),
                linestyle="--",
                label=format_tp_group_name_label(tp_group_name),
            )
            ax.text(
                mean_val,
                ax.get_ylim()[1] * 0.2,
                f"{mean_val:.2f}",
                color=COLOR_DELTA_TP_MAP.get(tp_group_name, None),
                rotation=90,
                verticalalignment="center",
                horizontalalignment="right",
                backgroundcolor="white",
            )
        else:
            counts, _ = np.histogram(values, bins=bin_edges)

            # Align bars to bin's left edge (= x-tick); side-by-side when multiple bar groups.
            j = bar_group_names.index(tp_group_name)
            xs = bin_edges[:-1] + j * single_bar_width
            color = COLOR_DELTA_TP_MAP.get(tp_group_name, None)
            ax.bar(
                xs,
                counts,
                width=single_bar_width,
                align="edge",
                label=format_tp_group_name_label(tp_group_name),
                alpha=0.7,
                edgecolor="black",
                color=color,
            )

    # X-ticks at the bin edges so users see round values (0.30, 0.35, ..., 1.00).
    ax.set_xticks(bin_edges)
    ax.set_xticklabels([f"{x:.2f}" for x in bin_edges])
    ax.set_xlim(bin_edges[0] - bin_width / 2.0, bin_edges[-1] + bin_width / 2.0)

    return tp_acc_data


def make_token_pair_acc_table_markdown(tp_acc_data: Dict[str, Dict[str, float]]) -> str:
    """Generate a markdown table of per-token-pair accuracy values, sorted top to lowest, with mean and std."""
    lines = []
    for group_name, tp_dict in tp_acc_data.items():
        lines.append(f"## {group_name}")
        lines.append("")
        lines.append("| Token Pair | Accuracy |")
        lines.append("| --- | --- |")
        values = list(tp_dict.values())
        for tp, acc in tp_dict.items():
            tp_label = "-".join(token_pair_name_to_items(tp))
            lines.append(f"| {tp_label} | {acc:.4f} |")
        if values:
            lines.append(f"| **Mean** | {np.mean(values):.4f} |")
            lines.append(f"| **Std** | {np.std(values):.4f} |")
        lines.append("")
    return "\n".join(lines)
