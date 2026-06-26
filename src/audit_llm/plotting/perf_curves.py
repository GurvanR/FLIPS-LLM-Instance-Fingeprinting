"""Performance curve plotting — bar/line plots vs axis iterators."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_llm.data_transforms import nested_loop
from audit_llm.plot_configs import (
    GRID_CONFIG,
    LEGEND_CONFIG,
    XLABEL_CONFIG,
    XTICKS_CONFIG,
    YLABEL_CONFIG,
    YTICKS_CONFIG,
)
from audit_llm.plotting.constants import COLOR_DELTA_TP_MAP
from audit_llm.plotting.figure_io import save_fig_and_show
from audit_llm.xp_tools.config_validation import (
    get_iter_idx_from_calculations_config,
    is_pattern_strictly_in_string,
)
from audit_llm.xp_tools.token_pair_grouping import get_token_pairs_of_group
from audit_llm.xp_tools.label_formatting import (
    TRUNC_COL_NAMES,
    calculation_item_namer,
    format_tp_group_name_label,
    get_calculation_item_name,
    put_uppercase_first,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_mean_scores_of_single_pipes(
    results: Dict[str, Any],
    axis_iterator_idx: str,
    fig_iterator: Any,
    pivot_col_idx: Any,
    pivot_col_name: str,
    xp_config: Dict,
    figure_config: Dict,
    calculations_iter_lists: Dict,
    metric: str,
    grouping: bool,
    pipe_summary_mode: bool = False,
) -> Tuple[pd.DataFrame, Any]:
    """Collect per-dataset best-classifier scores into a DataFrame.

    Note: currently assumes a single classifier throughout the process.
    """
    calculations_config = xp_config["calculations"]
    rows: list[dict] = []
    dataset_iter_idx = get_iter_idx_from_calculations_config(iterator_name="token_pairs", xp_config=xp_config)
    axis_iterator_name = xp_config["calculations"][axis_iterator_idx]

    def get_clf_scores_under_loop(calculation_item: dict) -> None:
        calculation_item_name = get_calculation_item_name(calculations_config, calculation_item)
        if fig_iterator == "no_repeat_for_each":
            add_calculation_item = True
            pivot_col_name_item = "no_repeat_for_each"
        else:
            fig_iterator_mention = calculation_item_namer(calculations_config, pivot_col_idx, fig_iterator)
            add_calculation_item = is_pattern_strictly_in_string(calculation_item_name, fig_iterator_mention)
            pivot_col_name_item = calculation_item[
                get_iter_idx_from_calculations_config(iterator_name=pivot_col_name, xp_config=xp_config)
            ]

        if add_calculation_item:
            if pipe_summary_mode:
                summary = results[calculation_item_name]
                assert len(summary) == 1, "Currently only one clf supported throughout the process."
                best_clf = next(iter(summary.keys()))
                score = float(summary[best_clf][f"{metric}_mean"])
            else:
                score = results[calculation_item_name]
                if isinstance(score, dict) and metric in score:
                    score = float(score[metric])
                best_clf = "no_clf"

            rows.append(
                {
                    axis_iterator_name: calculation_item[axis_iterator_idx],
                    "TokenPair": calculation_item[dataset_iter_idx],
                    pivot_col_name: pivot_col_name_item,
                    "Classifier": best_clf,
                    "Score": score,
                }
            )

    nested_loop(calculations_iter_lists, get_clf_scores_under_loop)

    df = pd.DataFrame(rows)

    # aggregating pivot_col_name items so that there is unique combinations
    aggregation = figure_config.get("aggregation", "mean")

    if grouping:
        assert figure_config["group_by"] == "token_pair_groups"
        cols_to_keep = [axis_iterator_name, "TokenPair", pivot_col_name, "Classifier"]
    else:
        cols_to_keep = [axis_iterator_name, pivot_col_name, "Classifier"]

    df = df.groupby(cols_to_keep, as_index=False).agg({"Score": aggregation})

    clfs = df["Classifier"].unique().tolist()
    if len(clfs) > 1:
        raise NotImplementedError(
            "Careful, currently it takes best clf per score but we want to use same clf overall. "
            "So as long as only one clf is engaged in the whole process, it is fine. "
            "But it should be corrected if ever want to run several classifiers."
        )

    return df, clfs[0]


def _make_pivots(
    df: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    index_values: List[Any],
) -> pd.DataFrame:
    """Pivot *df* and reindex to *index_values*."""
    scores = df.pivot(index=index, columns=columns, values=values)
    scores = scores.reindex(sorted(index_values))
    return scores


def _compute_aucs(scores_pivot: pd.DataFrame) -> Dict[str, float]:
    """Compute area-under-curve for each pivot column via trapezoidal rule."""
    aucs: Dict[str, float] = {}
    for pivot_col in scores_pivot.columns:
        series = scores_pivot[pivot_col].dropna()
        if len(series) >= 2:
            x = series.index.astype(float)
            y = series.values
            _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
            aucs[pivot_col] = float(_trapz(y, x))
        else:
            aucs[pivot_col] = float("nan")
    return aucs


def _select_pivot_cols_for_plot(
    pivot_cols: List[str],
    aucs: Dict[str, float],
    scores_pivot: pd.DataFrame,
    big_data_threshold: int = 10,
    topk: int = 5,
) -> List[str]:
    """Select top-k and bottom-k pivot columns by AUC for large data."""
    if len(pivot_cols) <= big_data_threshold:
        return list(pivot_cols)
    valid = {k: v for k, v in aucs.items() if not np.isnan(v)}
    if valid:
        sorted_items = sorted(valid.items(), key=lambda kv: kv[1])
        bottom = [k for k, _ in sorted_items[:topk]]
        top = [k for k, _ in sorted_items[-topk:]]
        selected = list(dict.fromkeys(bottom + top))
        return selected
    coverage = scores_pivot.count().sort_values(ascending=False)
    return list(coverage.index[: (topk * 2)])


def _plot_with_errorbars(
    ax: plt.Axes,
    data_dict: Dict,
    label: str,
    figure_config: Dict,
    fmt: str = "-o",
    linewidth: float = 2,
    alpha: float = 0.9,
    capsize: int = 3,
    capthick: float = 1,
    offset: float = 0.0,
    width: float = 0.8,
) -> None:
    """Generic plotter with error bars supporting both line and bar plots."""
    plot_type: str = figure_config.get("type", "lineplot")
    x_mode: str = figure_config.get("x_mode", "numeric")
    error_bar: str = figure_config.get("error_bar", "std")
    aggregation: str = figure_config.get("aggregation", "mean")

    data = data_dict["data"]
    color = data_dict.get("color", None)

    y_values = data.astype(float)
    x_values = y_values.index.values

    # --- Handle x_mode ---
    if x_mode == "categorical":
        x_ticks = np.arange(len(x_values))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_values)
    elif x_mode == "numeric":
        x_ticks = x_values
    else:
        raise ValueError(f"Unknown x_mode: {x_mode}. Use 'numeric' or 'categorical'.")

    # --- Aggregation ---
    if aggregation == "mean":
        y_aggregation = y_values.mean(axis=1)
    elif aggregation == "median":
        y_aggregation = y_values.median(axis=1)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}. Use 'mean' or 'median'.")

    # --- Error computation ---
    if y_values.shape[1] == 1:
        y_err = None
    elif error_bar == "std":
        y_err = y_values.std(axis=1)
    elif error_bar == "sem":
        y_err = y_values.sem(axis=1)
    elif error_bar == "iqr":
        q75 = y_values.quantile(0.75, axis=1)
        q25 = y_values.quantile(0.25, axis=1)
        y_err = (q75 - q25) / 2
    else:
        raise ValueError(f"Unknown error_bar: {error_bar}. Use 'std', 'sem', or 'iqr'.")

    # --- Plot depending on type ---
    if plot_type == "barplot":
        x_positions = np.array(x_ticks) + offset
        ax.bar(
            x_positions,
            y_aggregation,
            width=width,
            yerr=y_err,
            label=label,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            error_kw=dict(capsize=capsize, capthick=capthick, lw=linewidth),
        )

    elif plot_type == "lineplot":
        if y_err is None:
            ax.plot(
                x_ticks,
                y_aggregation,
                fmt,
                label=label,
                linewidth=linewidth,
                alpha=alpha,
                color=color,
            )
        else:
            ax.errorbar(
                x=x_ticks,
                y=y_aggregation,
                yerr=y_err,
                label=label,
                fmt=fmt,
                capsize=capsize,
                capthick=capthick,
                linewidth=linewidth,
                alpha=alpha,
                color=color,
            )
    else:
        raise ValueError(f"Unknown plot type: {plot_type}. Use 'lineplot' or 'barplot'.")


# ---------------------------------------------------------------------------
# Multi-figure grid
# ---------------------------------------------------------------------------

def _plot_multifigure_perf_vs_iter(
    data_dict: dict,
    axis_iterator_values: tuple,
    repeat_for_each: list,
    fig_iterator_name: str,
    calculations_iter_lists: dict,
    new_model_idx: dict,
    xp_config: dict,
    figure_config: dict,
    save_path: str | Path,
    tp_group_names: list[str],
    metric: str,
    pipe_summary_mode: bool,
    ncols: int = 1,
) -> None:
    """Create a grid of subplots for all repeat_for_each values."""
    n_plots = len(repeat_for_each)
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5 * nrows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    clf_name = None

    for idx, fig_iterator in enumerate(repeat_for_each):
        ax = axes[idx]

        clf_name = plot_performance_vs_axis_iterator(
            results=data_dict,
            axis_iterator_values=axis_iterator_values,
            fig_iterator=fig_iterator,
            calculations_iter_lists=calculations_iter_lists,
            new_model_idx=new_model_idx,
            xp_config=xp_config,
            figure_config=figure_config,
            save_path=None,
            pivot_col_name=fig_iterator_name,
            tp_group_names=tp_group_names,
            metric=metric,
            pipe_summary_mode=pipe_summary_mode,
            ax=ax,
            title=f"{fig_iterator}",
        )

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")

    fig.tight_layout()

    # Save multifigure
    axis_iterator_idx = axis_iterator_values[0]
    axis_iterator_name_str = xp_config["calculations"][axis_iterator_idx]

    if figure_config["group_by"] == "token_pair_groups":
        grouping_mention = "_grouped"
    else:
        grouping_mention = ""

    save_fig_and_show(
        Path(save_path) / (clf_name or "clf"),
        fig_name=f"{metric}_vs_{TRUNC_COL_NAMES[axis_iterator_name_str]}{grouping_mention}_multifig.pdf",
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

_DEFAULT_TP_GROUP_NAMES: list[str] = ["0-1", "FLiPS"]


def generate_personalized_figures(
    xp_config: dict,
    calculations_config: dict,
    calculations_iter_lists: dict,
    new_model_idx: dict,
    save_fig_path: str | Path,
    data_dict: dict,
    pipe_summary_mode: bool = False,
    tp_group_names: list[str] | None = None,
) -> None:
    """Generate personalized figures based on experiment configuration.

    Parameters
    ----------
    xp_config : dict
        Experiment configuration, containing ``'figures'`` section.
    calculations_config : dict
        Configuration mapping for iterators.
    calculations_iter_lists : dict
        Lists of iterator values for repeated plotting.
    new_model_idx : dict
        Model index remapping.
    save_fig_path : str or Path
        Base path where figures should be saved.
    data_dict : dict
        Performance or pipeline data.
    pipe_summary_mode : bool
        Whether the pipeline mode is active.
    tp_group_names : list[str] or None
        Token pair group names. Defaults to ``["0-1", "FLiPS"]``.
    """
    if tp_group_names is None:
        tp_group_names = list(_DEFAULT_TP_GROUP_NAMES)

    for figure_idx, figure_config in xp_config.get("figures", {}).items():
        logger.info("Plotting figure(s): %s", figure_idx)
        personalized_fig_save_path = Path(save_fig_path) / str(figure_idx)

        # Determine if figure should repeat for each iterator (like token_pairs)
        if figure_config.get("repeat_for_each", "none") != "none":
            fig_iterator_name = calculations_config[figure_config["repeat_for_each"]]
            repeat_for_each = calculations_iter_lists[figure_config["repeat_for_each"]]
        else:
            fig_iterator_name = "no_repeat_for_each"
            repeat_for_each = [fig_iterator_name]

        # Determine which axis corresponds to iterated values
        if figure_config["y-axis"] == "metric":
            axis_iterator = "x-axis"
        else:
            assert figure_config["x-axis"] == "metric"
            axis_iterator = "y-axis"

        axis_iterator_values = (figure_config[axis_iterator], calculations_iter_lists[figure_config[axis_iterator]])

        # --- Layout control ---
        layout = figure_config.get("layout", "inferred")
        grid_columns = figure_config.get("grid_columns", 1)

        if layout == "inferred":
            layout = "both" if len(repeat_for_each) > 1 else "individual"

        # --- Generate multifigure grid ---
        if layout in ("grid", "both"):
            logger.info("Generating multifigure (%d cols) for %s", grid_columns, fig_iterator_name)
            multifig_path = Path(personalized_fig_save_path) / "multifigure"
            multifig_path.mkdir(exist_ok=True, parents=True)

            for metric in figure_config["metrics"]:
                _plot_multifigure_perf_vs_iter(
                    data_dict=data_dict,
                    axis_iterator_values=axis_iterator_values,
                    repeat_for_each=repeat_for_each,
                    fig_iterator_name=fig_iterator_name,
                    calculations_iter_lists=calculations_iter_lists,
                    new_model_idx=new_model_idx,
                    xp_config=xp_config,
                    figure_config=figure_config,
                    save_path=multifig_path,
                    tp_group_names=tp_group_names,
                    metric=metric,
                    pipe_summary_mode=pipe_summary_mode,
                    ncols=grid_columns,
                )

        # --- Generate individual figures ---
        if layout in ("individual", "both"):
            for fig_iterator in repeat_for_each:
                logger.debug("Processing %s: %s", fig_iterator_name, fig_iterator)
                fig_iterator_path = (
                    Path(personalized_fig_save_path) / f"{TRUNC_COL_NAMES[fig_iterator_name]}_{fig_iterator}"
                )
                fig_iterator_path.mkdir(exist_ok=True, parents=True)

                for metric in figure_config["metrics"]:
                    plot_performance_vs_axis_iterator(
                        data_dict,
                        axis_iterator_values,
                        fig_iterator,
                        calculations_iter_lists,
                        new_model_idx,
                        xp_config,
                        figure_config,
                        save_path=fig_iterator_path,
                        pivot_col_name=fig_iterator_name,
                        tp_group_names=tp_group_names,
                        metric=metric,
                        pipe_summary_mode=pipe_summary_mode,
                    )


def plot_performance_vs_axis_iterator(
    results: Dict[str, Any],
    axis_iterator_values: Tuple[Any, List[Any]],
    fig_iterator: Any,
    calculations_iter_lists: dict,
    new_model_idx: Dict[str, int],
    xp_config: Dict,
    figure_config: Dict,
    save_path: str | Path | None,
    pivot_col_name: str,
    tp_group_names: List[str],
    metric: str = "f1",
    pipe_summary_mode: bool = False,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> str:
    """Plot performance vs an axis iterator for dataset groups.

    Parameters
    ----------
    results : dict
        Performance results dictionary.
    axis_iterator_values : tuple
        ``(axis_iterator_idx, axis_iterator_list)``.
    fig_iterator : Any
        Current iterator value for repeating.
    calculations_iter_lists : dict
        Dict of calculation iter lists.
    new_model_idx : dict
        Model index remapping.
    xp_config : dict
        Experiment configuration dictionary.
    figure_config : dict
        Figure-specific configuration.
    save_path : str, Path, or None
        Where to save the figure (``None`` to skip saving).
    pivot_col_name : str
        Name of the pivot column.
    tp_group_names : list[str]
        List of token pair group names to plot.
    metric : str
        Metric to plot.
    pipe_summary_mode : bool
        Whether using pipeline mode.
    ax : matplotlib Axes or None
        Optional axis for subplots.
    title : str or None
        Optional subplot title.

    Returns
    -------
    str
        Classifier name used.
    """
    if figure_config["group_by"] == "none":
        grouping = False
    elif figure_config["group_by"] == "token_pair_groups":
        assert figure_config["repeat_for_each"] != "token_pairs"
        grouping = True
    else:
        raise NotImplementedError("TODO (not urgent, allow other grouping)")

    # Step 1: collect dataset-level best-classifier scores & stds.
    axis_iterator_idx, _ = axis_iterator_values
    axis_iterator_name = xp_config["calculations"][axis_iterator_idx]
    pivot_col_idx = figure_config["repeat_for_each"]
    df, clf = _collect_mean_scores_of_single_pipes(
        results,
        axis_iterator_idx,
        fig_iterator,
        pivot_col_idx,
        pivot_col_name,
        xp_config,
        figure_config,
        calculations_iter_lists,
        metric,
        grouping,
        pipe_summary_mode=pipe_summary_mode,
    )

    # Sort axis_iterator_list for consistent plotting
    axis_iterator_list = list(df[axis_iterator_name].unique())
    axis_iterator_list.sort()

    # Step 2: Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        standalone_fig = True
    else:
        standalone_fig = False

    curves_to_plot: dict[str, dict] = {}

    if grouping:
        # Grouping values per tp_group and plotting
        if pivot_col_name != "token_pairs":
            nb_of_cols_of_current_pivot = len(df[pivot_col_name].unique().tolist())
            assert nb_of_cols_of_current_pivot == 1
            pivot_col_name = "TokenPair"

        # Drop exact duplicates
        before = len(df)
        df = df.drop_duplicates(subset=[axis_iterator_name, pivot_col_name, "Score"])
        after = len(df)
        if standalone_fig:
            logger.debug(
                "Removed %d exact duplicate rows (same axis_iterator_name, %s, and Score).",
                before - after,
                pivot_col_name,
            )

        # Check remaining duplicates
        dupes = df[df.duplicated(subset=[axis_iterator_name, pivot_col_name], keep=False)]
        if not dupes.empty and standalone_fig:
            logger.debug("Still found %d duplicates for (axis_iterator_name, %s) combinations:", len(dupes), pivot_col_name)
            logger.debug("%s", dupes.sort_values([axis_iterator_name, pivot_col_name]).to_string())
        elif standalone_fig:
            logger.debug("No remaining duplicates for (axis_iterator_name, pivot_col_name).")

        scores_pivot = _make_pivots(
            df, index=axis_iterator_name, columns=pivot_col_name, values="Score", index_values=axis_iterator_list
        )

        datasets = calculations_iter_lists[get_iter_idx_from_calculations_config("token_pairs", xp_config)]
        tp_groups = {
            tp_group_name: get_token_pairs_of_group(tp_group_name, token_pairs=datasets) for tp_group_name in tp_group_names
        }

        for tp_group_name, tp_group in tp_groups.items():
            if not tp_group:
                raise ValueError(
                    f"No datasets found for group '{tp_group_name}'. "
                    f"Check dataset names and group definitions. {datasets =}"
                )

            curves_to_plot[format_tp_group_name_label(tp_group_name)] = {
                "data": scores_pivot[tp_group],
                "color": COLOR_DELTA_TP_MAP[tp_group_name],
            }
    else:
        scores_pivot = _make_pivots(
            df, index=axis_iterator_name, columns=pivot_col_name, values="Score", index_values=axis_iterator_list
        )

        # AUC-based selection when big_data
        topk = 5
        nb_of_cols = len(list(scores_pivot.columns))
        if nb_of_cols > 2 * topk:
            aucs = _compute_aucs(scores_pivot)
            selected_pivot_cols_for_plot = _select_pivot_cols_for_plot(
                list(scores_pivot.columns), aucs, scores_pivot, topk=topk
            )
        else:
            selected_pivot_cols_for_plot = list(scores_pivot.columns)

        for pivot_col in selected_pivot_cols_for_plot:
            curves_to_plot[pivot_col] = {"data": scores_pivot[[pivot_col]]}

    # Compute bar width and offsets if barplot
    plot_type = figure_config.get("type", "lineplot")
    n_curves = len(curves_to_plot)
    bar_width = 0.8 / n_curves if plot_type == "barplot" else None

    for i, (curve_label, curve_data_dict) in enumerate(curves_to_plot.items()):
        offset = (i - (n_curves - 1) / 2) * bar_width if plot_type == "barplot" else 0
        _plot_with_errorbars(
            ax,
            curve_data_dict,
            label=curve_label,
            figure_config=figure_config,
            offset=offset,
            width=bar_width,
        )

    # Customize the plot
    axis_iterator_name = xp_config["calculations"][axis_iterator_idx]
    axis_iterator_name_to_pretty_name = {
        "models": "LLM",
    }

    pretty_name = axis_iterator_name_to_pretty_name.get(axis_iterator_name, axis_iterator_name)
    ax.set_xlabel(pretty_name, **XLABEL_CONFIG)
    ax.set_ylabel(f"{put_uppercase_first(metric)}", **YLABEL_CONFIG)

    # Set title if provided (for subplots)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    ax.tick_params(axis="x", **XTICKS_CONFIG, rotation=45)
    # Right-align x-tick labels to end at the tick
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.tick_params(axis="y", **YTICKS_CONFIG)
    # Get all handles and labels currently attached to the axes
    ax.legend(**LEGEND_CONFIG)

    handles, labels = ax.get_legend_handles_labels()
    # Only create the legend if there are 2 or more items
    if len(labels) > 1:
        ax.legend(handles, labels, **LEGEND_CONFIG)
    else:
        legend = ax.get_legend()
        if legend:
            legend.remove()

    ax.grid(**GRID_CONFIG)

    # Only tight_layout and save if standalone figure
    if standalone_fig:
        fig.tight_layout()

        if save_path is not None:
            axis_iterator_name = xp_config["calculations"][axis_iterator_idx]
            grouping_mention = "_grouped" if grouping else ""

            save_fig_and_show(
                Path(save_path) / clf,
                fig_name=f"{metric}_vs_{TRUNC_COL_NAMES[axis_iterator_name]}{grouping_mention}.pdf",
            )

    return clf
