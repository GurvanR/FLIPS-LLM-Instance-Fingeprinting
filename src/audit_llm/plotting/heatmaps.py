"""Heatmap visualization for accuracy matrices."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

from audit_llm.plot_configs import get_mpl_configs
from audit_llm.plotting.constants import SHORTLIST_OF_LLMS
from audit_llm.plotting.label_formatting import format_label_multiline, format_value
from audit_llm.plotting.micro_pr_curves import plot_micro_pr_curve_on_ax
from audit_llm.xp_tools.model_filtering import truncate_model_name


def make_accuracy_heatmap(
    stats: dict,
    figsize: tuple[int, int] = (12, 10),
    cmap: str = "RdYlGn",
    vmin: float = 0,
    vmax: float = 1,
    title: str = "",
    truncate_orig_names: bool = True,
    shortlist: bool = False,
    unseen_pr: dict | None = None,
    embedded_pr_curve: dict | None = None,
) -> tuple[Any, Any]:
    """Generate a heatmap visualization with formatted, two-line column headers."""
    if not stats:
        return None, None

    # Filter out "Unseen" from display if present
    has_unseen = stats["orig_labels"] and "Unseen" in stats["orig_labels"]
    orig_labels_to_display = [o for o in stats["orig_labels"] if o != "Unseen"]
    var_names_display = [v for v in stats["var_names"] if v != "Unseen"]

    fig_confg = get_mpl_configs(multiplier=1.5, col_type="double_col")
    n_total_cols = len(var_names_display) + 1  # +1 for LLM Average
    if shortlist:
        # Filtering
        figsize = (max(10, n_total_cols * 0.7) * 1.05, 4.5 * 1.05)
        orig_labels_to_display = [o for o in orig_labels_to_display if o in SHORTLIST_OF_LLMS]
    else:
        figsize = (max(12, n_total_cols * 0.8) * 1.05, 10 * 1.05)

    # Split indices for the quantized data block (bottom-right). Quantized variants
    # are columns, not rows; quant columns sit on the RIGHT and rows that hold data
    # in those columns sit at the BOTTOM. The blank rectangle is therefore the
    # TOP-RIGHT (non-quant-capable rows x quant columns).
    quant_var_set = set(stats.get("quant_vars", []))
    quant_capable_set = set(stats.get("quant_capable_origs", []))
    n_quant_vars = sum(1 for v in var_names_display if v in quant_var_set)
    n_top_origs = sum(1 for o in orig_labels_to_display if o not in quant_capable_set)
    n_bottom_origs = len(orig_labels_to_display) - n_top_origs
    n_base_vars = len(var_names_display) - n_quant_vars

    # Build data matrix including averages.
    # Column order: [LLM Average (col 0), base vars, quant vars]. The variation
    # columns are therefore offset by +1 in the data matrix.
    n_rows = len(orig_labels_to_display) + 1  # +1 for Variation Average row
    n_cols = len(var_names_display) + 1  # +1 for LLM Average column

    data_matrix = np.zeros((n_rows, n_cols))

    # Fill main matrix cells (variations shifted into columns 1..n_cols-1)
    for i, orig in enumerate(orig_labels_to_display):
        for j, var in enumerate(var_names_display):
            val = stats["matrix"].get((orig, var))
            if val and not np.isnan(val["mean"]):
                data_matrix[i, j + 1] = val["mean"]
            else:
                data_matrix[i, j + 1] = np.nan

        # Row average → LLM Average column (col 0)
        row_avg = stats["row_avgs"].get(orig)
        if row_avg and not np.isnan(row_avg["mean"]):
            data_matrix[i, 0] = row_avg["mean"]
        else:
            data_matrix[i, 0] = np.nan

    # Fill Variation Average row for variations (offset by +1)
    for j, var in enumerate(var_names_display):
        col_avg = stats["col_avgs"].get(var)
        if col_avg and not np.isnan(col_avg["mean"]):
            data_matrix[-1, j + 1] = col_avg["mean"]
        else:
            data_matrix[-1, j + 1] = np.nan

    # Grand average (bottom-left corner: LLM Average × Variation Average)
    if stats["grand_avg"] and not np.isnan(stats["grand_avg"]["mean"]):
        data_matrix[-1, 0] = stats["grand_avg"]["mean"]
    else:
        data_matrix[-1, 0] = np.nan

    # --- Create labels ---
    row_labels = orig_labels_to_display + ["Variation Average"]
    if truncate_orig_names:
        row_labels = [truncate_model_name(label) if label != "Variation Average" else label for label in row_labels]

    # APPLY FORMATTING HERE: Format variable names for display
    formatted_vars = [format_label_multiline(v) for v in var_names_display]
    col_labels = ["LLM Average"] + formatted_vars

    # Create DataFrame for easier handling
    df = pd.DataFrame(data_matrix, index=row_labels, columns=col_labels)

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap with annotations showing 2 decimal precision
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Per-class Recall"},
        linewidths=0.5,
        linecolor="lightgray",
        ax=ax,
        annot_kws={"size": 9},
    )

    # Add black border around grand average cell (bottom-left: LLM Average × Variation Average)
    grand_avg_rect = Rectangle((0, n_rows - 1), 1, 1, fill=False, edgecolor="black", linewidth=3, zorder=10)
    ax.add_patch(grand_avg_rect)

    # Inner black lines at 3/4 of the grand-average linewidth, marking the LLM
    # Average column boundary and the Variation Average row boundary. Each stops
    # at the grand-average cell border so the thicker grand-average contour stays clean.
    ax.plot(
        [1, 1], [0, n_rows - 1],
        color="black", linewidth=2.25, zorder=9, solid_capstyle="butt",
    )
    ax.plot(
        [1, n_cols], [n_rows - 1, n_rows - 1],
        color="black", linewidth=2.25, zorder=9, solid_capstyle="butt",
    )

    # Outer L-shape contour around the data area (everything except the top-right
    # blank "Quantizations" region) at 1/4 of the grand-average linewidth.
    ax.plot(
        [0, 1 + n_base_vars, 1 + n_base_vars, n_cols, n_cols, 0, 0],
        [0, 0, n_top_origs, n_top_origs, n_rows, n_rows, 0],
        color="black", linewidth=0.75, zorder=8, solid_capstyle="butt",
    )

    # Contour around the quantization data block (bottom-right rectangle) at 1/4
    # of the grand-average linewidth.
    if n_quant_vars > 0 and n_bottom_origs > 0:
        ax.plot(
            [1 + n_base_vars, n_cols, n_cols, 1 + n_base_vars, 1 + n_base_vars],
            [n_top_origs, n_top_origs, n_rows, n_rows, n_top_origs],
            color="black", linewidth=0.75, zorder=8, solid_capstyle="butt",
        )

    # Blank out the top-right rectangle (non-quant-capable rows x quant columns):
    # cover seaborn's lightgray gridlines with a white-on-white patch so the missing
    # region reads as empty space. Excludes the LLM Average column (leftmost) and
    # the Variation Average row (bottom), which still hold valid weighted aggregates.
    if n_top_origs > 0 and n_quant_vars > 0:
        blank_rect = Rectangle(
            (1 + n_base_vars, 0),
            n_quant_vars,
            n_top_origs,
            facecolor="white",
            edgecolor="white",
            linewidth=1.0,
            zorder=5,
        )
        ax.add_patch(blank_rect)

        # Label the blank region with "Quantizations" — same style as the axis titles.
        # Anchored near the bottom of the blank; shortlist has fewer rows so the
        # label sits a bit higher relative to the (smaller) blank height.
        label_y_frac = 23 / 24 if shortlist else 39 / 40
        ax.text(
            1 + n_base_vars + n_quant_vars / 2,
            n_top_origs * label_y_frac,
            "Quantizations",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            zorder=7,
        )

        # In closed-set mode (when caller passes the precomputed PR data),
        # embed the micro-PR curve in the upper portion of the blank rectangle,
        # leaving the "Quantizations" label visible at the bottom. Only shown in
        # shortlist mode; the full heatmap keeps the blank empty.
        if embedded_pr_curve is not None and shortlist:
            pad_x = 0.25
            pad_top = 0.25
            pad_above_lbl = n_top_origs * 0.06

            data_x0 = 1 + n_base_vars + pad_x
            data_x1 = n_cols - pad_x
            data_y_top = pad_top
            data_y_bot = n_top_origs * label_y_frac - pad_above_lbl

            # Seaborn heatmap has an inverted y-axis: convert the data-coord
            # rectangle to the axes-fraction expected by ax.inset_axes (origin
            # at bottom-left of the axes).
            ax_x = data_x0 / n_cols
            ax_w = (data_x1 - data_x0) / n_cols
            ax_y = 1 - data_y_bot / n_rows
            ax_h = (data_y_bot - data_y_top) / n_rows

            # Shrink the inset around its center (compounded ~28% total).
            shrink = 0.85 * 0.85
            mid_x = ax_x + ax_w / 2
            mid_y = ax_y + ax_h / 2
            ax_w *= shrink
            ax_h *= shrink
            ax_x = mid_x - ax_w / 2
            ax_y = mid_y - ax_h / 2

            # Translate the inset upward by 5% of its own height.
            ax_y += ax_h * 0.05

            inset = ax.inset_axes([ax_x, ax_y, ax_w, ax_h], zorder=7)
            inset.set_facecolor("white")

            plot_micro_pr_curve_on_ax(
                inset,
                thresholds=embedded_pr_curve["thresholds"],
                mean_prec=embedded_pr_curve["mean_prec"],
                std_prec=embedded_pr_curve["std_prec"],
                mean_rec=embedded_pr_curve["mean_rec"],
                std_rec=embedded_pr_curve["std_rec"],
                title="(Micro-averaged) P/R vs confidence",
                show_legend=True,
                show_annotations=True,
                show_xylabels=True,
                ylabel="Precision / Recall",
                xylabel_pad=1.5,
                fontsize_scale=0.7,
                legend_fontsize_scale=1.4,
                saturation_targets_filter=(0.9999,),
                include_saturation_in_legend=False,
                annot_y=0.60,
            )


    # Bold the last row (Variation Average)
    labels = ax.get_yticklabels()
    labels[-1].set_weight("bold")
    ax.set_yticklabels(
        labels,
        fontsize=9 if not shortlist else 9,
    )

    # Bold the first column (LLM Average)
    labels = ax.get_xticklabels()
    labels[0].set_weight("bold")
    ax.set_xticklabels(
        labels,
        fontsize=9 if not shortlist else 8,
    )

    # Formatting axes labels
    ax.set_xlabel("Variation", fontsize=11 if not shortlist else 11, fontweight="bold")
    ax.set_ylabel("LLM", fontsize=11 if not shortlist else 11, fontweight="bold")

    # --- ROTATION UPDATE ---
    # Rotate labels when many columns to avoid overlap
    rotation = 45 if len(var_names_display) > 10 else 0
    ha = "right" if rotation == 45 else "center"
    plt.xticks(rotation=rotation, ha=ha)
    plt.yticks(rotation=0)

    # Add subtitle with unseen info if applicable
    if has_unseen and "row_avgs" in stats:
        unseen_avg = stats["row_avgs"].get("Unseen")
        if unseen_avg and not np.isnan(unseen_avg["mean"]):
            if unseen_pr is not None:
                prec_str = format_value(unseen_pr["precision"])
                rec_str = format_value(unseen_pr["recall"])
                subtitle = (
                    f"$\\bf{{Unseen\\ Precision/Recall:\\ {prec_str}\\%/{rec_str}\\%}}$"
                    " (10% of samples are unseen models)"
                )
            else:
                unseen_rate = format_value(unseen_avg["mean"])
                subtitle = (
                    f"$\\bf{{Unseen\\ rate:\\ {unseen_rate}\\%}}$"
                    " (10% of samples are unseen models)"
                )

            fig.text(
                0.75,
                0.02,
                subtitle,
                ha="center",
                fontsize=9 if not shortlist else 7,
                style="italic",
            )

            plt.subplots_adjust(bottom=0.15)

    plt.tight_layout()

    return fig, ax
