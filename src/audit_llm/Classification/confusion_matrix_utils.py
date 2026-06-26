"""
Confusion matrix grouping, reordering, and visualization utilities.

Provides functions for creating grouped confusion matrices (reordered by variation),
plotting with bimodal colormaps, and averaging multiple confusion matrices.
"""

import logging
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)
from matplotlib import colors

from audit_llm.xp_tools.model_filtering import truncate_model_name
from audit_llm.xp_tools import assemble_iterator_name_and_value


def create_grouped_cm(
    cm: np.ndarray, new_models_idx: Dict[int, str], model_variation_dict: Dict, fixed_variation_name=None
) -> tuple:
    """
    Reorder confusion matrix rows/columns to group by variation items.
    Returns: (reordered_cm, group_labels) where group_labels is [(label, count), ...]
    """

    # Determine which variation to use for grouping
    should_average = False
    if len(model_variation_dict) == 1:
        grouping_variation_name = list(model_variation_dict.keys())[0]
    # Finding the key for which there is only a single value
    elif len(model_variation_dict) == 2:
        single_value_variations = [k for k, v in model_variation_dict.items() if len(v) == 1]
        if len(single_value_variations) == 1:
            # Taking the key that is not single-valued
            fixed_variation_name = [k for k in model_variation_dict.keys() if k not in single_value_variations]
            assert len(fixed_variation_name) == 1
            grouping_variation_name = fixed_variation_name[0]
            should_average = True  # Multiple variations case requires averaging
        else:
            raise ValueError(f"When len(model_variations)==2, one variation must have a single value to fix.")
    elif fixed_variation_name is not None:
        if fixed_variation_name not in model_variation_dict:
            raise ValueError(f"fixed_variation_name '{fixed_variation_name}' not found in model_variations")
        grouping_variation_name = fixed_variation_name
        should_average = True  # Multiple variations case requires averaging
    else:
        raise ValueError(f"fixed_variation_name must be specified when len(model_variations) > 1")

    grouping_items = model_variation_dict[grouping_variation_name]

    # Group model indices by the grouping variation item
    variation_groups = {item: [] for item in grouping_items}

    for idx, model_name in new_models_idx.items():
        # Extract variation from model_name
        if "_" not in model_name:
            continue  # this will avoid abliterated models.

        variation_str = "_".join(model_name.split("_")[1:])

        # Match to variation items
        for item in grouping_items:
            expected_var = assemble_iterator_name_and_value(grouping_variation_name, item)
            if expected_var in variation_str:
                variation_groups[item].append(idx)
                break

    # Create new ordering: all indices for item1, then item2, etc.
    new_order = []
    group_labels = []

    for item in grouping_items:
        indices = sorted(variation_groups[item])
        new_order.extend(indices)
        if indices:
            label = assemble_iterator_name_and_value(grouping_variation_name, item)
            group_labels.append((label, len(indices)))

    # Reorder the confusion matrix
    grouped_cm = cm[np.ix_(new_order, new_order)]

    # If multiple variations, average within each group
    if should_average:
        averaged_cm = np.zeros((len(grouping_items), len(grouping_items)))
        start_idx = 0

        for i, (item_i, count_i) in enumerate(group_labels):
            end_idx_i = start_idx + count_i
            start_idx_j = 0

            for j, (item_j, count_j) in enumerate(group_labels):
                end_idx_j = start_idx_j + count_j

                # Average the block
                block = grouped_cm[start_idx:end_idx_i, start_idx_j:end_idx_j]
                averaged_cm[i, j] = np.mean(block)

                start_idx_j = end_idx_j

            start_idx = end_idx_i

        # Update group_labels to reflect averaged structure (count = 1 per group)
        group_labels_averaged = [(label, 1) for label, _ in group_labels]

        return averaged_cm, group_labels_averaged

    return grouped_cm, group_labels


def plot_grouped_cm(
    cm: np.ndarray,
    save_path: Path,
    filename: str,
    title: str = None,
    models_idx: Dict[int, str] = None,
    group_labels: list = None,
    xlabel_config: dict = None,
    ylabel_config: dict = None,
):
    """
    Plot confusion matrix with bimodal colormap (similar to plot_confusion_matrices_on_tr_size_dict).
    Works both with and without group_labels.

    Args:
        cm: Confusion matrix (will be normalized by row)
        save_path: Directory to save the figure
        filename: Output filename
        title: Plot title
        models_idx: Dict mapping index to model name (for y-tick labels when no grouping)
        group_labels: List of (label, count) tuples for group visualization (optional)
        xlabel_config: Dict of kwargs for xlabel styling
        ylabel_config: Dict of kwargs for ylabel styling
    """
    fig, ax = plt.subplots(figsize=(12.3, 8))

    # Normalize by row (true labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = cm / np.where(row_sums != 0, row_sums, 1)
    cm_normalized = np.nan_to_num(cm_normalized)

    # Compute diagonal vs off-diagonal statistics
    n = cm_normalized.shape[0]
    diag_mask = np.eye(n, dtype=bool)
    diag_vals = cm_normalized[diag_mask]
    off_vals = cm_normalized[~diag_mask]

    # Choose center between the cluster means
    vmin = float(np.nanmin(cm_normalized))
    vmax = float(np.nanmax(cm_normalized))
    proposed_center = float((0.15 * np.nanmean(diag_vals) + np.nanmean(off_vals)) / 2.0)

    if not (vmin < proposed_center < vmax):
        # Fallback: mid-point of range
        center = 0.5 * (vmin + 0.1 * vmax)
    else:
        center = proposed_center

    # Create bimodal colormap
    cmap = colors.LinearSegmentedColormap.from_list("bimodal", ["white", "#1f77b4", "#d62728"])

    # TwoSlopeNorm for diverging colormap
    norm = colors.TwoSlopeNorm(vmin=0.0, vcenter=center, vmax=1.0)

    # Determine tick labels based on whether we have grouping or not
    if group_labels:
        # Grouped case: hide individual ticks
        xticklabels = False
        yticklabels = False
    else:
        # Non-grouped case: show model names if available
        if models_idx:
            ytickslabels = [f"{i}{' ' if i <= 9 else ''}: {truncate_model_name(models_idx[i], k=24)}" for i in range(n)]
            max_len = max(len(label) for label in ytickslabels)
            yticklabels = [label.ljust(max_len) for label in ytickslabels]
        else:
            yticklabels = [str(i) for i in range(n)]

        xticklabels = [str(i) for i in range(n)]

    # Plot heatmap
    sns.heatmap(
        cm_normalized,
        annot=False,
        fmt=".2f",
        cmap=cmap,
        norm=norm,
        cbar=True,
        ax=ax,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        linewidths=0.5,
        linecolor="lightgray",
        square=True,
    )

    cbar = ax.collections[0].colorbar

    # Set custom colorbar ticks
    valid_ticks = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    valid_ticks = [t for t in valid_ticks if vmin <= t <= vmax]

    if len(valid_ticks) < 2:
        valid_ticks = list(np.linspace(vmin, vmax, 5))

    cbar.set_ticks(valid_ticks)
    cbar.set_ticklabels([f"{t:.3f}" for t in valid_ticks])

    # Add group labels and separators if provided
    if group_labels:
        group_positions = []
        current_pos = 0

        for label, count in group_labels:
            group_positions.append((current_pos + count / 2, label))
            current_pos += count

            # Add separator lines between groups
            if current_pos < cm_normalized.shape[0]:
                ax.axhline(current_pos, color="red", linewidth=2, alpha=0.7)
                ax.axvline(current_pos, color="red", linewidth=2, alpha=0.7)

        # Add text labels at group centers
        for pos, label in group_positions:
            ax.text(-5, pos, label, ha="right", va="center", fontsize=9, fontweight="bold")
            ax.text(pos, -5, label, ha="center", va="bottom", fontsize=9, fontweight="bold", rotation=45)
    else:
        # Force monospace for y-tick labels in non-grouped case
        if models_idx:
            for tick in ax.get_yticklabels():
                tick.set_fontname("monospace")

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # Set labels with optional styling
    xlabel_kwargs = xlabel_config if xlabel_config else {}
    ylabel_kwargs = ylabel_config if ylabel_config else {}

    ax.set_xlabel("Prediction", **xlabel_kwargs)
    ax.set_ylabel("True Label", **ylabel_kwargs)

    if title:
        ax.set_title(title)

    plt.tight_layout()
    out_path = save_path / filename
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved CM: %s", filename)


def _get_mean_confusion_matrix(confusion_matrices_dict: Dict[str, List[np.ndarray]]) -> tuple[np.ndarray, str]:
    if len(confusion_matrices_dict) > 1:
        raise NotImplementedError("several clfs found in cms")

    cms = next(iter(confusion_matrices_dict.values()))
    best_clf = next(iter(confusion_matrices_dict.keys()))
    return np.mean(cms, axis=0), best_clf


def plot_confusion_matrices(
    xp_config,
    confusion_matrices_dict: Dict[str, Dict[str, List[np.ndarray]]],
    new_models_idx: Dict[int, str],
    save_path: Path,
    metric: str = "accuracy",
    plot_averaged: bool = True,
    plot_per_token_pair: bool = False,
    fixed_variation_name: str = "temperature",
):
    """
    Plots and saves confusion matrices.
    - Saves ONE file per token pair (no giant subplot grid).
    - Saves ONE averaged file.
    - annot=False is enforced to prevent hanging on large (70x70) matrices.

    Args:
    - new_models_idx: Dict{index, model_name} with model_name can be <base_model_name>_<variation> if xp_config['model_variations'] is not None
    - xp_config['model_variations'] : List[Dict] model_variations: [
                                                                {'temperature': [0.1, 0.5, 1.0], 'system_prompt_idx': [-1]},
                                                                {'temperature': [1.0], 'system_prompt_idx': [0, 9]},
                                                                ]
    - fixed_variation_name: str, required when len(model_variations) > 1. Specifies which variation to group by.
      Other variations will be averaged over.
    - plot_per_token_pair: bool, default False. If True, saves individual plots per token pair.
    - plot_averaged: bool, default True. If True, saves averaged confusion matrix.

    to have access to <variation>: model_name.split('_')[1:]
    <variation> is a string with '_' between variation_item.
    Example:
        xp_config['model_variations'] = {temperature: [0.1,0.2], system_prompt_idx: [1,2,3]}
        so model_name could be 'llama3_temp-0.2_sp-3' (base_model_name == llama3 and temp-0.2_sp-3 corresponds to variation)

        To obtain temp-0.2 from the example, use assemble_iterator_name_and_value(iterator_name='temperature', iterator_value=0.2)

    For a first exercise, set the case when len(xp_config['model_variations'])==1 i.e. only one variation_name (like temperature).
    The thing is that confusion matrices are indexed on this new_models_idx,
    and the goal is to regroup index by this variation name, so that on heatmap, first values are for variation_item1 of variation_name etc.)
    Make a label <variation_item1> to show clearly blocks.
    """
    all_cms_list = []  # Store for averaging later

    for i, (token_pair, confusion_matrices) in enumerate(confusion_matrices_dict.items(), 1):
        try:
            mean_cm, best_clf = _get_mean_confusion_matrix(confusion_matrices)
            all_cms_list.append(mean_cm)

            if plot_per_token_pair:
                # Sanitize filename (remove characters that might break paths)
                safe_name = token_pair.replace("<", "").replace(">", "").replace("/", "_")

                plot_grouped_cm(
                    cm=mean_cm,
                    save_path=save_path,
                    filename=f"CM_{safe_name}.pdf",
                    title=f"Confusion Matrix: {token_pair}\n(Best: {best_clf}, Metric: {metric})",
                )

                # Save grouped version if variations exist
                if "model_variations" in xp_config and xp_config["model_variations"]:
                    for k, model_variation_dict in enumerate(xp_config["model_variations"]):
                        grouped_cm, group_labels = create_grouped_cm(mean_cm, new_models_idx, model_variation_dict)
                        logger.debug("group_labels = %s", group_labels)
                        if group_labels:
                            plot_grouped_cm(
                                cm=grouped_cm,
                                save_path=save_path,
                                filename=f"CM_{safe_name}_grouped_{k}.pdf",
                                title=f"Grouped Confusion Matrix: {token_pair}\n(Best: {best_clf}, Metric: {metric})",
                                group_labels=group_labels,
                            )

        except Exception as e:
            logger.warning("Skipping %s due to error: %s", token_pair, e)

    # --- Plot Averaged CM ---
    if plot_averaged and all_cms_list:
        overall_mean_cm = np.mean(all_cms_list, axis=0)

        plot_grouped_cm(
            cm=overall_mean_cm,
            save_path=save_path,
            filename="CM_Average_All.pdf",
            title=f"Averaged Confusion Matrix ({len(all_cms_list)} pairs)\n(Best by {metric})",
            models_idx=new_models_idx,
        )

        # Save grouped average if variations exist
        if "model_variations" in xp_config and xp_config["model_variations"]:
            for k, model_variation_dict in enumerate(xp_config["model_variations"]):
                grouped_cm, group_labels = create_grouped_cm(
                    overall_mean_cm, new_models_idx, model_variation_dict
                )
                if group_labels:
                    plot_grouped_cm(
                        cm=grouped_cm,
                        save_path=save_path,
                        filename=f"CM_Average_All_grouped_{k}.pdf",
                        title=f"Grouped Averaged Confusion Matrix ({len(all_cms_list)} pairs)\n(Best by {metric})",
                        group_labels=group_labels,
                    )
