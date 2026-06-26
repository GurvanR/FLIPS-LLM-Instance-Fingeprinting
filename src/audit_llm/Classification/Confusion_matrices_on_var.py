import logging

from audit_llm.plotting.constants import TINYLIST_OF_LLMS
from audit_llm.xp_tools.token_pair_grouping import get_token_pairs_of_group, get_tp_names_of_group
from audit_llm.Classification.results_tables import _get_tp_names_for_key
from audit_llm.xp_tools.label_formatting import SP_IDX_TO_ICML_SP_IDX
from audit_llm.xp_tools.model_filtering import (
    full_var_model_name_to_original_model_name,
    group_models_idx_by_var_or_orig,
    truncate_model_name,
)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

def create_confusion_matrix_heatmaps(
    summary_results,
    models_idx,
    batch_sizes,
    clf,
    tp_group_name,
    save_path,
    batch_type,
    openset=False,
    target_bs=1,     # Added argument with default 1
    scale='both',    # Changed default to 'both'
    token_pairs=None,
):
    """
    Create three confusion matrix heatmaps for FLiPS data:
    1. Original labels matrix (n_orig × n_orig)
    2. Variations matrix (n_var × n_var)
    3. Full matrix with TINYLIST subset (n_subset × n_subset)
    
    Args:
        summary_results: Summary dict with confusion matrices
        models_idx: Dict mapping model indices to model names
        batch_sizes: List of batch sizes
        clf: Classifier name
        tp_group_name: Dataset group name
        save_path: Path to save figures
        batch_type: 'tp_wise' or 'mix_tp_at_pred'
        openset: Whether to include Unseen class
        target_bs: Batch size to visualize (default: 1)
        scale: 'linear', 'symlog', or 'both' (default: 'both')
    """
    
    if target_bs not in batch_sizes:
        logger.warning(f"Batch size {target_bs} not available, skipping confusion matrix heatmaps")
        return
    
    # Determine mode and dataset names — use actual batch_type key for data access
    group_names, mode = _get_tp_names_for_key(summary_results, batch_type, target_bs, tp_group_name, token_pairs)
    if not group_names:
        logger.warning(f"No datasets for {tp_group_name}/{batch_type}/bs={target_bs}, skipping")
        return

    # Get confusion matrix mean
    batch_size_confusion_matrix_map = summary_results[mode]
    
    # Aggregate confusion matrices across datasets
    all_cms = []
    for tp in group_names:
        cm_mean = batch_size_confusion_matrix_map[target_bs][tp][clf]['confusion_matrix_mean']
        all_cms.append(cm_mean)
    
    # Average across datasets
    avg_cm = np.mean(all_cms, axis=0)
    
    # Determine which scales to loop over
    scales_to_plot = ['linear', 'symlog'] if scale == 'both' else [scale]

    for current_scale in scales_to_plot:
        for normalize in (True, False):
            create_original_labels_confusion_heatmap(
                avg_cm, models_idx, tp_group_name, save_path, openset, current_scale, target_bs,
                normalize=normalize,
            )

            create_variations_confusion_heatmap(
                avg_cm, models_idx, tp_group_name, save_path, openset, current_scale, target_bs,
                normalize=normalize,
            )

            create_full_tinylist_confusion_heatmap(
                avg_cm, models_idx, tp_group_name, save_path, openset, current_scale, target_bs,
                normalize=normalize,
            )


def _get_heatmap_norm_args(scale, normalize=True, vmax=None):
    """Helper to return normalization arguments based on scale type and normalization mode."""
    if normalize:
        if scale == 'symlog':
            return {
                'norm': SymLogNorm(linthresh=0.01, linscale=1, vmin=0, vmax=1),
                'cbar_kws': {
                    'label': 'Proportion (Log scale)',
                    'ticks': [1.000, 0.100, 0.050, 0.025, 0.010, 0.005, 0.001],
                    'format': '%.3f'
                }
            }
        else:
            return {
                'norm': None,
                'vmin': 0,
                'vmax': 1,
                'cbar_kws': {'label': 'Proportion'}
            }
    else:
        if scale == 'symlog':
            return {
                'norm': SymLogNorm(linthresh=1, linscale=1, vmin=0, vmax=vmax),
                'cbar_kws': {
                    'label': 'Count (Log scale)',
                    'format': '%.0f',
                }
            }
        else:
            return {
                'norm': None,
                'vmin': 0,
                'vmax': vmax,
                'cbar_kws': {'label': 'Count'}
            }


def create_original_labels_confusion_heatmap(
    confusion_matrix, models_idx, tp_group_name, save_path, openset, scale='linear', target_bs=1,
    normalize=True,
):
    """
    Create confusion matrix aggregated by original labels (n_orig × n_orig).
    Confusion is averaged over variations.
    """
    # Group by original labels
    orig_groups = group_models_idx_by_var_or_orig(models_idx, group_by="orig")
    orig_labels = sorted(orig_groups.keys())

    # Ensure Unseen is last
    if 'Unseen' in orig_labels:
        orig_labels.remove('Unseen')
        orig_labels.append('Unseen')

    n_orig = len(orig_labels)
    aggregated_cm = np.zeros((n_orig, n_orig))

    # Aggregate confusion matrix by original labels
    for i, orig_true in enumerate(orig_labels):
        for j, orig_pred in enumerate(orig_labels):
            # Get all indices for this original label (all variations)
            true_indices = [idx for idx, _ in orig_groups[orig_true]]
            pred_indices = [idx for idx, _ in orig_groups[orig_pred]]

            # Sum confusions for all variation combinations
            for true_idx in true_indices:
                for pred_idx in pred_indices:
                    aggregated_cm[i, j] += confusion_matrix[true_idx, pred_idx]

    if normalize:
        row_sums = aggregated_cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_to_plot = aggregated_cm / row_sums
    else:
        cm_to_plot = aggregated_cm

    plot_args = _get_heatmap_norm_args(scale, normalize=normalize, vmax=cm_to_plot.max())
    cbar_kws = plot_args.pop('cbar_kws')

    # Create heatmap
    fig, ax = plt.subplots(figsize=(9, 7))

    # Truncate long names for display
    display_labels = [truncate_model_name(label) for label in orig_labels]

    sns.heatmap(
        cm_to_plot,
        annot=False,  # annot=False for original labels
        fmt='.2f' if normalize else '.0f',
        cmap='YlOrRd',
        xticklabels=display_labels,
        yticklabels=display_labels,
        ax=ax,
        cbar_kws=cbar_kws,
        linewidths=0.5,
        linecolor='gray',
        **plot_args
    )

    ax.set_xlabel('Predicted LLM', fontsize=14, fontweight='bold')
    ax.set_ylabel('True LLM', fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save
    scale_tag = "_symlog" if scale == "symlog" else ""
    norm_tag = "" if normalize else "_raw"
    save_file = Path(save_path) / f"{tp_group_name}_confusion_orig_labels{norm_tag}{scale_tag}_bs{target_bs}.pdf"
    fig.savefig(save_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
    logger.info(f"Saved original labels confusion heatmap ({scale}, normalize={normalize}) to {save_file}")


def create_variations_confusion_heatmap(
    confusion_matrix, models_idx, tp_group_name, save_path, openset, scale='linear', target_bs=1,
    normalize=True,
):
    """
    Create confusion matrix by variations (n_var × n_var).
    Shows which variation is confused with which variation.
    """
    # Group by variations
    var_groups = group_models_idx_by_var_or_orig(models_idx, group_by="var")
    var_names = sorted(var_groups.keys())

    # Reorder to put 'ablit' at end if present, then 'Unseen' as absolute last
    if 'ablit' in var_names:
        var_names.remove('ablit')
        var_names.append('ablit')

    if 'Unseen' in var_names:
        var_names.remove('Unseen')
        var_names.append('Unseen')

    n_var = len(var_names)
    aggregated_cm = np.zeros((n_var, n_var))

    # Aggregate confusion matrix by variations
    for i, var_true in enumerate(var_names):
        for j, var_pred in enumerate(var_names):
            # Get all indices for this variation (across all original models)
            true_indices = [idx for idx, _ in var_groups[var_true]]
            pred_indices = [idx for idx, _ in var_groups[var_pred]]

            # Sum confusions for all combinations
            for true_idx in true_indices:
                for pred_idx in pred_indices:
                    aggregated_cm[i, j] += confusion_matrix[true_idx, pred_idx]

    if normalize:
        row_sums = aggregated_cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_to_plot = aggregated_cm / row_sums
    else:
        cm_to_plot = aggregated_cm

    plot_args = _get_heatmap_norm_args(scale, normalize=normalize, vmax=cm_to_plot.max())
    cbar_kws = plot_args.pop('cbar_kws')

    # Create heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    var_names_renamed = [proper_labeling(name) for name in var_names]
    sns.heatmap(
        cm_to_plot,
        annot=True,  # annot=True for variations
        fmt='.2f' if normalize else '.0f',
        cmap='YlOrRd',
        xticklabels=var_names_renamed,
        yticklabels=var_names_renamed,
        ax=ax,
        cbar_kws=cbar_kws,
        linewidths=0.5,
        linecolor='gray',
        **plot_args
    )

    ax.set_xlabel('Predicted Variation', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Variation', fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save
    scale_tag = "_symlog" if scale == "symlog" else ""
    norm_tag = "" if normalize else "_raw"
    save_file = Path(save_path) / f"{tp_group_name}_confusion_variations{norm_tag}{scale_tag}_bs{target_bs}.pdf"
    fig.savefig(save_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
    logger.info(f"Saved variations confusion heatmap ({scale}, normalize={normalize}) to {save_file}")


def create_full_tinylist_confusion_heatmap(
    confusion_matrix, models_idx, tp_group_name, save_path, openset, scale='linear', target_bs=1,
    normalize=True,
):
    """
    Create full confusion matrix (no grouping) for models in TINYLIST_OF_LLMS.
    Shows all variations of the models in the tinylist.
    """
    # Filter models_idx to only include models whose original name is in TINYLIST_OF_LLMS
    filtered_indices = []
    filtered_names = []

    for idx, full_var_model_name in models_idx.items():
        orig_name = full_var_model_name_to_original_model_name(full_var_model_name)
        if orig_name in TINYLIST_OF_LLMS:
            filtered_indices.append(idx)
            filtered_names.append(full_var_model_name)

    if len(filtered_indices) == 0:
        logger.warning(f"No models found in TINYLIST_OF_LLMS, skipping full tinylist confusion heatmap")
        return

    # Extract submatrix for filtered models
    subset_cm = confusion_matrix[np.ix_(filtered_indices, filtered_indices)]

    if normalize:
        row_sums = subset_cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_to_plot = subset_cm / row_sums
    else:
        cm_to_plot = subset_cm

    plot_args = _get_heatmap_norm_args(scale, normalize=normalize, vmax=cm_to_plot.max())
    cbar_kws = plot_args.pop('cbar_kws')

    # Create heatmap
    figsize = max(10, len(filtered_indices) * 0.4)
    fig, ax = plt.subplots(figsize=(figsize, figsize))

    # Truncate names for display
    display_names = [truncate_model_name(name) for name in filtered_names]

    sns.heatmap(
        cm_to_plot,
        annot=True,
        fmt='.2f' if normalize else '.0f',
        cmap='YlOrRd',
        xticklabels=display_names,
        yticklabels=display_names,
        ax=ax,
        cbar_kws=cbar_kws,
        linewidths=0.5,
        linecolor='gray',
        **plot_args
    )

    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=12, fontweight='bold')

    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    # Save
    scale_tag = "_symlog" if scale == "symlog" else ""
    norm_tag = "" if normalize else "_raw"
    save_file = Path(save_path) / f"{tp_group_name}_confusion_full_tinylist{norm_tag}{scale_tag}_bs{target_bs}.pdf"
    fig.savefig(save_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
    logger.info(f"Saved full tinylist confusion heatmap ({scale}, normalize={normalize}) to {save_file} ({len(filtered_indices)} models)")


def proper_labeling(label: str):
    from audit_llm.plotting.label_formatting import _parse_variation_label, QUANTIZATION_DISPLAY_MAP

    if label == 'ablit':
        return 'abliterated'
    elif label == 'Unseen':
        return 'Unseen'

    parsed = _parse_variation_label(label)
    if parsed["type"] != "temp_sp":
        raise ValueError(f"Label {label} not recognized for proper labeling.")

    quant_prefix = parsed.get("quant_prefix")
    quant_display = QUANTIZATION_DISPLAY_MAP.get(quant_prefix, quant_prefix) + " " if quant_prefix else ""
    temp, sp = parsed["temp_val"], parsed["sp_val"]
    if sp == '-1':
        return f'{quant_display}temp {temp}'
    elif temp == '1.0':
        return f'{quant_display}sp {SP_IDX_TO_ICML_SP_IDX[sp]}'
    raise ValueError(f"Label {label} not recognized for proper labeling.")