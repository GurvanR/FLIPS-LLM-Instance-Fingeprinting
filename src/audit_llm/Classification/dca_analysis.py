"""
DCA (Dataset Confusion Analysis) computation and visualization.

Provides functions for computing Wasserstein distances between abliterated and safe model
probability distributions, and plotting DCA showcase results grouped by original model or
by variation.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

from audit_llm.models_management.model_names import ABLITERATED_MODELS_MAP_TO_ORIGINAL
from audit_llm.plot_configs import get_mpl_configs
from audit_llm.data_transforms import revert_dictionary

logger = logging.getLogger(__name__)
from audit_llm.xp_tools.model_filtering import (
    full_var_model_name_to_original_model_name,
    full_var_model_name_to_var_name,
)


def compute_dca_showcase_data(
    train_size_dict: Dict[float, Dict[float, Dict[Any, Any]]],
    classification_config: Dict,
    full_var_model_name_to_full_safe_var_model_name_map: Dict[str, int],
) -> None:
    """
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
        summary_results[batch_size][tp][clf]['['probs_save_map'] = probs_save_map # Dict{class labels of dca-showcase}]

        where key_name in ['tp_wise', 'mix_tp_at_pred']
        where vals are values of the metrics.

    """
    batch_types = classification_config.get("batch_types") or ["tp_wise"]
    train_sizes = sorted(train_size_dict.keys())
    calculation_item_names = sorted({t for ts in train_size_dict.values() for t in ts.keys()})
    for calculation_item_name in calculation_item_names:
        for clf in classification_config["classifiers"]:
            for train_size in train_sizes:
                summary_results = train_size_dict[train_size][calculation_item_name]
                for batch_type in batch_types:
                    if 8 not in summary_results[batch_type]:
                        logger.warning(
                            "DCA analysis skipped for %s: batch size 8 not available (available: %s)",
                            batch_type,
                            sorted(summary_results[batch_type].keys()),
                        )
                        continue
                    for tp, clf_dict in summary_results[batch_type][8].items():
                        probs_save_map = clf_dict[clf]["probs_save_map"]
                        if clf == "llmmap_clf":
                            mode = "LLMmap"
                        else:
                            mode = "FLiPS"
                        dca_showcase_data: Dict[str, Dict[str, float]] = compute_dca_showcase_from_probs_save_map(
                            probs_save_map, full_var_model_name_to_full_safe_var_model_name_map, mode=mode
                        )
                        clf_dict[clf]["dca_showcase_data"] = dca_showcase_data


def compute_dca_showcase_from_probs_save_map(
    probs_save_map,
    full_var_model_name_to_full_safe_var_model_name_map,
    mode: str,
    normalize=True,
    method="mean",
    verbose=True,
):
    """
    probs_save_map:
        - probs_save_map[f"{class_label}"] = List[float]
          # probs of the true class when presented this class.
        - probs_save_map[f"safe_{class_label}"] = List[float]
          # probs of the safe class when presented this class.
        - probs_save_map[f"top1_pred_{class_label}"] = List[str]
          # top1 prediction class_label for each sample.

    full_var_model_name_to_full_safe_var_model_name_map:
        Dict mapping class_label to its reference safe_label.

    normalize: bool, whether to normalize the distance.
    method: str, normalization method:
        - 'range': divide by the range of combined distributions (max - min)
        - 'max': divide by the maximum value in combined distributions
        - 'mean': divide by the mean of combined distributions
        - 'std': divide by the pooled standard deviation

    Returns:
        Dict: {
            'distance': {class_label: float}, # Wasserstein distance to reference
            'top1_safe': {class_label: float}  # % of time top1_pred == safe_label
        }
    """
    result = {"distance": {}, "top1_safe": {}}

    # Extract all class labels (those without "safe_" prefix)
    class_labels = ["_".join(key.split("_")[1:]) for key in probs_save_map.keys() if key.startswith("safe_")]

    if verbose:
        logger.debug("Processing labels: %s", class_labels)

    for class_label in class_labels:
        # Identify the reference label and relevant keys
        safe_label = full_var_model_name_to_full_safe_var_model_name_map[class_label]
        safe_reference_key = f"safe_{safe_label}"
        safe_key = f"safe_{class_label}"
        pred_key = f"top1_pred_{class_label}"

        # Safety Checks
        if safe_reference_key not in probs_save_map:
            raise ValueError(f"Reference key '{safe_reference_key}' not found in map.")
        if safe_key not in probs_save_map:
            raise ValueError(f"Safe distribution key '{safe_key}' not found in map.")

        # --- 1. Compute Wasserstein Distance ---
        probs_safe_reference = np.array(probs_save_map[safe_reference_key])
        probs_safe_class = np.array(probs_save_map[safe_key])

        distance = wasserstein_distance(probs_safe_class, probs_safe_reference)

        if normalize:
            combined = np.concatenate([probs_safe_class, probs_safe_reference])
            normalizer = 1.0
            if method == "range":
                normalizer = np.max(combined) - np.min(combined)
            elif method == "max":
                normalizer = np.max(combined)
            elif method == "mean":
                normalizer = np.mean(combined)
            elif method == "std":
                normalizer = np.std(combined)

            if normalizer > 0:
                distance = distance / normalizer

        result["distance"][class_label] = float(distance)

        # --- 2. Compute Top-1 Safe Percentage ---
        if pred_key in probs_save_map:
            predictions = np.array(probs_save_map[pred_key])
            # Check if the predicted label matches the intended safe_label
            if mode == "LLMmap":
                safe_label_pred = full_var_model_name_to_original_model_name(safe_label)
            else:
                safe_label_pred = safe_label
            matches = predictions == safe_label_pred
            logger.debug("predictions=%s\n safe_label_pred=%s", predictions, safe_label_pred)
            accuracy_safe_pct = np.mean(matches) * 100.0
            logger.debug("accuracy_safe_pct=%s", accuracy_safe_pct)
            result["top1_safe"][class_label] = float(accuracy_safe_pct)
        elif verbose:
            logger.warning("Key '%s' not found. 'top1_safe' skipped for %s.", pred_key, class_label)

    return result


def plot_dca_showcase_by_original_model(
    train_size_dict: Dict[float, Dict[float, Dict[Any, Any]]],
    classification_config: Dict,
    fig_save_path: Path | str,
    batch_type: Optional[str] = None,
    label_1: str = "LLMmap-IPP",
    label_2: str = "FLIPS",
    group_by: str = "var",  # NEW: 'orig' (default) or 'var'
):
    # --- 1. Initialization & Config ---
    if batch_type is None:
        batch_types = classification_config.get("batch_types") or ["tp_wise"]
        batch_type = batch_types[0]

    if batch_type == "across_and_tp_wise":
        raise NotImplementedError("plot_dca_showcase_by_original_model() does not support 'across_and_tp_wise'.")

    if group_by not in ("orig", "var"):
        raise ValueError("group_by must be 'orig' or 'var'")

    def _safe_name(s: str) -> str:
        return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(s))

    train_sizes = sorted(train_size_dict.keys())
    calculation_item_names = sorted({t for ts in train_size_dict.values() for t in ts.keys()})
    classifiers = classification_config.get("classifiers", [])

    # Handle Comparison Mode
    train_size_dict_2_path = classification_config.get("train_size_dict_2_checkpoint_path", None)
    compare_mode = train_size_dict_2_path is not None
    train_size_dict_2 = {}
    classifiers_2 = []

    if compare_mode:
        if "pkl" in train_size_dict_2_path:
            train_size_dict_2 = joblib.load(train_size_dict_2_path)
        else:
            pass
            # train_size_dict_2, _ = load_train_size_dict(train_size_dict_2_path, train_sizes=train_sizes)

    # --- 2. Main Iteration Loops ---
    for calc_name in calculation_item_names:
        for clf_idx, clf in enumerate(classifiers):
            # In compare mode, get the corresponding classifier from dataset 2
            clf_2 = classifiers_2[clf_idx] if classifiers_2 and clf_idx < len(classifiers_2) else clf

            for train_size in train_sizes:
                summary_results = train_size_dict[train_size][calc_name]
                if (
                    compare_mode and "pkl" in train_size_dict_2_path
                ):  # this is kind of artifact, from old way of saving summary (now FLiPS pipeline ahs 'all' key in train_size_dict)
                    summary_results_2 = train_size_dict_2
                else:
                    summary_results_2 = train_size_dict_2[train_size][calc_name]

                # --- 3. Aggregate data across all datasets for EACH summary_dict separately ---
                aggregated_dca_1 = {}  # {metric_key: {class_label: [values]}}
                aggregated_dca_2 = {}

                # Aggregate dataset 1 data across all tp
                for tp, clf_dict in summary_results[batch_type][8].items():
                    if clf not in clf_dict:
                        continue
                    dca_data_1 = clf_dict[clf]["dca_showcase_data"]

                    for metric_key in ["distance", "top1_safe"]:
                        m_data_1 = dca_data_1.get(metric_key, {})
                        if metric_key not in aggregated_dca_1:
                            aggregated_dca_1[metric_key] = {}

                        for class_label, val in m_data_1.items():
                            if class_label not in aggregated_dca_1[metric_key]:
                                aggregated_dca_1[metric_key][class_label] = []
                            aggregated_dca_1[metric_key][class_label].append(val)

                # Aggregate dataset 2 data across all tp (if in compare mode)
                if compare_mode and summary_results_2:
                    if 8 in summary_results_2.get(batch_type, {}):
                        for tp, clf_dict in summary_results_2[batch_type][8].items():
                            if clf_2 not in clf_dict:
                                continue
                            dca_data_2 = clf_dict[clf_2].get("dca_showcase_data")
                            if dca_data_2:
                                for metric_key in ["distance", "top1_safe"]:
                                    m_data_2 = dca_data_2.get(metric_key, {})
                                    if metric_key not in aggregated_dca_2:
                                        aggregated_dca_2[metric_key] = {}

                                    for class_label, val in m_data_2.items():
                                        # maintain the ablit naming you had
                                        if "abliterated" in class_label and not "_ablit" in class_label:
                                            class_label = "_".join([class_label, "ablit"])
                                        if class_label not in aggregated_dca_2[metric_key]:
                                            aggregated_dca_2[metric_key][class_label] = []
                                        aggregated_dca_2[metric_key][class_label].append(val)

                # --- 4. Average the aggregated data for each dataset separately ---
                averaged_dca_1 = {}
                averaged_dca_2 = {}

                for metric_key in ["distance", "top1_safe"]:
                    averaged_dca_1[metric_key] = {
                        class_label: float(np.mean(values))
                        for class_label, values in aggregated_dca_1.get(metric_key, {}).items()
                    }

                    if compare_mode:
                        averaged_dca_2[metric_key] = {
                            class_label: float(np.mean(values))
                            for class_label, values in aggregated_dca_2.get(metric_key, {}).items()
                        }

                # --- 5. Generate plots for each metric ---
                for metric_key in [
                    #'distance',
                    "top1_safe"
                ]:
                    _generate_metric_plots(
                        metric_key=metric_key,
                        dca_data_1=averaged_dca_1,
                        dca_data_2=averaged_dca_2 if compare_mode else None,
                        calc_name=calc_name,
                        clf=clf,
                        clf_2=clf_2 if compare_mode else None,
                        train_size=train_size,
                        ds="avg_all_ds",
                        fig_save_path=fig_save_path,
                        compare_mode=compare_mode,
                        labels_cfg=(label_1, label_2),
                        safe_name_func=_safe_name,
                        group_by=group_by,  # pass through the new option
                    )


def _prepare_metric_plot_data(metric_key, dca_data_1, dca_data_2, group_by="orig"):
    """Extracts and groups data for plotting using the grouping helper."""
    m_data_1 = dca_data_1.get(metric_key, {})
    m_data_2 = dca_data_2.get(metric_key, {}) if dca_data_2 else {}

    if not m_data_1:
        return None, None, None

    # Use the factorized helper for both datasets
    grouped_1 = _group_dca_metric_data(m_data_1, group_by=group_by)
    grouped_2 = _group_dca_metric_data(m_data_2, group_by=group_by)

    per_unit_name = "orig_model" if group_by == "orig" else "variation"

    return grouped_1, grouped_2, per_unit_name


def _group_dca_metric_data(data, group_by="orig"):
    """
    Groups raw metric data by either original model name or variation.
    Returns a dictionary of {unit_name: {label: value}}.
    """
    if not data:
        return {}

    if group_by == "orig":
        grouped = {}
        for class_label, val in data.items():
            orig = full_var_model_name_to_original_model_name(class_label)
            grouped.setdefault(orig, {})[class_label] = val
        return grouped

    # group_by != "orig" (variation logic)
    grouped_raw = {}
    for class_label, val in data.items():
        var = full_var_model_name_to_var_name(class_label)
        orig = full_var_model_name_to_original_model_name(class_label)
        grouped_raw.setdefault(var, {}).setdefault(orig, []).append(val)

    # Calculate means for variations
    return {
        var: {orig: float(np.mean(vals)) for orig, vals in orig_map.items()} for var, orig_map in grouped_raw.items()
    }


def _generate_metric_plots(
    metric_key,
    dca_data_1,
    dca_data_2,
    calc_name,
    clf,
    clf_2,
    train_size,
    ds,
    fig_save_path,
    compare_mode,
    labels_cfg,
    safe_name_func,
    group_by="orig",
):

    # --- 1. Data Preparation ---
    grouped, grouped_2, per_unit_name = _prepare_metric_plot_data(metric_key, dca_data_1, dca_data_2, group_by)

    if grouped is None:
        return

    # --- 2. Configuration & Formatting ---
    ylabel_text = "Wasserstein Distance" if metric_key == "distance" else "Original LLM Prediction Rate"
    val_format = ".3f" if metric_key == "distance" else ".1f"

    # --- 3. Plotting Loop ---
    for unit_name, cl_map in grouped.items():
        cl_map_2 = grouped_2.get(unit_name, {})

        # Reverse Order: Changed reverse=True to reverse=False to flip the x-axis order
        all_labels = sorted(set(cl_map.keys()) | set(cl_map_2.keys()), key=lambda l: cl_map.get(l, 0), reverse=False)

        vals_1 = [cl_map.get(l, 0) for l in all_labels]
        vals_2 = [cl_map_2.get(l, 0) for l in all_labels] if compare_mode else None

        # --- Create Figure ---
        fig_config = get_mpl_configs(multiplier=1, col_type="single_col")
        fig, ax = plt.subplots(**fig_config["fig_config"])
        indices = np.arange(len(all_labels))

        # Rename labels (extracting model name after '/')
        model_name_to_ablit = revert_dictionary(ABLITERATED_MODELS_MAP_TO_ORIGINAL)
        all_labels_renamed = [model_name_to_ablit[name].split("/")[1] for name in all_labels]

        # --- Plotting ---
        if compare_mode and vals_2 and any(v is not None for v in vals_2):
            width = 0.35
            rects1 = ax.bar(indices - width / 2, vals_1, width, label=f"{labels_cfg[0]}")
            rects2 = ax.bar(indices + width / 2, vals_2, width, label=f"{labels_cfg[1]}", color="red")
            ax.legend(**fig_config["legend_config"])

            # Values on top of bars
            ax.bar_label(rects1, padding=3, fmt=f"{{:{val_format}}}", fontsize=6)
            ax.bar_label(rects2, padding=3, fmt=f"{{:{val_format}}}", fontsize=6)
        else:
            rects1 = ax.bar(indices, vals_1)
            ax.bar_label(rects1, padding=3, fmt=f"{{:{val_format}}}", fontsize=6)

        # --- Style & Labels ---
        ax.set_xticks(indices)
        ax.set_xticklabels(all_labels_renamed, rotation=0)

        # Labels
        ax.set_xlabel("Abliterated LLM", **fig_config["xlabel_config"])
        ax.set_ylabel(ylabel_text, **fig_config["ylabel_config"])

        # Tick params & layout
        ax.tick_params(
            axis="x",
            labelsize=3.6,
        )
        ax.tick_params(axis="y", **fig_config["yticks_config"])
        ax.grid(**fig_config["grid_config"])
        ax.margins(y=0.2)  # Space for bar labels

        plt.tight_layout()

        # --- 4. Save Logic ---
        os.makedirs(fig_save_path, exist_ok=True)
        fname = (
            f"dca_{metric_key}_{safe_name_func(calc_name)}_{safe_name_func(clf)}_"
            f"{safe_name_func(train_size)}_{safe_name_func(ds)}_"
            f"{safe_name_func(per_unit_name)}_{safe_name_func(unit_name)}_groupby_{group_by}.pdf"
        )

        fig.savefig(os.path.join(fig_save_path, fname), bbox_inches="tight")
        plt.close(fig)
