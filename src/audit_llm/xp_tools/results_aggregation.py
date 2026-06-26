"""Results aggregation and normalization utilities.

This module consolidates the duplicate aggregate_results_dict functions.
"""

from collections import defaultdict
from itertools import product
from typing import Dict, List

import numpy as np

from audit_llm.file_io import load_json


def dict_product_with_fix_item(calculations_iter_lists: Dict, fix_iterator_idx=None, fix_item=None):
    """
    Normal Functionning fix_iterator_idx = 'for_each'
    """
    if not calculations_iter_lists:
        return [{"all": None}]

    ## removing for_each key from calculations_iter_lists
    calculations_iter_lists_minus_fix_iterator_idx = calculations_iter_lists.copy()
    if fix_iterator_idx is not None:
        calculations_iter_lists_minus_fix_iterator_idx.pop(fix_iterator_idx)

    if not calculations_iter_lists_minus_fix_iterator_idx:
        return [{"all": None}]

    ## Get the keys and values
    keys = calculations_iter_lists_minus_fix_iterator_idx.keys()
    values = calculations_iter_lists_minus_fix_iterator_idx.values()
    calculations_items = []  # cartesian_product e.g. (iterator_1^i, iterator_2^j, ..)_i,j,...
    ## Loop dynamically over all combinations
    for combo in product(*values):
        if fix_item is not None:
            assert fix_iterator_idx is not None
            item = dict(zip(keys, combo)) | {fix_iterator_idx: fix_item}  # adding the for_each, for every combos.
        else:
            item = dict(zip(keys, combo))

        calculations_items.append(item)

    return calculations_items


def get_actual_sp_from_sp_indices(sp_indices: List[int]):
    """ """
    system_prompt_list: List[str] = load_json("system_prompts")
    sp_dict = {}
    for sp_idx in sp_indices:
        if sp_idx == -1:
            sp_dict[sp_idx] = "NO SYSTEM PROMPT"
        elif sp_idx in range(len(system_prompt_list)):
            sp_dict[sp_idx] = system_prompt_list[sp_idx]
        else:
            raise ValueError(f"System prompt index {sp_idx} is out of range.")
    return sp_dict


def normalize_matrix(matrix, method="min_max"):
    """
    Normalize a matrix using the specified method.

    Args:
        matrix: np.ndarray to normalize
        method: str, either 'min_max' or 'z_score'

    Returns:
        normalized matrix
    """
    if method == "min_max":
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        if max_val - min_val == 0:
            return np.zeros_like(matrix)
        return (matrix - min_val) / (max_val - min_val)
    elif method == "z_score":
        mean_val = np.mean(matrix)
        std_val = np.std(matrix)
        if std_val == 0:
            return np.zeros_like(matrix)
        return (matrix - mean_val) / std_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def aggregate_results_dict(per_for_each_dict, mode="results", aggregation="mean", normalization=None):
    """
    Aggregate results and confusion matrices across multiple 'for_each_item' entries.
    
    This function consolidates the previously duplicated aggregate_results_dict and
    aggregate_results_dict_NORMALIZE functions by adding a normalization parameter.

    Args:
        per_for_each_dict: Dict[
            for_each_item: {
                pair_id: summary if mode == 'results' else confusion_matrix
            }
        ]
            - pair_id: (i, j)
            - summary: Dict[clf_name][<metric>_mean/<metric>_std] = value
            - confusion_matrix: Dict[clf_name][List[np.ndarray]]
        mode: str
            Either "results" or "confusion_matrix".
        aggregation: str
            Either "mean" or "median".
        normalization: str or None
            If None, no normalization is applied (original behavior).
            If "min_max" or "z_score", normalizes matrices before aggregation.

    Returns:
        aggregated_dict: same structure as values of per_for_each_dict.
    """
    assert mode in ("results", "confusion_matrix"), "mode must be 'results' or 'confusion_matrix'"
    assert aggregation in ("mean", "median"), "aggregation must be 'mean' or 'median'"

    agg_func = np.mean if aggregation == "mean" else np.median

    if mode == "results":
        if normalization is None:
            # Original behavior: no normalization
            # Collect all metric values for aggregation
            accumulator = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

            for _, results_dict in per_for_each_dict.items():
                for pair_id, summary in results_dict.items():
                    for clf_name, metrics in summary.items():
                        for metric_name, value in metrics.items():
                            accumulator[pair_id][clf_name][metric_name].append(value)

            # Aggregate across for_each_item
            aggregated_dict = {}
            for pair_id, clf_data in accumulator.items():
                aggregated_dict[pair_id] = {}
                for clf_name, metrics in clf_data.items():
                    aggregated_dict[pair_id][clf_name] = {
                        metric_name: float(agg_func(values)) for metric_name, values in metrics.items()
                    }
        else:
            # Normalized behavior
            # First pass: determine matrix dimensions
            all_pair_ids = set()
            for for_each_item, results_dict in per_for_each_dict.items():
                all_pair_ids.update(results_dict.keys())

            if not all_pair_ids:
                return {}

            max_i = max(pair_id[0] for pair_id in all_pair_ids) + 1
            max_j = max(pair_id[1] for pair_id in all_pair_ids) + 1

            # Collect normalized matrices for each for_each_item
            # Structure: normalized_data[clf_name][metric_name][for_each_item] = normalized_matrix
            normalized_data = defaultdict(lambda: defaultdict(dict))

            for for_each_item, results_dict in per_for_each_dict.items():
                # Build matrices for this for_each_item
                # Structure: matrices[clf_name][metric_name] = matrix
                matrices = defaultdict(lambda: defaultdict(lambda: np.full((max_i, max_j), np.nan)))

                for pair_id, summary in results_dict.items():
                    for clf_name, metrics in summary.items():
                        for metric_name, value in metrics.items():
                            matrices[clf_name][metric_name][pair_id[0], pair_id[1]] = value

                # Normalize each matrix
                for clf_name, metric_dict in matrices.items():
                    for metric_name, matrix in metric_dict.items():
                        # Only normalize non-NaN values
                        mask = ~np.isnan(matrix)
                        if np.any(mask):
                            normalized_matrix = np.full_like(matrix, np.nan)
                            normalized_matrix[mask] = normalize_matrix(
                                matrix[mask].reshape(-1, 1), method=normalization
                            ).flatten()
                            normalized_data[clf_name][metric_name][for_each_item] = normalized_matrix
                        else:
                            normalized_data[clf_name][metric_name][for_each_item] = matrix

            # Aggregate across for_each_items
            # Structure: aggregated[clf_name][metric_name] = aggregated_matrix
            aggregated = defaultdict(lambda: defaultdict(lambda: None))

            for clf_name, metric_dict in normalized_data.items():
                for metric_name, for_each_dict in metric_dict.items():
                    # Stack all matrices for this clf_name and metric_name
                    matrices_list = list(for_each_dict.values())
                    if matrices_list:
                        stacked = np.stack(matrices_list, axis=0)
                        # Aggregate along the for_each_item axis (axis=0), ignoring NaNs
                        if aggregation == "mean":
                            aggregated[clf_name][metric_name] = np.nanmean(stacked, axis=0)
                        else:  # median
                            aggregated[clf_name][metric_name] = np.nanmedian(stacked, axis=0)
                    else:
                        raise ValueError("No matrices to aggregate.")

            # Convert back to the original structure (pair_id -> clf_name -> metric -> value)
            aggregated_dict = {}
            for i in range(max_i):
                for j in range(max_j):
                    pair_id = (i, j)
                    if pair_id in all_pair_ids:
                        aggregated_dict[pair_id] = {}
                        for clf_name, metric_dict in aggregated.items():
                            aggregated_dict[pair_id][clf_name] = {}
                            for metric_name, matrix in metric_dict.items():
                                value = matrix[i, j]  # type:ignore
                                if not np.isnan(value):
                                    aggregated_dict[pair_id][clf_name][metric_name] = value

    else:  # mode == "confusion_matrix"
        # Collect all confusion matrices for aggregation
        accumulator = defaultdict(lambda: defaultdict(list))

        for _, cm_dict in per_for_each_dict.items():
            for pair_id, confusion_matrices_dict in cm_dict.items():
                for clf_name, cm_list in confusion_matrices_dict.items():
                    # cm_list is already a list of confusion matrices
                    for cm in cm_list:
                        accumulator[pair_id][clf_name].append(np.array(cm))

        # Aggregate confusion matrices per pair_id and clf_name
        aggregated_dict = {}
        for pair_id, clf_data in accumulator.items():
            aggregated_dict[pair_id] = {}
            for clf_name, cm_list in clf_data.items():
                # Stack all confusion matrices and aggregate along axis 0
                aggregated_dict[pair_id][clf_name] = agg_func(np.stack(cm_list, axis=0), axis=0)

    return aggregated_dict
