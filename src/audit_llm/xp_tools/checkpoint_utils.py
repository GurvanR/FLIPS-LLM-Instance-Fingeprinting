"""Checkpoint save/load utilities for classification results."""

import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Set, Tuple

import joblib
import numpy as np

from audit_llm.file_io import load_json

logger = logging.getLogger(__name__)


def load_results_checkpoint(results_dict, checkpoint_path, item_name):
    results = joblib.load(checkpoint_path)
    results_dict[item_name] = results


def load_classification_checkpoint(checkpoint_path: Path) -> Tuple:
    """
    Load checkpoint if it exists, otherwise initialize empty structures.

    Args:
        checkpoint_path (Path): Path to the checkpoint file.

    Returns:
        results_dict (defaultdict): Nested dictionary to store results.
        confusion_matrices_dict (defaultdict): Nested dictionary for confusion matrices.
        completed_pairs (set): Set of completed pairs.
        # new_models_idx (Optional[int]): Index of new models (None if starting fresh).
    """
    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        results_dict = defaultdict(tuple, checkpoint.get("results_dict", {}))
        confusion_matrices_dict = defaultdict(tuple, checkpoint.get("confusion_matrices_dict", {}))
        completed_pairs = set(checkpoint.get("completed_pairs", []))
        logger.info(f"Resuming from checkpoint, {len(completed_pairs)} pairs already done.")
    else:
        results_dict = defaultdict(tuple)
        confusion_matrices_dict = defaultdict(tuple)
        completed_pairs = set()
        logger.info("Starting fresh, no checkpoint found.")

    return results_dict, confusion_matrices_dict, completed_pairs  # new_models_idx


def save_classification_checkpoint(
    checkpoint_path: Path,
    results_dict: defaultdict,
    confusion_matrices_dict: defaultdict,
    completed_pairs: Set[Tuple[Any, Any]],
    pair_id: Any = None,
) -> None:
    """
    Save a classification checkpoint to disk, including results, confusion matrices,
    and completed pairs. Creates the checkpoint directory if it doesn't exist.

    Args:
        checkpoint_path (Path): Path to save the checkpoint.
        results_dict (defaultdict): Nested dictionary with results.
        confusion_matrices_dict (defaultdict): Nested dictionary with confusion matrices.
        completed_pairs (set): Set of completed pairs.
        pair_id (Optional[Any]): ID of the last processed pair (for printing/logging).
    """

    # Ensure parent directory exists
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data safely
    checkpoint_data = {
        "results_dict": results_dict if results_dict else {},
        "confusion_matrices_dict": confusion_matrices_dict if confusion_matrices_dict else {},
        "completed_pairs": list(completed_pairs) if completed_pairs else [],
    }

    # Save checkpoint
    try:
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)
        if checkpoint_path.exists():
            if pair_id is not None:
                logger.info(f"Checkpoint saved after {pair_id} -> {checkpoint_path}")
            else:
                logger.info(f"Checkpoint saved -> {checkpoint_path}")
        else:
            logger.warning(f"Failed to create checkpoint at {checkpoint_path}")
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")


def load_train_size_dict(checkpoint_dir_path, label):
    """
    Load a complete train_size_dict from saved checkpoints in either JSON or PKL format.

    Parameters:
    -----------
    checkpoint_dir_path : str or Path
        Path to the checkpoint directory (e.g., "train_size_checkpoints")
    mode : str
        'pkl' or 'json' (default is 'pkl')

    Returns:
    --------
    train_size_dict : dict
        Nested dictionary {train_size: {calculation_item_name: results}}
    """
    checkpoint_dir = Path(checkpoint_dir_path)
    train_size_dict = {}

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist.")

    # 1. Iterate through calculation_item_name directories
    for item_dir in checkpoint_dir.iterdir():
        if not item_dir.is_dir():
            continue

        calculation_item_name = item_dir.name

        # 2. Iterate through train_size directories
        for size_dir in item_dir.iterdir():
            if not size_dir.is_dir():
                continue

            try:
                train_size = int(size_dir.name)
            except ValueError:
                continue  # Skip non-integer directories

            # 3. Define path based on mode
            if "FLiPS".lower() in label.lower():
                # Original logic filename: train_size{train_size}_{item_name}.pkl
                checkpoint_path = size_dir / f"train_size{train_size}_{calculation_item_name}.pkl"
            elif "LLMmap" in label:
                # JSON mode usually uses a fixed filename
                checkpoint_path = size_dir / "results.json"
            else:
                raise ValueError("Mode must be 'pkl' or 'json'")

            # 4. Load the data
            if checkpoint_path.exists():
                if "FLiPS".lower() in label.lower():
                    results = joblib.load(checkpoint_path)
                else:
                    with open(checkpoint_path, "r") as f:
                        results = json.load(f)

                # Normalise legacy ds_wise key -> tp_wise for any source that writes it.
                if "ds_wise" in results and "tp_wise" not in results:
                    results["tp_wise"] = results.pop("ds_wise")

                if train_size not in train_size_dict:
                    train_size_dict[train_size] = {}

                train_size_dict[train_size][calculation_item_name] = results

    # 5. Apply cleanup logic (as seen in your snippet)
    if "LLMmap" in label:
        # JSON often requires converting list-based matrices back to numpy
        if "arrayify_confusion_matrices" in globals():
            train_size_dict = arrayify_confusion_matrices(train_size_dict)

    # Ensure keys are integers (useful if dict was initialized from JSON strings)
    if "set_key_as_int" in globals():
        train_size_dict = set_key_as_int(train_size_dict)

    # Find new_var_models_idx.json — LLMmap writes to <xp_root>/checkpoint_dir/, FLiPS to <xp_root>/checkpoints/,
    # and openset adds an extra alpha_<x>/ level above train_size_checkpoints.
    candidates = []
    for ancestor in (checkpoint_dir.parent.parent, checkpoint_dir.parent.parent.parent):
        for folder in ("checkpoint_dir", "checkpoints"):
            candidates.append(ancestor / folder / "new_var_models_idx.json")
    new_var_models_idx = {}
    for candidate in candidates:
        if candidate.exists():
            new_var_models_idx = load_json(path=candidate, set_keys_as_int=True)
            break
    else:
        logger.warning("new_var_models_idx.json not found for %s; returning empty dict.", checkpoint_dir)

    return train_size_dict, new_var_models_idx


def arrayify_confusion_matrices(train_size_dict):
    for train_size in train_size_dict:
        for calculation_item_name in train_size_dict[train_size]:
            for batch_type in train_size_dict[train_size][calculation_item_name]:
                for bs in train_size_dict[train_size][calculation_item_name][batch_type]:
                    for tp in train_size_dict[train_size][calculation_item_name][batch_type][bs]:
                        for clf in train_size_dict[train_size][calculation_item_name][batch_type][bs][tp]:
                            for metric in train_size_dict[train_size][calculation_item_name][batch_type][bs][tp][clf]:
                                if metric in ["confusion_matrix_mean", "confusion_matrix_std"]:
                                    object = train_size_dict[train_size][calculation_item_name][batch_type][bs][tp][
                                        clf
                                    ][metric]
                                    array = np.array(object)
                                    if len(array.shape) != 2:
                                        logger.warning("array shape is not 2D for confusion matrix mean/std")
                                        logger.warning("hardcoding to (205,205)")
                                        array = np.zeros((205, 205))
                                    train_size_dict[train_size][calculation_item_name][batch_type][bs][tp][clf][
                                        metric
                                    ] = array
                                if metric == "confusion_matrix_all":
                                    list_of_matrices = train_size_dict[train_size][calculation_item_name][batch_type][
                                        bs
                                    ][tp][clf][metric]
                                    array_of_matrices = [np.array(cm) for cm in list_of_matrices]
                                    train_size_dict[train_size][calculation_item_name][batch_type][bs][tp][clf][
                                        metric
                                    ] = array_of_matrices
    return train_size_dict


def set_key_as_int(train_size_dict):
    """
    Set bs keys as int

    """
    for train_size in train_size_dict:
        for calculation_item_name in train_size_dict[train_size]:
            for batch_type in train_size_dict[train_size][calculation_item_name]:
                bs_map = train_size_dict[train_size][calculation_item_name][batch_type]
                int_bs_map = {}
                for bs_str, tp_map in bs_map.items():
                    int_bs_map[int(bs_str)] = tp_map
                train_size_dict[train_size][calculation_item_name][batch_type] = int_bs_map

    return train_size_dict
