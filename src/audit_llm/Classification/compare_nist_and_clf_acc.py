import logging
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from audit_llm.plot_configs import *

logger = logging.getLogger(__name__)
from audit_llm.xp_tools.checkpoint_utils import load_train_size_dict
from audit_llm.xp_tools.model_filtering import (
    full_var_model_name_to_original_model_name,
    full_var_model_name_to_var_name,
)


def clean_name(n: object) -> str:
    return str(n).replace("/", "_").replace(" ", "_")


def parse_nist_key(key):
    """
    Parse NIST performance key robustly using regex.

    Returns dict with: orig_model_name, tp, feature, temp (optional), sp (optional)

    Example keys:
    - tp-HE-mind_sp-3_temp-0.6_model-CohereForAI/c4ai-command-r-plus_feature-overlapping patterns_110_75_pv 0.464
    - tp-way-ither_sp--1_temp-1.0_model-CohereForAI/c4ai-command-r-plus_feature-spectral_pv 0.1637
    """
    result = {}

    # Extract tp (everything between tp- and next underscore before another keyword)
    tp_match = re.search(r"tp-([^_]+(?:_[^_]+)*?)(?=_(?:model|temp|sp|feature)-)", key)
    if tp_match:
        result["tp"] = tp_match.group(1)

    # Extract model (everything between model- and _feature-)
    model_match = re.search(r"model-(.+?)_feature-", key)
    if model_match:
        result["orig_model_name"] = model_match.group(1)

    # Extract feature (everything after feature- to end or before _pv)
    feature_match = re.search(r"feature-(.+?)(?:_pv|$)", key)
    if feature_match:
        result["feature"] = feature_match.group(1).strip()

    # Extract temp (optional)
    temp_match = re.search(r"temp-([\d.]+)", key)
    if temp_match:
        result["temp"] = temp_match.group(1)

    # Extract sp (optional, can be negative)
    sp_match = re.search(r"sp-(-?\d+)", key)
    if sp_match:
        result["sp"] = sp_match.group(1)

    return result


def build_var_name_from_nist(parsed):
    """
    Build variation name from parsed NIST data to match full_var_model_name format.

    Returns: var_name like "temp-0.6_sp-3" or "temp-1.0_sp--1" or None if no variation
    """
    parts = []

    if "temp" in parsed:
        parts.append(f"temp-{parsed['temp']}")
    if "sp" in parsed:
        parts.append(f"sp-{parsed['sp']}")

    return "_".join(parts) if parts else None


def parse_nist_data(nist_perfs_dict):
    """
    Parse NIST performance dictionary into structured format.

    Returns: nested dict[orig_model_name][var_name][tp] = [success_rates for features]
             where var_name could be None for base models
    """
    nist_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for key, val in nist_perfs_dict.items():
        parsed = parse_nist_key(key)

        if "orig_model_name" not in parsed or "tp" not in parsed:
            logger.warning(f"Skipping malformed NIST key: {key}")
            continue

        orig_model = parsed["orig_model_name"]
        tp = parsed["tp"]
        var_name = build_var_name_from_nist(parsed)

        # Store: nist_data[orig_model][var_name][tp].append(success_rate)
        nist_data[orig_model][var_name][tp].append(val)

    return nist_data


def match_full_var_model_to_nist(full_var_model_name, nist_data):
    """
    Match a full_var_model_name to NIST data.

    Args:
        full_var_model_name: e.g., "CohereForAI/c4ai-command-r-plus_temp-0.6_sp-3"
        nist_data: parsed NIST data structure

    Returns: (orig_model_name, var_name) or (None, None) if no match
    """
    orig_model = full_var_model_name_to_original_model_name(full_var_model_name)
    var_name = full_var_model_name_to_var_name(full_var_model_name)

    # Handle case where var_name equals orig_model (no variation)
    if var_name == full_var_model_name:
        var_name = None

    if orig_model in nist_data and var_name in nist_data[orig_model]:
        return orig_model, var_name

    return None, None


def plot_scatter_accuracy_vs_nist(bs_data, clf, loaded_models_idx, nist_data, train_size, save_dir):
    """
    PLOT 4: Scatter plot of Accuracy vs NIST for each model variation at each TP.
    """
    logger.info("--- SCATTER PLOT ---")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define TPs and their properties
    tp_info = {
        "0-1": {"color": "blue", "label": "0-1", "marker": "o"},
        "scar-este": {"color": "green", "label": "scar-este", "marker": "s"},
        "four-iously": {"color": "red", "label": "four-iously", "marker": "^"},
    }

    total_points = 0

    for tp, props in tp_info.items():
        if tp not in bs_data:
            logger.debug(f"  TP '{tp}' not in data")
            continue

        if clf not in bs_data[tp]:
            logger.debug(f"  clf '{clf}' not in TP '{tp}'")
            continue

        cm_mean = bs_data[tp][clf]["confusion_matrix_mean"]
        row_sums = cm_mean.sum(axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            per_class_acc = np.diag(cm_mean) / row_sums
            per_class_acc = np.nan_to_num(per_class_acc)

        acc_vals = []
        nist_vals = []

        # For each model variation
        for model_idx, full_var_model_name in loaded_models_idx.items():
            if model_idx >= len(per_class_acc):
                continue

            # Match to NIST data
            orig_model, var_name = match_full_var_model_to_nist(full_var_model_name, nist_data)

            if orig_model and var_name in nist_data[orig_model] and tp in nist_data[orig_model][var_name]:
                nist_scores = nist_data[orig_model][var_name][tp]
                if nist_scores:
                    acc_vals.append(per_class_acc[model_idx])
                    nist_vals.append(np.mean(nist_scores))  # Average across features

        if acc_vals and nist_vals:
            ax.scatter(
                acc_vals, nist_vals, color=props["color"], label=props["label"], marker=props["marker"], alpha=0.6, s=50
            )
            total_points += len(acc_vals)
            logger.info(f"  Plotted {len(acc_vals)} points for TP '{tp}'")

    logger.info(f"Total points plotted: {total_points}")

    ax.set_xlabel("Accuracy")
    ax.set_ylabel("NIST Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = f"nist_vs_clf_scatter_ts{clean_name(train_size)}_{clean_name(clf)}.pdf"
    plt.savefig(save_dir / fname, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {fname}")


def plot_scatter_accuracy_vs_nist_two_categories(bs_data, clf, loaded_models_idx, nist_data, train_size, save_dir):
    """
    PLOT 5: Scatter plot with two categories: '0-1' and average of all other TPs.
    """
    logger.info("--- SCATTER PLOT (Two Categories) ---")
    fig_config_single_col = get_mpl_configs(multiplier=2.5, col_type="single_col")

    fig, ax = plt.subplots(**fig_config_single_col["fig_config"])

    # Category 1: TP='0-1'
    if "0-1" in bs_data and clf in bs_data["0-1"]:
        cm_mean = bs_data["0-1"][clf]["confusion_matrix_mean"]
        row_sums = cm_mean.sum(axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            per_class_acc = np.diag(cm_mean) / row_sums
            per_class_acc = np.nan_to_num(per_class_acc)

        acc_vals_01 = []
        nist_vals_01 = []

        for model_idx, full_var_model_name in loaded_models_idx.items():
            if model_idx >= len(per_class_acc):
                continue

            orig_model, var_name = match_full_var_model_to_nist(full_var_model_name, nist_data)

            if orig_model and var_name in nist_data[orig_model] and "0-1" in nist_data[orig_model][var_name]:
                nist_scores = nist_data[orig_model][var_name]["0-1"]
                if nist_scores:
                    acc_vals_01.append(per_class_acc[model_idx])
                    nist_vals_01.append(np.mean(nist_scores))

        if acc_vals_01 and nist_vals_01:
            ax.scatter(acc_vals_01, nist_vals_01, color="blue", label="0-1", marker="o", alpha=0.6, s=50)
            logger.info(f"  Plotted {len(acc_vals_01)} points for '0-1'")

    # Category 2: Average of all other TPs
    other_tps = [tp for tp in bs_data.keys() if tp != "0-1"]

    if other_tps:
        # For each model, collect accuracies and NIST scores across all other TPs
        model_acc_dict = defaultdict(list)
        model_nist_dict = defaultdict(list)

        for tp in other_tps:
            if clf not in bs_data[tp]:
                continue

            cm_mean = bs_data[tp][clf]["confusion_matrix_mean"]
            row_sums = cm_mean.sum(axis=1)

            with np.errstate(divide="ignore", invalid="ignore"):
                per_class_acc = np.diag(cm_mean) / row_sums
                per_class_acc = np.nan_to_num(per_class_acc)

            for model_idx, full_var_model_name in loaded_models_idx.items():
                if model_idx >= len(per_class_acc):
                    continue

                orig_model, var_name = match_full_var_model_to_nist(full_var_model_name, nist_data)

                if orig_model and var_name in nist_data[orig_model] and tp in nist_data[orig_model][var_name]:
                    nist_scores = nist_data[orig_model][var_name][tp]
                    if nist_scores:
                        model_acc_dict[model_idx].append(per_class_acc[model_idx])
                        model_nist_dict[model_idx].append(np.mean(nist_scores))

        # Average across TPs for each model
        acc_vals_other = []
        nist_vals_other = []

        for model_idx in model_acc_dict:
            if model_acc_dict[model_idx] and model_nist_dict[model_idx]:
                acc_vals_other.append(np.mean(model_acc_dict[model_idx]))
                nist_vals_other.append(np.mean(model_nist_dict[model_idx]))

        if acc_vals_other and nist_vals_other:
            ax.scatter(
                acc_vals_other,
                nist_vals_other,
                color="red",
                label="XP set of 30 token pairs (avg)",
                marker="^",
                alpha=0.6,
                s=50,
            )
            logger.info(f"  Plotted {len(acc_vals_other)} points for 'Other TPs (avg)'")

    ax.set_xlabel("Fingerprinting Accuracy", **fig_config_single_col["xlabel_config"])
    ax.set_ylabel("NIST Score", **fig_config_single_col["ylabel_config"])
    ax.tick_params(axis="x", **fig_config_single_col["xticks_config"])
    ax.tick_params(axis="y", **fig_config_single_col["yticks_config"])

    ax.legend(title="Token Pair", loc="lower left", **fig_config_single_col["legend_config"])
    ax.grid(**fig_config_single_col["grid_config"])
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    plt.tight_layout()

    fname = f"nist_vs_clf_scatter_two_categories_ts{clean_name(train_size)}_{clean_name(clf)}.pdf"
    plt.savefig(save_dir / fname, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {fname}")


def compare_nist_perf_with_classsif_accuracy(nist_perfs_dict, xp_config, save_path, fake_data=None, batch_size_focus=1):
    """
    Generates comparison plots between NIST success rates and Classifier accuracy.

    Args:
        nist_perfs_dict (dict): Dictionary of NIST success rates.
        xp_config (dict): Experiment configuration containing paths.
        save_path (str or Path): Directory to save the PDF figures.
        fake_data (tuple, optional): (loaded_dict, loaded_models_idx) to bypass loading.
        batch_size_focus (int, optional): Batch size to focus on. Default is 1.
    """

    # Ensure save directory exists
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    loaded_dict = {}
    loaded_models_idx = {}

    if fake_data:
        loaded_dict, loaded_models_idx = fake_data
    elif xp_config.get("train_size_dict_path", None) is not None:
        if isinstance(xp_config["train_size_dict_path"], dict):
            assert len(xp_config["train_size_dict_path"]) == 1
            for path, label in xp_config["train_size_dict_path"].items():
                loaded_dict, loaded_models_idx = load_train_size_dict(path, label)
        else:
            path, label = xp_config["train_size_dict_path"]
            loaded_dict, loaded_models_idx = load_train_size_dict(path, label)
    else:
        logger.warning("No train_size_dict_path found in xp_config.")
        return

    logger.info(f"{len(loaded_models_idx)} models loaded.")

    # 2. Parse NIST Data
    nist_data = parse_nist_data(nist_perfs_dict)
    logger.info(f"Parsed NIST data for {len(nist_data)} original models")

    # 3. Iterate over Train Sizes and Classifiers
    for train_size, inner_dict in loaded_dict.items():
        if not inner_dict:
            continue

        single_key = list(inner_dict.keys())[0]
        summary = inner_dict[single_key]

        if not summary:
            continue

        next_single_key = list(summary.keys())[0]

        # Focus on specified batch_size
        if batch_size_focus not in summary[next_single_key]:
            logger.warning(f"Batch size {batch_size_focus} not found for train_size={train_size}")
            continue

        bs_data = summary[next_single_key][batch_size_focus]

        first_tp = list(bs_data.keys())[0]
        clfs = list(bs_data[first_tp].keys())

        for clf in clfs:
            logger.info(f"{'='*80}")
            logger.info(f"Processing: train_size={train_size}, clf={clf}, batch_size={batch_size_focus}")
            logger.info(f"{'='*80}")

            # PLOT 4: Scatter Plot (original with specific TPs)
            plot_scatter_accuracy_vs_nist(bs_data, clf, loaded_models_idx, nist_data, train_size, save_dir)

            # PLOT 5: Scatter Plot (two categories: '0-1' and avg of others)
            plot_scatter_accuracy_vs_nist_two_categories(
                bs_data, clf, loaded_models_idx, nist_data, train_size, save_dir
            )

            logger.info(f"{'='*80}")
