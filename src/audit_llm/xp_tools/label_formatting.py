"""Label and name formatting utilities for experiment configuration."""

from typing import Dict, Optional

import numpy as np

from audit_llm.data_transforms import revert_dictionary


# Constants for column name abbreviations
TRUNC_COL_NAMES = {
    "token_pairs": "tp",
    "system_prompt_idx": "sp",
    "frequency_penalty": "fp",
    "temperature": "temp",
    "no_repeat_for_each": "nr",
    "none": "none",
    # 'aggregated_repeat_for_each':'agg_r', # a priori useless
    "models": "model",
    "features": "feature",
}

SP_IDX_TO_ICML_SP_IDX = {"0": "1", "3": "2", "6": "3", "7": "4"}

COLS_NOT_IN_MAIN_DATASET_DF = ["token_pairs", "models", "features"]


def calculation_item_namer(calculations_config, idx, item):
    """
    Args:
    - idx: idx of calculations in xp_config e.g. 'iterator_1'
    - item: value of the calculation item, e.g. 'cat-pow' (a token pair)
    Returns: e.g. looks like sp-12_temp-0.65_tp-Fail-Ition
    """
    return assemble_iterator_name_and_value(calculations_config[idx], item)


def assemble_iterator_name_and_value(iterator_name, value):
    return f"{TRUNC_COL_NAMES[iterator_name]}-{value}"


def get_calculation_item_name(calculations_config, calculation_item):
    """
    calculation_item {iterator_idx: iterator} iterator_idx corresponds to token_pairs, iterator_item is a token_pair
    Sorting is same as in TRUNC_COL_NAMES.
    Returns: e.g. looks like sp-12_temp-0.65_tp-Fail-Ition
    """
    if calculation_item == {"all": None}:
        return "all"
    else:
        # reordering key values of calculation_item to have same way each time.
        calculations_config_reverted = revert_dictionary(calculations_config)  # {iterator_name: iterator_idx}
        return "_".join(
            [
                assemble_iterator_name_and_value(
                    iterator_name_sorted, calculation_item[calculations_config_reverted[iterator_name_sorted]]
                )
                for iterator_name_sorted in list(TRUNC_COL_NAMES.keys())
                if (iterator_name_sorted in calculations_config_reverted)
                and (
                    calculations_config_reverted[iterator_name_sorted] in calculation_item
                )  # can happen that it is not inside for good reason, follow e.g. SPImpactCross
            ]
        )


def relabel_y_labels(y_labels, label_map: Optional[Dict] = None):
    if label_map is None:
        label_map = {old: new for new, old in enumerate(np.unique(y_labels))}

    y_relabed = np.array([label_map[label] for label in y_labels])

    return y_relabed, label_map


def origin_label(labels_relabeled, y_O_K_label_map):
    inv_map = {v: k for k, v in y_O_K_label_map.items()} | {-1: -1}  # Ensure that -1 (unknown) maps to -1
    return np.array([inv_map[label] for label in labels_relabeled])


def format_tp_group_name_label(tp_group_name: str) -> str:
    if tp_group_name == "0-1":
        return "0-1"
    elif tp_group_name == "Monochar":
        return "Monochar"
    elif tp_group_name == "FLiPS":
        return "FLIPS"
    elif tp_group_name == "no_token_pairs":
        return "LLMmap-IF"
    else:
        return tp_group_name


def set_aggregation_mention(values: list, value_type: str):
    if value_type == "datasets":
        dataset_name1, dataset_name2, n_Renyi, p_Renyi, end_prompt = values[0].split("_")
        single_mention = f"(n,p)=({n_Renyi}, {p_Renyi})"
        all_mention = "(n,p) = Mean"  # This line was being overridden
    elif value_type == "temperatures":
        single_mention = f"T={values[0]}"
        all_mention = "averaged on all temperatures"
    elif value_type == "models":
        single_mention = f"{values[0]}"
        all_mention = "averaged on all models"
    else:
        raise ValueError("Unknown value_type for set_aggregation_mention")

    if len(values) > 1:
        aggregation_mention = all_mention  # Handles multiple values
    elif len(values) == 1:
        aggregation_mention = single_mention
    else:
        raise ValueError(f"{values} is empty and should not be", single_mention)

    return aggregation_mention


def set_interquantile_mention(q1, q2):
    if (q1, q2) == (0.25, 0.75):
        return "Interquartile Range"
    else:
        return f"Interquantile Range {q1*100:.0f}%-{q2*100:.0f}%"


def clean_feat_label(label: str) -> str:

    label = label.replace("_ts", "")
    label = label.replace("non overlapping", "non overlap")
    label = label.replace("overlapping patterns", "overlap")
    label = label.replace("frequency", "freq")
    label = label.replace("_75", "")
    label = label.replace("_", " ")
    return label


def put_uppercase_first(string):
    if string.lower() == "fail_mean":
        return "Failure Rate"
    if not string:
        return string
    return string[0].upper() + string[1:]
