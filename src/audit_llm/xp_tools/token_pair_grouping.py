"""Dataset grouping and organization utilities."""

import logging
from typing import List, Optional

from audit_llm.Bits_Generation.parsing_bits_tools import token_pair_to_string

logger = logging.getLogger(__name__)
from audit_llm.math_tools import random_combinations


def get_token_pairs_of_group(group: str, test_mode: bool = False, token_pairs=None):

    from audit_llm.Tokens_analysis.token_sampling import get_random_token_pairs_from_intersection

    if group == "no_grouping":
        assert token_pairs is not None
        return token_pairs

    if group == "FLiPS":
        group_tp_items = get_random_token_pairs_from_intersection(nb_of_uplets=30, seed=70, monochar=False)
        if test_mode:
            group_tp_items = group_tp_items[:20]
    elif group == "Monochar":
        group_tp_items = get_random_token_pairs_from_intersection(nb_of_uplets=50, seed=70, monochar=True)
        if test_mode:
            group_tp_items = group_tp_items[:10]
    elif group == "0-1":
        group_tp_items = [["0", "1"]]
    else:
        raise ValueError(f"Unknown group: {group}. Available groups: FLiPS, Monochar, 0-1.")

    group_token_pairs = [token_pair_to_string(items) for items in group_tp_items]

    if token_pairs is None:
        return group_token_pairs
    else:
        return [dataset for dataset in token_pairs if dataset in group_token_pairs]


def get_tp_uplets_dict_from_group(
    group: str,
    max_nb_of_uplet=3,
    token_pairs=None,
    batch_prediction_sizes: Optional[List[int]] = None,
    unique_elements: Optional[int] = 2,
):
    """
    ds_uplet_dict: {bs: [[dataset_A, dataset_C, dataset_F], [dataset_B, dataset_D, dataset_G], ...]]}
    """
    # None sentinel avoids a shared mutable default list across calls
    if batch_prediction_sizes is None:
        batch_prediction_sizes = list(range(2, 9))

    group_token_pairs = get_token_pairs_of_group(group, token_pairs=token_pairs)
    logger.debug("group_token_pairs = %s", group_token_pairs)

    if group == "Monochar":
        max_nb_of_uplet = int(max_nb_of_uplet / 2)

    ds_uplet_dict = {
        bs: random_combinations(
            iterable=group_token_pairs, combination_size=bs, num_samples=max_nb_of_uplet, unique_elements=(bs if unique_elements == 'max' else unique_elements)
        )
        for bs in batch_prediction_sizes
        if bs > 1
    }
    return ds_uplet_dict


def get_tp_uplet_name(tp_uplet: List[str]):
    return "_".join([tp_name.split("_")[0] for tp_name in tp_uplet])


def get_tp_names_of_group(tp_group_name: str, mode, bs, token_pairs=None, max_nb_of_uplet=3, unique_elements=None):
    """Get the appropriate group name based on mode."""

    if tp_group_name == "no_grouping":
        assert token_pairs is not None
        return token_pairs

    if mode == "mix_tp_at_pred":
        assert bs > 1
        # Request only the target bs to avoid eager construction of range(2, 9), which fails
        # for small token-pair sets when unique_elements='max'. Each bs is built independently,
        # so scoping to [bs] yields an identical result.
        group_ds_uplets = get_tp_uplets_dict_from_group(
            group=tp_group_name, max_nb_of_uplet=max_nb_of_uplet, unique_elements=unique_elements,
            token_pairs=token_pairs, batch_prediction_sizes=[bs],
        )[bs]
        group_ds_names = [get_tp_uplet_name(tp_uplet) for tp_uplet in group_ds_uplets]
    else:
        group_ds_names = get_token_pairs_of_group(group=tp_group_name, token_pairs=token_pairs)

    return group_ds_names
