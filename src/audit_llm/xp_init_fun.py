import logging
from typing import List, Dict, Callable, Tuple, Optional
from pathlib import Path

import polars as pl

from audit_llm.xp_tools import *

logger = logging.getLogger(__name__)


def prepare_token_pairs(xp_config: Dict, token_pairs_set: List[str], experiments_path: str) -> List[str]:
    """
    Prepare and filter token_pairs for the experiment.

    Args:
        xp_config: Experiment configuration dictionary
        token_pairs_set: Set of token pairs
        experiments_path: Path to experiments directory

    Returns:
        List of filtered token_pairs
    """
    token_pairs: List[str] = xp_config.get("token_pairs", [])

    if not token_pairs:
        token_pairs = token_pairs_set
    token_pairs_banned_path = Path(experiments_path) / "banned_token_pairs.json"
    token_pairs = filter_token_pairs(token_pairs, token_pairs_banned_path)
    if not token_pairs:
        raise ValueError(
            f"All token pairs were filtered out. Check {token_pairs_banned_path} and remove entries to allow token pairs back."
        )
    logger.debug("token_pairs = %s", token_pairs)

    return token_pairs


def prepare_models(xp_config: Dict, model_idx) -> List[str]:
    """
    Prepare and filter models for the experiment.

    Args:
        xp_config: Experiment configuration dictionary
        model_idx: Model index dictionary

    Returns:
        List of filtered and sorted models
    """
    models: List[str] = xp_config.get("models", [])

    if not models:
        models = list(model_idx.keys())
        # Filter models
        models = filter_models(models, xp_config)
        models = remove_closed_source_model(models)
    models.sort()

    # printing models
    for model in models:
        logger.debug("Model: %s", model)

    return models


def build_calculation_iterators(
    xp_config: Dict,
    token_pairs: Optional[List[str]],
    models: List[str],
    intra_samples_feature_index_dict: Optional[Dict],
    main_dataset_df,
    answers_df,
) -> Dict:
    """
    Section 2: Building calculation iterators.

    Args:
        xp_config: Experiment configuration dictionary
        token_pairs: List of token_pairs
        models: List of models
        intra_samples_feature_index_dict: Feature index dictionary
        main_dataset_df: Main dataset DataFrame
        answers_df: Answers DataFrame

    Returns:
        Dictionary of calculation iterator lists looking like {iterator_idx: calculation_list}
        e.g. {'iterator_1': ['temp1', 'temp2'], 'iterator_2': [sp1, sp2, sp3]}

    Can be empty dict if xp_config['calculations'] is empty. In this case, it means that there are model_variations
    and/or classif will be done on several sampling param at same time, or there simply no sampling parameter variation wihtin
    initial experiment.

    """
    sampling_parameters: Dict[str, List] = xp_config["sampling_parameters"]
    calculations_iterators: Optional[Dict[str, str]] = xp_config.get("calculations", None)
    all_sampling_iterators, all_answers_df_iterators = get_sampling_and_answers_iterators_from_Dataset(
        main_dataset_df, answers_df
    )

    calculations_iter_lists: Dict = {}
    if calculations_iterators is None:
        return calculations_iter_lists

    for iterator_idx, iterator_name in calculations_iterators.items():
        assert iterator_name != xp_config.get("model_variations", None)

        if iterator_name in all_answers_df_iterators + COLS_NOT_IN_MAIN_DATASET_DF:
            if iterator_name == "token_pairs":
                assert token_pairs is not None
                calculation_list = token_pairs
            elif iterator_name == "models":
                calculation_list = models
            elif iterator_name == "features":
                logger.info(
                    "Iterate over features, assume that independant of dataset/token_pair i.e. no use of token_stats."
                )
                assert intra_samples_feature_index_dict is not None

                selected_features_for_classification = select_features_for_token_pair(
                    intra_samples_feature_index_dict, xp_config
                )
                calculation_list = list(selected_features_for_classification.keys())
            else:
                raise ValueError(
                    f"Iterator name {iterator_name} found in all_answers_df_iterators not yet implemented."
                )

        elif iterator_name in all_sampling_iterators:
            calculation_list = sampling_parameters[iterator_name]
        else:
            if iterator_idx == "for_each" and iterator_name == "none":
                calculation_list = ["no_for_each"]
            else:
                raise ValueError(
                    f"Iterator name {iterator_name} not found in all_answers_df_iterators nor all_sampling_iterators."
                )

        if len(calculation_list) == 0:
            calculation_list = main_dataset_df[iterator_name].unique().to_list()

        if len(calculation_list) == 1 and iterator_idx != "for_each":
            raise ValueError(f"Calculation iterator {iterator_name} has only one value, no need to iterate over it.")

        calculations_iter_lists[iterator_idx] = calculation_list

    return calculations_iter_lists


def extract_global_samples_indices(xp_config: Dict, main_dataset_df) -> List[int]:
    """
    Section 3: Extracting indices valid regardless where the iterators are.

    Args:
        xp_config: Experiment configuration dictionary
        main_dataset_df: Main dataset DataFrame

    Returns:
        List of global sample indices
    """
    sampling_parameters: Dict[str, List] = xp_config["sampling_parameters"]
    filtered_MainDatasetdf_for_global_indices = main_dataset_df.clone()

    for name, allowed_values in sampling_parameters.items():
        # If no specific values given, include all unique ones.
        if not allowed_values:
            allowed_values = main_dataset_df[name].unique().to_list()

        # Apply filter
        filtered_MainDatasetdf_for_global_indices = filtered_MainDatasetdf_for_global_indices.filter(
            pl.col(name).is_in(allowed_values)
        )
        # Displaying values
        if name == "system_prompt_idx":
            logger.debug("SYSTEM PROMPT IDX:")
            for sp_idx, sp in get_actual_sp_from_sp_indices(allowed_values).items():
                logger.debug("%s: %s", sp_idx, sp)

    global_samples_indices: List[int] = filtered_MainDatasetdf_for_global_indices["Index"].to_list()
    assert all(isinstance(i, int) for i in global_samples_indices)

    return global_samples_indices
