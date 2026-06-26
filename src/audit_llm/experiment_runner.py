"""Experiment orchestration and running logic (extracted from Analysis_Classes.py).

This module contains the experiment runner function and experiment function mapping.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from audit_llm.Classification import (
    batch_classification_across_token_pairs,
    classify,
    classify_cross_token_pairs,
)
from audit_llm.Classification.Feature_Visualization import (
    Nist_perf_chart,
    Save_pv_in_parquet,
    Seq_Length_visualization,
    Valid_count_chart,
    feature_space_visualization,
)
from audit_llm.file_io import write_dict_on_file
from audit_llm.xp_tools.logging_utils import (
    setup_experiment_logging,
    teardown_experiment_logging,
)
from audit_llm.xp_init_fun import (
    build_calculation_iterators,
    compute_model_variations_indices,
    extract_global_samples_indices,
    integrate_nist_test_parameters,
    prepare_token_pairs,
    prepare_models,
)


logger = logging.getLogger(__name__)


# Lazy import for LLMmap_classification to avoid requiring pytorch_lightning
# Will be imported when actually used in run_xp() if needed
def _get_llmmap_classification():
    """Lazy loader for LLMmap_classification to avoid import-time dependency on pytorch_lightning."""
    from Fingerprinting_methods.LLMmap.cross_classif_task.main_classif import LLMmap_classification

    return LLMmap_classification


# Experiment function mapping constant
EXPERIMENT_FUNCTION_MAP = {
    #'F8_Feature_Visualization': F8_Feature_Visualization,
    "Nist_perf_chart": Nist_perf_chart,
    "Seq_Length_visualization": Seq_Length_visualization,
    "Valid_count_chart": Valid_count_chart,
    "classify": classify,
    "classify_cross_token_pairs": classify_cross_token_pairs,
    "batch_classification_across_token_pairs": batch_classification_across_token_pairs,
    "Batch_Classification_across_token_pairs": batch_classification_across_token_pairs,  # legacy name
    "Save_pv_in_parquet": Save_pv_in_parquet,
    "feature_space_visualization": feature_space_visualization,
    # LLMmap_classification will be added dynamically if needed
}


def _xp_config_init(
    xp_config: Dict,
    token_pairs_set: List[str],
    Experiments_path: str,
    intra_samples_feature_index_dict: Optional[Dict] = None,
    MainDataset_df=None,
    Answers_df=None,
    model_idx: Optional[Dict] = None,
) -> tuple:
    """Initialize experiment configuration and prepare calculation iterators.

    Unified initialization function that works for both FLiPS experiments and LLMmap.
    For LLMmap, pass intra_samples_feature_index_dict=None.

    Args:
        xp_config: Experiment configuration dictionary
        token_pairs_set: Set of token pairs for dataset filtering
        Experiments_path: Path to experiments directory
        intra_samples_feature_index_dict: Feature index dict (None for LLMmap)
        MainDataset_df: Main dataset DataFrame
        Answers_df: Answers DataFrame
        model_idx: Model index dictionary

    Returns:
        Tuple of (calculations_iter_lists, token_pairs_banned_path, models,
                  datasets, global_samples_indices, model_variations_indices,
                  quantized_model_variations_indices, abliterated_variation)

        ``abliterated_variation`` is the scenario-driven
        ``(temperature, system_prompt_idx)`` pin for abliterated models, or
        ``None`` on the legacy path (where ``prepare_dataset_features`` falls
        back to its hardcoded pin).
    """
    if "features" in xp_config:
        xp_config = integrate_nist_test_parameters(xp_config)

    # For LLMmap experiments, datasets will be None
    is_llmmap = intra_samples_feature_index_dict is None

    # Section 1: Prepare datasets and models
    if not is_llmmap:
        datasets = prepare_token_pairs(xp_config, token_pairs_set, Experiments_path)
    else:
        datasets = None

    models = prepare_models(xp_config, model_idx)
    token_pairs_banned_path = Path(Experiments_path) / "banned_token_pairs.json"

    # Section 2: Build calculation iterators
    calculations_iter_lists = build_calculation_iterators(
        xp_config, datasets, models, intra_samples_feature_index_dict, MainDataset_df, Answers_df
    )

    # Section 3: Extract global samples indices
    global_samples_indices = extract_global_samples_indices(xp_config, MainDataset_df)

    # Section 4: Compute model variations indices.
    # Prefer a declarative `scenario:` key (analysis-layer SSOT via build_instances);
    # fall back to the legacy model_variations / quantized_model_variations /
    # abliterated_models keys. Both paths coexist (the legacy collapse is the
    # manual headline release — see docs/reproduction/legacy-paths.md).
    scenario_path = xp_config.get("scenario", None)
    if scenario_path is not None:
        from audit_llm.scenarios.enumerator import build_analysis_variation_structures
        from audit_llm.scenarios.loader import load_scenario

        scenario = load_scenario(scenario_path)
        structures = build_analysis_variation_structures(scenario, MainDataset_df)
        model_variations_indices = structures.model_variations_indices
        quantized_model_variations_indices = structures.quantized_model_variations_indices
        # The scenario is the SSOT for abliterated selection: feed its repos to
        # prepare_dataset_features (None when empty → skip abliterated entirely).
        xp_config["abliterated_models"] = list(structures.abliterated_models) or None
        abliterated_variation = structures.abliterated_variation
    else:
        model_variations_indices = compute_model_variations_indices(xp_config, MainDataset_df)

        # Section 4b: Compute separate quantized model variations indices (if specified)
        quant_var_config = xp_config.get("quantized_model_variations", None)
        if quant_var_config is not None:
            quantized_model_variations_indices = compute_model_variations_indices(
                xp_config, MainDataset_df, model_variations=quant_var_config
            )
        else:
            quantized_model_variations_indices = {}

        abliterated_variation = None

    return (
        calculations_iter_lists,
        token_pairs_banned_path,
        models,
        datasets,
        global_samples_indices,
        model_variations_indices,
        quantized_model_variations_indices,
        abliterated_variation,
    )


def run_xp(
    xp_config: dict,
    Answers_df,
    TokenIDs_df,
    MainDataset_df,
    token_pairs_set: List[str],
    Dataset_path: str,
    Experiments_path: str,
    intra_samples_features_dict: Optional[Dict] = None,
    inter_samples_features_map: Optional[Dict] = None,
    intra_samples_feature_index_dict: Optional[Dict] = None,
    inter_samples_feature_index_dict: Optional[Dict] = None,
    hard_datasets: list[str] = [],
):
    """Run an experiment based on configuration.

    This function orchestrates the entire experiment pipeline:
    1. Prepares models and calculation iterators
    2. Constructs Experiment_config dict
    3. Calls the appropriate experiment function from EXPERIMENT_FUNCTION_MAP

    Args:
        xp_config: Experiment configuration dictionary
        Answers_df: Answers DataFrame
        TokenIDs_df: Token IDs DataFrame
        MainDataset_df: Main dataset DataFrame
        token_pairs_set: Set of token pairs
        Dataset_path: Path to dataset
        Experiments_path: Path to experiments directory
        intra_samples_features_dict: Intra-sample features (None for LLMmap)
        inter_samples_features_map: Inter-sample features (None for LLMmap)
        intra_samples_feature_index_dict: Intra-sample feature index (None for LLMmap)
        inter_samples_feature_index_dict: Inter-sample feature index (None for LLMmap)
        hard_datasets: List of hard datasets
    """
    from audit_llm.data_loader import _prepare_models_with_PRNGs
    from audit_llm.xp_tools import UP_TO_DATE_COMPUTE_CONFIG

    # Determine if this is an LLMmap experiment
    is_llmmap = "LLMmap" in Dataset_path

    logger.info("doing analysis" if not is_llmmap else "preparing LLMmap experiment")
    experiment_fun = xp_config["experiment_fun"]

    save_fig_path = os.path.join(Experiments_path, f"{experiment_fun}", xp_config["xp_name"])
    os.makedirs(save_fig_path, exist_ok=True)

    if is_llmmap:
        datasets, models, model_idx = _prepare_models_with_PRNGs(Answers_df)

        (
            calculations_iter_lists,
            token_pairs_banned_path,
            models,
            datasets,
            global_samples_indices,
            model_variations_indices,
            quantized_model_variations_indices,
            abliterated_variation,
        ) = _xp_config_init(
            xp_config,
            token_pairs_set,
            Experiments_path,
            intra_samples_feature_index_dict=None,  # LLMmap
            MainDataset_df=MainDataset_df,
            Answers_df=Answers_df,
            model_idx=model_idx,
        )
    else:
        # Loading up-to-date data (FLiPS experiments with feature computation)
        from audit_llm.data_loader import _compute_save_load_experiments

        logger.info("starting computing_save_load_xp")

        # Get compute_config for the dataset type
        dataset_type = Dataset_path.split("/")[-2]
        compute_config = UP_TO_DATE_COMPUTE_CONFIG.copy()[dataset_type]

        if xp_config.get("set_constant_seq_length") is not None:
            compute_config["set_constant_seq_length"] = xp_config["set_constant_seq_length"]

        # Determine max_tokens from MainDataset or a default value
        max_tokens = 300  # Default, should ideally come from experiment config

        (
            intra_samples_features_dict,
            inter_samples_features_map,
            model_idx,
            intra_samples_feature_index_dict,
            inter_samples_feature_index_dict,
            compute_config,
        ) = _compute_save_load_experiments(
            Experiments_path,
            Answers_df,
            TokenIDs_df,
            MainDataset_df,
            max_tokens,
            compute_config,
            Dataset_path=Dataset_path,
        )

        (
            calculations_iter_lists,
            token_pairs_banned_path,
            models,
            datasets,
            global_samples_indices,
            model_variations_indices,
            quantized_model_variations_indices,
            abliterated_variation,
        ) = _xp_config_init(
            xp_config,
            token_pairs_set,
            Experiments_path,
            intra_samples_feature_index_dict,
            MainDataset_df,
            Answers_df,
            model_idx,
        )

    # 1. Initialize the common configuration
    # models_indices: indices of filtered models (abliterated/closed-source removed by prepare_models)
    models_indices = [model_idx[m] for m in models if m in model_idx]
    Experiment_config: Dict[str, Any] = {
        "analysis_df": Answers_df,
        "calculations_iter_lists": calculations_iter_lists,
        "global_samples_indices": global_samples_indices,
        "model_variations_indices": model_variations_indices,
        "quantized_model_variations_indices": quantized_model_variations_indices,
        "abliterated_variation": abliterated_variation,
        "model_idx": model_idx,
        "models_indices": models_indices,
        "xp_config": xp_config,
        "save_fig_path": save_fig_path,
        "DatasetPath": Dataset_path,
        "Answers_dfPath": "",  # Will be set later
        "models": models,
    }

    # 2. Add specific data if NOT LLMmap
    if not is_llmmap:
        Experiment_config["intra_samples_features_dict"] = intra_samples_features_dict
        Experiment_config["inter_samples_features_map"] = inter_samples_features_map
        Experiment_config["intra_samples_feature_index_dict"] = intra_samples_feature_index_dict
        Experiment_config["inter_samples_feature_index_dict"] = inter_samples_feature_index_dict
        Experiment_config["token_pairs"] = datasets
        Experiment_config["token_stats_dict"] = compute_config.get("token_stats_dict", {})
        Experiment_config["token_pairs_banned_path"] = token_pairs_banned_path

    if xp_config.get("save", True):
        file_path = Path(save_fig_path) / "Experiment_config.txt"
        write_dict_on_file(Experiment_config, file_path)

    # Handle lazy loading of LLMmap_classification
    if experiment_fun == "LLMmap_classification" and experiment_fun not in EXPERIMENT_FUNCTION_MAP:
        EXPERIMENT_FUNCTION_MAP["LLMmap_classification"] = _get_llmmap_classification()

    if experiment_fun in EXPERIMENT_FUNCTION_MAP:
        setup_experiment_logging(Path(save_fig_path))
        start_time = time.time()
        EXPERIMENT_FUNCTION_MAP[experiment_fun](Experiment_config)
        end_time = time.time()
        time_spent = end_time - start_time
        logger.info("TIME SPENT for %s: %.2f minutes.", xp_config["xp_name"], time_spent / 60)
        teardown_experiment_logging()
        sys.stdout.write(f"XP {xp_config['xp_name']} done successfully.\n")
        sys.stdout.flush()

    else:
        raise ValueError(f"No experiment function found for: {experiment_fun}")
