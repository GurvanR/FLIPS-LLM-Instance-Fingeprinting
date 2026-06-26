"""Data preparation and experiment context utilities.

This module handles data filtering, feature extraction, model variation reshaping,
and experiment context preparation.
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from audit_llm.file_io import QUANTIZATION_SEPARATOR
from audit_llm.models_management.model_names import ABLITERATED_MODELS, PRNG_MODELS
from audit_llm.data_transforms import revert_dictionary
from audit_llm.xp_tools.config_validation import check_xp_config_coherence
from audit_llm.xp_tools.feature_selection import select_features_for_token_pair
from audit_llm.xp_tools.label_formatting import COLS_NOT_IN_MAIN_DATASET_DF
from audit_llm.xp_tools.logging_utils import setup_output_logging

logger = logging.getLogger(__name__)


# Configuration constant
UP_TO_DATE_COMPUTE_CONFIG = {
    "Graph_Datasets": {},
    "Bits_Datasets": {
        "minimal_seq_length": 10,  # The minimal number of '0' '1' ' ' ',' in sequences to be extracted for experiments
        "seq_length_for_inter_sample_features": 300,  # The number of '0' and '1' in sequences to be analyzed for inter_sample_feautures, seq will be truncated to this amount.
        "features": "DefaultConfig",
    },
}


def prepare_dataset_features(
    token_pair: str,
    calculation_item: Dict,
    intra_samples_features_dict: Dict,
    intra_samples_feature_index_dict: Dict,
    selected_features: Dict[str, int],
    new_models_idx: Dict[int, str],
    models_indices: List[int],
    xp_config: Dict,
    Experiment_config,
    MainDataset_df_iterators,
    Answers_df,
    verbose=False,
) -> Tuple[np.ndarray, Dict]:
    """
    Prepare and transform feature data for a single token_pair and filtered according to its iterators from calculations in xp_config (samples_indices) and
    from its specifc values put in xp_config (global_samples_indices).
    Here intra_samples_features_matrix contains nan values because not all models have the maximum number of iterations (N_iter),
    because some sequences are not passing the minimum number of bits required basically.

    model_variations_indices: Dict {item: indices of models axis} where item is from xp_config['model_variations'] (e.g. temperature)
    """

    # Shape: (N_total_samples, N_models, N_features)
    X = intra_samples_features_dict[token_pair]

    min_seq = xp_config.get("min_seq_length", None)
    if min_seq is not None:
        seq_length_idx: int = intra_samples_feature_index_dict[token_pair]["seq_length"]
        seq_length = X[:, :, seq_length_idx]  # shape: (N_iter, len(models))
        mask = seq_length < min_seq  # shape: (N_iter, len(models))
        mask = mask[..., np.newaxis]  # shape: (N_iter, len(models), 1)
        X = np.where(mask, np.nan, X)  # mask now broadcastable to X's shape

    # Filtering samples
    all_sampling_iterators, all_answers_df_iterators = get_sampling_and_answers_iterators_from_Dataset(
        MainDataset_df_iterators, Answers_df
    )

    ## Filtering by iterators
    samples_indices, models_indices, features_list = get_samples_indices(
        calculation_item,
        xp_config,
        models_indices,
        all_sampling_iterators,
        MainDataset_df_iterators,
        Experiment_config,
        selected_features,
        intra_samples_feature_index_dict,
        token_pair,
    )

    assert features_list is not None, "features_list is None"
    # selected_features_for_classification ={feature_name: feature_idx} where feature_idx corresponds to index on X
    X = X[np.ix_(samples_indices, models_indices, features_list)]

    n_models = X.shape[1]
    # Reshaping if model_variations is not empty
    model_variations_indices: Dict = Experiment_config["model_variations_indices"]
    quantized_model_variations_indices: Dict = Experiment_config.get("quantized_model_variations_indices", {})

    # Build model name lookup: remap_model_index gives {local_pos: model_name}
    remapped = remap_model_index(Experiment_config["model_idx"], models_indices)

    # Identify PRNG model positions within the sliced X so they can be handled
    # separately (no variation expansion — one class per PRNG model).
    prng_config_val = xp_config.get("PRNGs", None)
    prng_positions: Dict[int, str] = {}  # {local_pos_in_X: prng_name}
    if prng_config_val is not None:
        for pos, name in remapped.items():
            if name in PRNG_MODELS:
                prng_positions[pos] = name

    if model_variations_indices:
        new_models_idx = {}

        # Map absolute sample indices to relative positions within the sliced X
        abs_to_rel = {abs_idx: rel_idx for rel_idx, abs_idx in enumerate(samples_indices)}

        # Classify non-PRNG models into base vs quantized positions
        base_positions: List[int] = []
        quantized_positions: List[int] = []
        for model_pos in range(n_models):
            if model_pos in prng_positions:
                continue
            if QUANTIZATION_SEPARATOR in remapped[model_pos]:
                quantized_positions.append(model_pos)
            else:
                base_positions.append(model_pos)

        # Determine which variation set quantized models use
        use_separate_quant = bool(quantized_model_variations_indices)
        quant_var = quantized_model_variations_indices if use_separate_quant else model_variations_indices

        num_base_variations = len(model_variations_indices)
        num_quant_variations = len(quant_var)

        # Each variation selects a subset of samples; use the subset size for the first axis
        first_variation_size = len(next(iter(model_variations_indices.values())))
        if use_separate_quant and quant_var:
            quant_first_size = len(next(iter(quant_var.values())))
            assert first_variation_size == quant_first_size, (
                f"Base and quantized variations must have the same sample count per variation: "
                f"base={first_variation_size}, quantized={quant_first_size}"
            )

        total_variation_classes = len(base_positions) * num_base_variations + len(quantized_positions) * num_quant_variations
        extended_shape = (first_variation_size, total_variation_classes, X.shape[2])
        X_extended = np.empty(extended_shape)
        X_extended[:] = np.nan

        # Fill X_extended with a running column counter
        col = 0
        # Phase 1: Base (non-quantized) models
        for model_pos in base_positions:
            for variation_name, sample_indices_for_variation in model_variations_indices.items():
                new_models_idx[col] = f"{remapped[model_pos]}_{variation_name}"
                relative_indices = [abs_to_rel[idx] for idx in sample_indices_for_variation if idx in abs_to_rel]
                X_extended[:, col, :] = X[np.ix_(relative_indices, [model_pos], range(X.shape[2]))].squeeze(axis=1)
                col += 1
        # Phase 2: Quantized models (with their own variation grid if specified)
        for model_pos in quantized_positions:
            for variation_name, sample_indices_for_variation in quant_var.items():
                new_models_idx[col] = f"{remapped[model_pos]}_{variation_name}"
                relative_indices = [abs_to_rel[idx] for idx in sample_indices_for_variation if idx in abs_to_rel]
                X_extended[:, col, :] = X[np.ix_(relative_indices, [model_pos], range(X.shape[2]))].squeeze(axis=1)
                col += 1

        X = X_extended

        # Append PRNG models as standalone classes (no variation expansion).
        # Use the first first_variation_size absolute dataset indices as canonical samples;
        # PRNG data is random and does not depend on temperature / system_prompt.
        if prng_positions:
            original_X = intra_samples_features_dict[token_pair]
            prng_sample_abs_indices = samples_indices[:first_variation_size]
            for pos, prng_name in sorted(prng_positions.items()):
                prng_global_idx = models_indices[pos]
                X_prng = original_X[np.ix_(prng_sample_abs_indices, [prng_global_idx], features_list)]
                X = np.concatenate((X, X_prng), axis=1)
                new_models_idx[len(new_models_idx)] = prng_name

    # Adding abliterated models if they exist
    # None → skip entirely, [] → use all ABLITERATED_MODELS, [names] → use specific models
    abliterated_models_config = xp_config.get("abliterated_models", None)
    logger.debug(f"abliterated_models_config from xp_config: {abliterated_models_config}")
    if abliterated_models_config is not None:
        abliterated_models = abliterated_models_config if abliterated_models_config else ABLITERATED_MODELS
        orginal_model_idx: Dict[str, int] = Experiment_config["model_idx"]
        abliterated_models_indices = [
            orginal_model_idx[model_name] for model_name in abliterated_models if model_name in orginal_model_idx
        ]

        if abliterated_models_indices:
            main_dataset = MainDataset_df_iterators

            # Abliteration-as-resolver: on the scenario path the abliterated
            # (temperature, system_prompt_idx) come from the scenario's
            # abliteration group; otherwise fall back to the legacy hardcoded pin
            # (kept physically present — its deletion is the manual headline
            # release; see docs/reproduction/legacy-paths.md).
            abliterated_variation = Experiment_config.get("abliterated_variation", None)
            if abliterated_variation is not None:
                ablit_temp, ablit_sp = abliterated_variation
                abliterated_samples_indices = main_dataset.filter(
                    (pl.col("temperature") == ablit_temp) & (pl.col("system_prompt_idx") == ablit_sp)
                )["Index"].to_list()
            else:
                # Legacy hardcoded pin (bypassed on the scenario path).
                abliterated_samples_indices = main_dataset.filter(
                    (pl.col("temperature") == 1.0) & (pl.col("system_prompt_idx") == -1)
                )["Index"].to_list()
            abliterated_samples_indices = list(set(abliterated_samples_indices) & set(samples_indices))
            abliterated_samples_indices.sort()
            abliterated_samples_indices = [
                samples_indices.index(abliterated_idx) for abliterated_idx in abliterated_samples_indices
            ]
            if model_variations_indices:
                assert len(abliterated_samples_indices) == len(
                    next(iter(model_variations_indices.values()))
                ), f"abliterated_samples_indices length {len(abliterated_samples_indices)} does not match model_variations_indices length {len(next(iter(model_variations_indices.values())))}"

            original_X = intra_samples_features_dict[token_pair]
            X_abliterated_all_indices = original_X[:, abliterated_models_indices, :]
            nan_percentage = np.isnan(X_abliterated_all_indices).mean() * 100
            logger.debug(f"Abliterated Models all indices - NaN values percentage: {nan_percentage:.2f}")

            X_abliterated = original_X[np.ix_(abliterated_samples_indices, abliterated_models_indices, features_list)]
            # show Nan values percentage in X_abliterated for debug
            nan_percentage = np.isnan(X_abliterated).mean() * 100
            logger.debug(f"Abliterated Models - NaN values percentage: {nan_percentage:.2f}%")

            # Concatenate X and X_abliterated
            X = np.concatenate((X, X_abliterated), axis=1)

            # Update new_models_idx accordingly
            current_model_count = len(new_models_idx)
            for model_name in abliterated_models:
                if model_name in orginal_model_idx:
                    new_models_idx[current_model_count] = "_".join([model_name, "ablit"])
                    current_model_count += 1

            # Verification
            if model_variations_indices:
                expected_models = total_variation_classes + len(abliterated_models_indices) + len(prng_positions)
                assert X.shape[1] == expected_models, f"Shape mismatch: Expected {expected_models} models, got {X.shape[1]}"

    assert len(X.shape) == 3, f"{X.shape = }"

    return X, new_models_idx


def get_var_models_idx(Experiment_config, model_idx, verbose: bool = True):

    # Reshaping if model_variations is not empty
    model_variations_indices: Dict = Experiment_config["model_variations_indices"]

    if model_variations_indices:

        # Build new model index dictionary for how models now appear in X
        var_models_idx = {}
        counter = 0

        for item_name in model_variations_indices.keys():
            for non_var_idx, model_name in model_idx.items():
                new_name = f"{model_name}_{item_name}"
                var_models_idx[counter] = new_name
                counter += 1

        model_idx = var_models_idx
        if verbose:
            logger.info("Variation Models and their new indices :")
            for new_idx, model_name in model_idx.items():
                logger.info(f"{new_idx}: {model_name}")
    xp_config = Experiment_config["xp_config"]

    return model_idx


def get_samples_indices(
    calculation_item: Dict[str, Any],
    xp_config: Dict[str, Any],
    models_indices: List[int],
    all_sampling_iterators: List[str],
    MainDataset_df_iterators: pl.DataFrame,
    Experiment_config: Dict[str, Any],
    selected_features: Optional[Dict[str, int]] = None,
    intra_samples_feature_index_dict: Optional[Dict[str, Any]] = None,
    token_pair: Optional[str] = None,
) -> Tuple[List[int], List[int], Optional[List[int]]]:
    """
    Filters the MainDataset based on calculation items and computes the intersection
    of sample indices. Also updates models_indices and features_list if specific
    keys are present.
    """
    if selected_features is not None:
        features_list = list(selected_features.values())
    else:
        features_list = None

    global_samples_indices: List[int] = Experiment_config["global_samples_indices"]

    if calculation_item == {"all": None}:  # means empty calculation_item, happens if calculations in xp_config is None
        samples_indices = global_samples_indices
    else:
        for idx, item in calculation_item.items():
            it_name = xp_config["calculations"][idx]
            if it_name in all_sampling_iterators:
                MainDataset_df_iterators = MainDataset_df_iterators.filter(pl.col(it_name) == item)
            else:
                assert it_name in COLS_NOT_IN_MAIN_DATASET_DF, it_name  # assert it does not go beyond that case, case we handle well already.
                if it_name == "models":
                    model_idx = Experiment_config["model_idx"]
                    models_indices = [
                        model_idx[item]
                    ]  # replace model_indices by a single idx for a single model 'item' (useful within nist perf computations, will not have model variations and it_name=='models' at same time.)
                elif it_name == "features":
                    assert intra_samples_feature_index_dict is not None
                    assert token_pair is not None
                    features_list = [
                        intra_samples_feature_index_dict[token_pair][item]
                    ]  # a single feature for a single feature
                else:
                    pass  # do nothing if token_pairs case as it is handled separately

        samples_indices: List[int] = MainDataset_df_iterators["Index"].to_list()
        samples_indices = list(set(global_samples_indices) & set(samples_indices))

    samples_indices.sort()

    return samples_indices, models_indices, features_list


def get_sampling_and_answers_iterators_from_Dataset(MainDataset_df: pl.DataFrame, Answers_df: pl.DataFrame):

    all_sampling_iterators: List[str] = [col for col in MainDataset_df.columns if col != "Index"]

    all_answers_df_iterators: List[str] = [
        col
        for col in Answers_df.columns
        if col not in all_sampling_iterators and col not in ["Dataset_Question Index", "Answer", "Model"]
    ]
    return all_sampling_iterators, all_answers_df_iterators


def process_data_per_datasets_and_nist_features(
    nist_tests_result_type: List,
    datasets: List,
    temp_save_fig_path: Path,
    xp_config: Dict,
    intra_samples_feature_index_dict: Dict,
    intra_samples_features_dict: Dict,
    models_idx: List,
    dtm_index: Dict,
    temperature: Optional[float] = None,
    temperatures_index: Optional[List[int]] = None,
    min_seq_length_for_nist_perf: Optional[int] = None,
) -> tuple[dict, dict]:
    """
    Processes multiple datasets to select features, prepare them, and validate consistency.

    Args:
        nist_tests_result_type (list): NIST test result type (currently supports only one).
        datasets (list): Dataset names to process.
        temp_save_fig_path (str): Base path for temporary figure outputs.
        xp_config (dict): Experiment configuration.
        intra_samples_feature_index_dict (dict): Maps intra-sample feature names to indices.
        intra_samples_features_dict (dict): Intra-sample features data.
        models_idx (list): Model indices for feature preparation.
        dtm_index (list): DTM (Data Transformation Matrix) index.
        temperature (float): Temperature for feature preparation.

    Returns:
        tuple[dict, dict, dict]:
            - X_s (dict): Prepared feature matrices per token_pair.
            - selected_features_dataset_dict (dict): Selected NIST features per token_pair.
            - remapped_nist_selected_features (dict): Remapped NIST selected features.

    Raises:
        ValueError: If `nist_tests_result_type` contains more than one element.
    """

    X_s = {}
    selected_features_dataset_dict = {}

    if len(nist_tests_result_type) != 1:
        raise ValueError("Error: Only one nist_tests_result_type supported at a time.")

    for token_pair in datasets:
        # Setup output logging for the current token_pair
        outputs_folder_path = setup_output_logging(temp_save_fig_path, token_pair, xp_config)

        # Select NIST-only features for this token_pair
        selected_features = select_features_for_token_pair(
            intra_samples_feature_index_dict,
            xp_config,
            token_pair,
            nist_only=True,
        )
        selected_features_dataset_dict[token_pair] = selected_features

        logger.info(f"Selected features for token_pair '{token_pair}': {selected_features_dataset_dict[token_pair]}")

        # Prepare token_pair features for analysis
        # X_s[token_pair] = prepare_dataset_features(...)

    # Remap and validate feature consistency
    from audit_llm.xp_tools.feature_selection import (
        remap_features_index,
        validate_feature_consistency,
    )

    validate_feature_consistency(selected_features_dataset_dict)
    remapped_nist_selected_features = remap_features_index(selected_features_dataset_dict[datasets[0]])

    return X_s, selected_features_dataset_dict, remapped_nist_selected_features


def prepare_experiment_context(
    Experiment_config: Dict[str, Any],
    verbose: bool = True,
):
    """Prepare common experiment setup shared by classify and classify_cross_token_pairs.
    
    Returns an ExperimentContext (or FLiPSExperimentContext for FLiPS experiments)
    dataclass instead of the 11-element tuple.
    
    Args:
        Experiment_config: Experiment configuration dictionary from run_xp()
        verbose: Whether to print verbose output
    
    Returns:
        ExperimentContext or FLiPSExperimentContext dataclass
    """
    from audit_llm.experiment_context import ExperimentContext, FLiPSExperimentContext
    from audit_llm.xp_tools.variation_context import VariationContext
    
    intra_samples_features_dict = Experiment_config.get("intra_samples_features_dict", None)
    intra_samples_feature_index_dict = Experiment_config.get("intra_samples_feature_index_dict", None)
    xp_config = Experiment_config["xp_config"]

    # Extracting config parameters
    calculations_iter_lists = Experiment_config.get("calculations_iter_lists", {})
    save_fig_path = Path(Experiment_config["save_fig_path"])
    models_indices = Experiment_config.get("models_indices", None)
    new_models_idx = Experiment_config.get("new_var_models_idx", {})
    model_idx = Experiment_config.get("model_idx", {})
    models = Experiment_config.get("models", [])
    global_samples_indices = Experiment_config.get("global_samples_indices", [])
    model_variations_indices = Experiment_config.get("model_variations_indices", {})
    analysis_df = Experiment_config.get("analysis_df")

    if verbose:
        logger.info(f"Total number of models (new_models_idx): {len(new_models_idx)}")

    # Checkpoint dir path - might be in xp_config or generated from save_fig_path
    checkpoint_dir_path = xp_config.get("checkpoint_dir_path", None)
    if checkpoint_dir_path is None:
        checkpoint_dir_path = save_fig_path / "checkpoints"
    else:
        checkpoint_dir_path = Path(checkpoint_dir_path)
        
    token_pairs_banned_path = Experiment_config.get("token_pairs_banned_path", None)
    if token_pairs_banned_path is not None:
        token_pairs_banned_path = Path(token_pairs_banned_path)

    # Validate xp config coherence
    checkpoint_xp_config_path = checkpoint_dir_path / "xp_config.json"
    check_xp_config_coherence(checkpoint_xp_config_path, xp_config)

    yaml_source = xp_config.get("_yaml_source_path")
    if yaml_source:
        yaml_source = Path(yaml_source)
        if yaml_source.exists():
            shutil.copy2(yaml_source, checkpoint_dir_path / yaml_source.name)

    DatasetPath = Path(Experiment_config["DatasetPath"])
    Answers_dfPath = Experiment_config.get("Answers_dfPath", "")
    if Answers_dfPath:
        Answers_dfPath = Path(Answers_dfPath)

    # Load DataFrames
    MainDataset_df_iterators = pl.read_csv(DatasetPath).clone()
    
    if Answers_dfPath and Answers_dfPath.exists():
        if Answers_dfPath.is_dir() and any(Answers_dfPath.glob("*.parquet")):
            # Per-model partitioned Parquet directory — lazy scan
            Answers_df = pl.scan_parquet(Answers_dfPath).collect()
        elif ".csv" in str(Answers_dfPath):
            Answers_df = pl.read_csv(Answers_dfPath, schema_overrides={"Answer": pl.String})
        elif ".parquet" in str(Answers_dfPath):
            Answers_df = pl.scan_parquet(Answers_dfPath).collect()
        else:
            # Use analysis_df if available
            Answers_df = analysis_df if analysis_df is not None else pl.DataFrame()
    else:
        Answers_df = analysis_df if analysis_df is not None else pl.DataFrame()

    # Create VariationContext if model_variations exist
    quantized_model_variations_indices = Experiment_config.get("quantized_model_variations_indices", {})
    if model_variations_indices:
        # Extract variations structure from model_variations_indices keys
        # This is a simplified approach - in reality we'd need to reconstruct from the keys
        variation_context = VariationContext(
            base_models=[m for m in models if m not in model_variations_indices],
            variations={},  # TODO: reconstruct from model_variations_indices if needed
            abliterated_models=[],  # TODO: extract if needed
            model_variations_indices=model_variations_indices,
            quantized_model_variations_indices=quantized_model_variations_indices,
            n_classes=len(new_models_idx)
        )
    else:
        variation_context = None

    # Determine if this is a FLiPS experiment (has feature dicts)
    is_flips = intra_samples_features_dict is not None

    if is_flips:
        # Return FLiPSExperimentContext
        return FLiPSExperimentContext(
            # Base ExperimentContext fields
            xp_config=xp_config,
            save_fig_path=save_fig_path,
            calculations_iter_lists=calculations_iter_lists,
            models=models,
            model_idx=model_idx,
            global_samples_indices=global_samples_indices,
            variation_context=variation_context,
            analysis_df=analysis_df if analysis_df is not None else Answers_df,
            checkpoint_dir_path=checkpoint_dir_path,
            DatasetPath=DatasetPath,
            Answers_dfPath=Answers_dfPath,
            token_pairs_banned_path=token_pairs_banned_path,
            models_indices=models_indices if models_indices is not None else list(model_idx.values()),
            new_var_models_idx=new_models_idx,
            MainDataset_df_iterators=MainDataset_df_iterators,
            Answers_df=Answers_df,
            # FLiPS-specific fields
            intra_samples_features_dict=intra_samples_features_dict,
            inter_samples_features_map=Experiment_config.get("inter_samples_features_map", {}),
            intra_samples_feature_index_dict=intra_samples_feature_index_dict,
            inter_samples_feature_index_dict=Experiment_config.get("inter_samples_feature_index_dict", {}),
            token_pairs=Experiment_config.get("token_pairs", []),
            token_stats_dict=Experiment_config.get("token_stats_dict", {}),
        )
    else:
        # Return base ExperimentContext (for LLMmap or other experiments)
        return ExperimentContext(
            xp_config=xp_config,
            save_fig_path=save_fig_path,
            calculations_iter_lists=calculations_iter_lists,
            models=models,
            model_idx=model_idx,
            global_samples_indices=global_samples_indices,
            variation_context=variation_context,
            analysis_df=analysis_df if analysis_df is not None else Answers_df,
            checkpoint_dir_path=checkpoint_dir_path,
            DatasetPath=DatasetPath,
            Answers_dfPath=Answers_dfPath,
            token_pairs_banned_path=token_pairs_banned_path,
            models_indices=models_indices if models_indices is not None else list(model_idx.values()),
            new_var_models_idx=new_models_idx,
            MainDataset_df_iterators=MainDataset_df_iterators,
            Answers_df=Answers_df,
        )



def get_idx_from_dtm_index(
    dtm_index: dict,
    datasets: Optional[List] = None,
    temperatures: Optional[List] = None,
    models: Optional[List] = None
):
    """
    Extract indices from dtm_index for datasets, temperatures, and models.
    
    Fixed mutable default arguments (was datasets=[], temperatures=[], models=[]).
    """
    # None sentinels avoid shared mutable defaults across calls
    if datasets is None:
        datasets = []
    if temperatures is None:
        temperatures = []
    if models is None:
        models = []
        
    if datasets:
        datasets_idx = [dtm_index["datasets"][token_pair] for token_pair in datasets]
    else:
        datasets_idx = list(dtm_index["datasets"].values())
    datasets_idx.sort()

    if temperatures:
        temperatures_idx = [dtm_index["temperatures"][temp] for temp in temperatures]
    else:
        temperatures_idx = list(dtm_index["temperatures"].values())
    temperatures_idx.sort()

    if models:
        models_idx = [dtm_index["models"][model] for model in models]
    else:
        models_idx = list(dtm_index["models"].values())
    models_idx.sort()

    return datasets_idx, temperatures_idx, models_idx


def remap_model_index(model_idx, models_indices):
    """
    Returns: {model_indice: model_name}
    """
    models_dict_reverted: Dict[int, str] = revert_dictionary(model_idx)
    new_models_indices = {
        new_model_indice: models_dict_reverted[old_model_indice]
        for new_model_indice, old_model_indice in enumerate(models_indices)
    }
    return new_models_indices
