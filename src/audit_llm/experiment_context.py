"""Experiment context dataclasses.

This module defines typed dataclasses to replace the 11-element positional tuple
previously returned by prepare_experiment_context() and the god-object dict
Experiment_config.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from audit_llm.xp_tools.variation_context import VariationContext


@dataclass
class ExperimentContext:
    """Base experiment context replacing the 11-element tuple.
    
    Replaces the positional tuple returned by prepare_experiment_context():
    (intra_samples_features_dict, intra_samples_feature_index_dict, 
     calculations_iter_lists, models_indices, new_models_idx, save_fig_path,
     token_pairs_banned_path, xp_config, checkpoint_dir_path, 
     MainDataset_df_iterators, Answers_df)
    
    Also replaces fields from the Experiment_config god-object dict.
    
    Attributes:
        xp_config: Experiment configuration dictionary
        save_fig_path: Path for saving figures
        calculations_iter_lists: Dictionary of calculation iterators
        models: List of model names
        model_idx: Mapping from model name to index
        global_samples_indices: List of global sample indices
        variation_context: VariationContext (if model_variations exist)
        analysis_df: Main answers DataFrame (pl.DataFrame)
        checkpoint_dir_path: Path for checkpoints
        DatasetPath: Path to the main dataset CSV
        Answers_dfPath: Path to answers parquet/csv
        token_pairs_banned_path: Path to banned datasets JSON
        models_indices: List of model indices for filtering
        new_var_models_idx: Remapped model index after variations
    """
    xp_config: Dict[str, Any]
    save_fig_path: Path
    calculations_iter_lists: Dict[str, List]
    models: List[str]
    model_idx: Dict[str, int]
    global_samples_indices: List[int]
    variation_context: Optional[VariationContext]  # None if no model_variations
    analysis_df: pl.DataFrame
    checkpoint_dir_path: Path
    
    # Additional fields from original Experiment_config dict
    DatasetPath: Path
    Answers_dfPath: Path
    token_pairs_banned_path: Path
    models_indices: List[int]
    new_var_models_idx: Dict[int, str]
    
    # MainDataset and Answers DataFrames (loaded from paths)
    MainDataset_df_iterators: pl.DataFrame
    Answers_df: pl.DataFrame


@dataclass
class FLiPSExperimentContext(ExperimentContext):
    """FLiPS-specific context carrying feature matrices and inter-sample features.
    
    Used for experiments that require FLiPS (Feature-Level Probabilistic Scoring)
    feature computation. Extends ExperimentContext with feature-related fields.
    
    Attributes:
        intra_samples_features_dict: Dict[dataset: 3D array (N_iter, len(models), nb_of_features)]
        inter_samples_features_map: {dataset : {(k, feature_name): feature}} with k: model idx
        intra_samples_feature_index_dict: Dict[dataset: Dict[feature_name, feature_idx]]
        inter_samples_feature_index_dict: Dict[dataset: Dict[feature_name, feature_idx]]
        token_pairs: List of token pair names
        token_stats_dict: Token statistics per dataset
    """
    intra_samples_features_dict: Dict[str, Any]
    inter_samples_features_map: Dict[str, Dict]
    intra_samples_feature_index_dict: Dict[str, Dict[str, int]]
    inter_samples_feature_index_dict: Dict[str, Dict[str, int]]
    token_pairs: List[str]
    token_stats_dict: Dict[str, Any]
