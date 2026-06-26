"""xp_tools package - Re-exports all experiment tools utilities.

This package organizes experiment tools into focused modules while maintaining
backward compatibility through re-exports.
"""

# Re-export all modules
from audit_llm.xp_tools.checkpoint_utils import *
from audit_llm.xp_tools.config_validation import *
from audit_llm.xp_tools.data_preparation import *
from audit_llm.xp_tools.token_pair_grouping import *
from audit_llm.xp_tools.feature_selection import *
from audit_llm.xp_tools.label_formatting import *
from audit_llm.xp_tools.logging_utils import *
from audit_llm.xp_tools.model_filtering import *
from audit_llm.xp_tools.results_aggregation import *
from audit_llm.xp_tools.variation_context import *

# Re-export threshold functions (moved to plotting.threshold_plots)
from audit_llm.plotting.threshold_plots import *  # noqa: F401, F403

# Re-export constants explicitly
from audit_llm.xp_tools.data_preparation import UP_TO_DATE_COMPUTE_CONFIG
from audit_llm.xp_tools.feature_selection import (
    BLOCK_FREQUENCY_MAP,
    NON_OVERLAPPING_PATTERN_MAP,
    OVERLAPPING_PATTERN_MAP,
)
from audit_llm.xp_tools.label_formatting import (
    COLS_NOT_IN_MAIN_DATASET_DF,
    SP_IDX_TO_ICML_SP_IDX,
    TRUNC_COL_NAMES,
)

__all__ = [
    # checkpoint_utils
    "load_results_checkpoint",
    "load_classification_checkpoint",
    "save_classification_checkpoint",
    "load_train_size_dict",
    "arrayify_confusion_matrices",
    "set_key_as_int",
    # config_validation
    "compare_xp_configs",
    "check_xp_config_coherence",
    "get_iter_idx_from_calculations_config",
    "is_pattern_strictly_in_string",
    # data_preparation
    "UP_TO_DATE_COMPUTE_CONFIG",
    "prepare_dataset_features",
    "get_var_models_idx",
    "get_samples_indices",
    "get_sampling_and_answers_iterators_from_Dataset",
    "process_data_per_datasets_and_nist_features",
    "prepare_experiment_context",
    "get_idx_from_dtm_index",
    "remap_model_index",
    # token_pair_grouping
    "get_token_pairs_of_group",
    "get_tp_uplets_dict_from_group",
    "get_tp_uplet_name",
    "get_tp_names_of_group",
    # feature_selection
    "OVERLAPPING_PATTERN_MAP",
    "NON_OVERLAPPING_PATTERN_MAP",
    "BLOCK_FREQUENCY_MAP",
    "get_features_from_xp_config",
    "integrate_nist_test_parameters",
    "select_nist_features",
    "select_features",
    "get_nist_features_info",
    "display_feature_index",
    "select_features_for_token_pair",
    "validate_feature_consistency",
    "remap_features_index",
    "filter_features_idx_by_nist_result_type",
    # label_formatting
    "TRUNC_COL_NAMES",
    "SP_IDX_TO_ICML_SP_IDX",
    "COLS_NOT_IN_MAIN_DATASET_DF",
    "calculation_item_namer",
    "assemble_iterator_name_and_value",
    "get_calculation_item_name",
    "relabel_y_labels",
    "origin_label",
    "format_tp_group_name_label",
    "set_aggregation_mention",
    "set_interquantile_mention",
    "clean_feat_label",
    "put_uppercase_first",
    # logging_utils
    "setup_output_logging",
    # model_filtering
    "truncate_model_name",
    "filter_models",
    "filter_token_pairs",
    "remove_closed_source_model",
    "group_models_idx_by_var_or_orig",
    "full_var_model_name_to_original_model_name",
    "full_var_model_name_to_var_name",
    "original_model_name_to_safe_var_model_idx_mapper",
    "full_var_model_name_to_full_safe_var_model_name_mapper",
    # results_aggregation
    "dict_product_with_fix_item",
    "get_actual_sp_from_sp_indices",
    "normalize_matrix",
    "aggregate_results_dict",
    # variation_context
    "VariationContext",
    "compute_model_variations_indices",
    # threshold_plots (re-exported from plotting.threshold_plots)
    "extract_thresholds",
    "plot_thresholds_distribution",
    "plot_alpha_tradeoff",
    "plot_openset_roc_curves",
    "plot_unseen_and_global_pr_vs_confidence",
    "plot_unseen_and_global_pr_vs_alpha",
    "plot_alpha_roc_curves",
    "plot_roc_curves_overlay",
]
