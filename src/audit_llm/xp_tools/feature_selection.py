"""Feature selection and NIST configuration utilities."""

import functools
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

import yaml

from audit_llm.Bits_Generation.NIST_Tests_lib.Full_Nist_Testsuite import (
    make_nist_feat_name_from_info,
    make_template_pattern,
)
from audit_llm.path_utils import get_repository_level_path


# NIST test parameter maps
OVERLAPPING_PATTERN_MAP = {
    k: [{"template_pattern": template, "block_size": 75} for template in make_template_pattern(size=k, all=True)]
    for k in range(2, 6)
    # Test parameters adapted from documentation recommendations and bit size (300 to 500)
}

NON_OVERLAPPING_PATTERN_MAP = {
    k: [{"template_pattern": template, "block": 75} for template in make_template_pattern(size=k, all=True)]
    for k in range(2, 6)
    # Test parameters adapted from documentation recommendations and bit size (300 to 500)
}
BLOCK_FREQUENCY_MAP = {
    "set_1": [
        {"block_size": block_size} for block_size in [30, 100]
    ]  # we keep only two as correlations features showed they led to same values
}


@functools.lru_cache(maxsize=32)
def _load_features_config_cached(features_config_name: str) -> Dict:
    """
    Internal cached function to load feature configuration from YAML.
    
    This function is cached to avoid redundant file I/O.
    The cache key is the feature config name only.
    """
    FeaturesConfigs_path = Path(get_repository_level_path()) / "XP_configs" / "XP_config_libs" / "FeaturesConfigs.yaml"

    with open(FeaturesConfigs_path, "r") as f:
        all_features_configs = yaml.safe_load(f)

    selected_features_config = all_features_configs[features_config_name]

    return selected_features_config


def get_features_from_xp_config(xp_config: Dict) -> Dict:
    """
    Load features configuration from experiment config.
    
    This function extracts the feature config name from xp_config and delegates
    to the cached loader. Caching is based on the feature config name only.
    
    # TODO: support inline overrides.
    """
    feature_config_name = xp_config["features"]
    return _load_features_config_cached(feature_config_name)


def integrate_nist_test_parameters(xp_config):
    """
    Transforms xp_config['features'] that is primarily a name into a config a bit nested.
    """
    features_config = get_features_from_xp_config(xp_config)
    nist_tests = features_config["nist_tests"]

    for test_name, test_parameters in nist_tests.items():
        if test_name == "overlapping patterns":
            nist_tests[test_name] = [
                pattern for test_parameter in test_parameters for pattern in OVERLAPPING_PATTERN_MAP[test_parameter]
            ]
        elif test_name == "non overlapping":
            nist_tests[test_name] = [
                pattern for test_parameter in test_parameters for pattern in NON_OVERLAPPING_PATTERN_MAP[test_parameter]
            ]
        elif test_name == "block frequency":
            nist_tests[test_name] = [
                block_size for test_parameter in test_parameters for block_size in BLOCK_FREQUENCY_MAP[test_parameter]
            ]

    xp_config["features"] = features_config

    return xp_config


def select_nist_features(features_index, selected_features_config) -> Dict[str, int]:
    """
    Select from features_index those whose key contains any substring in include_keys,
    and (optionally) contains any substring in include_results.
    """

    """up_to_date_nist_features_info = get_nist_features_info(integrate_nist_test_parameters(UP_TO_DATE_COMPUTE_CONFIG_2['Bits_Datasets']))
    
    # Choper les bonnes leys de up_to_date_nist_features_info pour les mettre sur selected_nist_tests_config

    logger.debug("up_to_date_nist_features_info = %s", up_to_date_nist_features_info)
    logger.debug("selected_nist_tests_config = %s", selected_nist_tests_config)
    selected_nist_tests_config=remap_with_grouping(selected_nist_tests_config, up_to_date_nist_features_info)"""

    """selected_numbers_by_feature_type={}
    for feat_name, feat_info_list in selected_nist_items.items():
            if feat_info_list:
                selected_numbers_by_feature_type[feat_name]=[]
                for feat_info in feat_info_list: 
                    feat_info_hash=nist_info_hash_table[tuple(info for info in feat_info)]
                    selected_numbers_by_feature_type[feat_name].append(feat_info_hash)"""

    nist_tests_result_type: List[str] = selected_features_config["nist_tests_result_type"]
    results_tuple = tuple(nist_tests_result_type)

    selected_nist_tests_config = get_nist_features_info(selected_features_config)  # Dict[feat_full_name: feat_info]

    selected = {}
    for name, idx in features_index.items():

        if not any(k in name for k in tuple(selected_nist_tests_config.keys())):
            continue
        if results_tuple and not any(r in name for r in results_tuple):
            continue
        selected[name] = idx

    return selected


def select_features(features_index, include_keys, include_results=None) -> Dict[str, int]:
    """
    Select from features_index those whose key contains any substring in include_keys,
    and (optionally) contains any substring in include_results.
    """
    # Turn include_keys into a tuple so 'in' can iterate directly
    keys_tuple = tuple(include_keys)
    results_tuple = tuple(include_results) if include_results is not None else None
    selected = {}
    for name, idx in features_index.items():
        if not any(k in name for k in keys_tuple):
            continue
        if results_tuple and not any(r in name for r in results_tuple):
            continue
        selected[name] = idx

    return selected


def get_nist_features_info(selected_features_config):
    nist_features_info = {}
    nist_features = selected_features_config.get("nist_tests", {})
    for feat_name, feat_info_list in nist_features.items():
        if feat_info_list:
            for feat_info in feat_info_list:
                nist_features_info[make_nist_feat_name_from_info(feat_name, feat_info)] = feat_info
        else:
            nist_features_info[feat_name] = {}

    return nist_features_info


def display_feature_index(xp_config, nist_tests_result_type, remapped_nist_selected_features):
    nist_features_info = get_nist_features_info(xp_config)
    for nist_result_type in nist_tests_result_type:
        logger.debug("nist_result_type = %s", nist_result_type)
        nist_result_type_filtered_features = {
            feat_name: feat_idx
            for feat_name, feat_idx in remapped_nist_selected_features.items()
            if nist_result_type in feat_name
        }
        for feat_name, feat_idx in nist_result_type_filtered_features.items():
            text = f"{feat_idx}: {feat_name}"
            nist_info = nist_features_info.get(f"{'_'.join(feat_name.split('_')[:-1])}", "")
            if nist_info:
                text += f";  {nist_info}"
            logger.debug("%s", text)


def select_features_for_token_pair(
    intra_samples_feature_index_dict: Dict, xp_config: Dict, token_pair: Optional[str] = None, nist_only: bool = False
) -> Dict[str, int]:
    """Select and merge features for classification for a given token_pair."""
    if token_pair is None:
        token_pair = next(iter(intra_samples_feature_index_dict.keys()))
        logger.warning("Running xp as there was no token stats related features.")

    features_index = intra_samples_feature_index_dict[token_pair]
    features_config = xp_config["features"]
    intra_keys = features_config.get("intra_samples_features", [])
    if xp_config.get("set_constant_seq_length") is not None:
        intra_keys = [k for k in intra_keys if k != "seq_length"]

    nist_selected = select_nist_features(features_index, features_config)
    if nist_only:
        return nist_selected

    # build each selection
    intra_selected = select_features(features_index, intra_keys)
    # merge them (latter keys overwrite former if duplicates exist)
    selected_features = {**intra_selected, **nist_selected}
    # Ensuring dict values (index) are sorted so that when extracted from matrix, the remapping of index will still be aligned.
    selected_features = dict(sorted(selected_features.items(), key=lambda item: item[1]))
    return selected_features


def validate_feature_consistency(selected_features_for_classification: Dict) -> None:
    """Validate that all datasets have the same feature indices."""
    assert all(
        v == list(selected_features_for_classification.values())[0]
        for v in selected_features_for_classification.values()
    ), "Not all features index are the same, please don't choose token features for the moment"


def remap_features_index(selected_features: Dict[str, int]) -> Dict[str, int]:
    """
    # Dict[feature_name: feature_idx]
    Create mapping from old feature indices to new positions and remap selected features.
    """
    remapped_selected_features = {
        feature_name: new_idx for new_idx, feature_name in enumerate(selected_features.keys())
    }

    return remapped_selected_features


def filter_features_idx_by_nist_result_type(features_idx, nist_result_type):
    return {feat_name: feat_idx for feat_name, feat_idx in features_idx.items() if nist_result_type in feat_name}
