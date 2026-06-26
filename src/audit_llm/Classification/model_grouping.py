"""Model grouping utilities for group-based classification.

Provides ``build_group_mapping`` which maps individual model indices to group
indices based on either hardcoded group definitions or parameter-based grouping.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Abbreviated parameter names used in variation suffixes (from label_formatting.py)
_PARAM_SUFFIX_PREFIXES = {
    "temperature": "temp",
    "system_prompt_idx": "sp",
    "frequency_penalty": "fp",
}

# Suffix appended to abliterated model names in new_models_idx
_ABLIT_SUFFIX = "_ablit"


def build_group_mapping(
    new_models_idx: Dict[int, str],
    model_groups_config: Dict[str, Any],
) -> Tuple[Dict[int, int], Dict[int, str]]:
    """Build a mapping from model indices to group indices.

    Parameters
    ----------
    new_models_idx : Dict[int, str]
        Mapping of model index to expanded model name (after variations/abliteration).
        Example: ``{0: "llama-3-8b_temp-0.4_sp--1", 1: "llama-3-8b_temp-1.0_sp--1", 2: "qwen_ablit"}``
    model_groups_config : Dict[str, Any]
        Group configuration from ``xp_config["model_groups"]``.
        Either parameter-based (has ``"group_by"`` key) or hardcoded.

    Returns
    -------
    model_to_group : Dict[int, int]
        Maps model index → group index. Models not in any group are absent.
    group_names : Dict[int, str]
        Maps group index → group name for display.
    """
    if "group_by" in model_groups_config:
        return _build_parameter_based_groups(new_models_idx, model_groups_config["group_by"])
    else:
        return _build_hardcoded_groups(new_models_idx, model_groups_config)


def _build_hardcoded_groups(
    new_models_idx: Dict[int, str],
    groups_config: Dict[str, Any],
) -> Tuple[Dict[int, int], Dict[int, str]]:
    """Build groups from explicit model name lists.

    Supports two forms per group:
    - Simple list: ``group_name: [model_a, model_b]`` — all variations included
    - Dict with filters: ``group_name: {models: [...], temperature: [...]}``
    """
    model_to_group: Dict[int, int] = {}
    group_names: Dict[int, str] = {}

    for group_idx, (group_name, group_spec) in enumerate(groups_config.items()):
        group_names[group_idx] = group_name

        if isinstance(group_spec, list):
            # Simple form: list of base model names
            base_names = group_spec
            param_filters: Dict[str, List] = {}
        elif isinstance(group_spec, dict):
            # Dict form with optional parameter filters
            base_names = group_spec.get("models", [])
            param_filters = {
                k: v for k, v in group_spec.items()
                if k != "models" and isinstance(v, list)
            }
        else:
            logger.warning("Unexpected group spec type for '%s': %s", group_name, type(group_spec))
            continue

        for model_idx, expanded_name in new_models_idx.items():
            if _model_matches(expanded_name, base_names, param_filters):
                if model_idx in model_to_group:
                    logger.warning(
                        "Model '%s' (idx=%d) matched multiple groups: already in group '%s', "
                        "also matches '%s'. Keeping first assignment.",
                        expanded_name, model_idx, group_names[model_to_group[model_idx]], group_name,
                    )
                else:
                    model_to_group[model_idx] = group_idx

    _log_group_summary(new_models_idx, model_to_group, group_names)
    return model_to_group, group_names


def _model_matches(
    expanded_name: str,
    base_names: List[str],
    param_filters: Dict[str, List],
) -> bool:
    """Check if an expanded model name matches any base name and passes parameter filters.

    If ``base_names`` is empty, all models are considered (only parameter filters apply).
    """
    if base_names:
        # Check if any base name is a prefix of the expanded name
        matched_base = False
        for base_name in base_names:
            if expanded_name == base_name or expanded_name.startswith(base_name + "_"):
                matched_base = True
                break

        if not matched_base:
            return False

    # If no parameter filters, match is confirmed
    if not param_filters:
        return True

    # Check each parameter filter against the variation suffix
    for param_name, allowed_values in param_filters.items():
        prefix = _PARAM_SUFFIX_PREFIXES.get(param_name, param_name)
        # Pattern: _{prefix}-{value} followed by _ or end of string
        pattern = rf"_{re.escape(prefix)}-([^_]+)"
        match = re.search(pattern, expanded_name)
        if match:
            parsed_value = match.group(1)
            # Try numeric comparison first
            try:
                numeric_val = float(parsed_value)
                if numeric_val not in [float(v) for v in allowed_values]:
                    return False
            except ValueError:
                if parsed_value not in [str(v) for v in allowed_values]:
                    return False
        else:
            # Parameter not found in name — no variation suffix means no filtering possible
            # Skip this filter (model has no variation for this parameter)
            pass

    return True


def _build_parameter_based_groups(
    new_models_idx: Dict[int, str],
    group_by: str,
) -> Tuple[Dict[int, int], Dict[int, str]]:
    """Build groups automatically from a shared parameter.

    Supported values for ``group_by``:
    - ``"temperature"`` → one group per temperature value
    - ``"system_prompt_idx"`` → one group per system prompt index
    - ``"abliteration"`` → two groups: "abliterated" vs "non-abliterated"
    """
    model_to_group: Dict[int, int] = {}
    group_names: Dict[int, str] = {}

    if group_by == "abliteration":
        # Special case: binary grouping based on _ablit suffix
        group_names = {0: "non-abliterated", 1: "abliterated"}
        for model_idx, name in new_models_idx.items():
            if name.endswith(_ABLIT_SUFFIX):
                model_to_group[model_idx] = 1
            else:
                model_to_group[model_idx] = 0
    else:
        # Parse parameter value from variation suffix
        prefix = _PARAM_SUFFIX_PREFIXES.get(group_by, group_by)
        pattern = rf"_{re.escape(prefix)}-([^_]+)"

        value_to_group: Dict[str, int] = {}

        for model_idx, name in new_models_idx.items():
            match = re.search(pattern, name)
            if match:
                val = match.group(1)
                if val not in value_to_group:
                    group_idx = len(value_to_group)
                    value_to_group[val] = group_idx
                    group_names[group_idx] = f"{group_by}={val}"
                model_to_group[model_idx] = value_to_group[val]
            else:
                logger.debug(
                    "Model '%s' has no '%s' suffix — excluded from parameter-based grouping.",
                    name, group_by,
                )

    _log_group_summary(new_models_idx, model_to_group, group_names)
    return model_to_group, group_names


def _log_group_summary(
    new_models_idx: Dict[int, str],
    model_to_group: Dict[int, int],
    group_names: Dict[int, str],
) -> None:
    """Log a summary of the group mapping."""
    total = len(new_models_idx)
    assigned = len(model_to_group)
    excluded = total - assigned
    logger.info(
        "Model grouping: %d models → %d groups (%d excluded)",
        total, len(group_names), excluded,
    )
    for gid, gname in group_names.items():
        members = [new_models_idx[mid] for mid, g in model_to_group.items() if g == gid]
        logger.info("  Group '%s' (idx=%d): %d models", gname, gid, len(members))
        for m in members:
            logger.debug("    - %s", m)
    if excluded:
        excluded_models = [new_models_idx[mid] for mid in new_models_idx if mid not in model_to_group]
        logger.info("  Excluded: %d models", excluded)
        for m in excluded_models:
            logger.debug("    - %s", m)
