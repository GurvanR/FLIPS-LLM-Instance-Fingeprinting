"""Model filtering and name manipulation utilities."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Union

logger = logging.getLogger(__name__)

from audit_llm.file_io import QUANTIZATION_SEPARATOR, get_base_model_name, get_quantization_suffix
from audit_llm.models_management.model_names import (
    ABLITERATED_MODELS,
    ABLITERATED_MODELS_MAP_TO_ORIGINAL,
    PRNG_MODELS,
)


def truncate_model_name(model_name: str, k=26):
    if len(model_name) <= k:
        return model_name

    # Safely take part after last '/' if present
    model_name = model_name.rsplit("/", 1)[-1]

    if len(model_name) > k:
        model_name = model_name[: k - 3] + "..."

    return model_name


def filter_models(models: List[str], xp_config: Dict) -> List[str]:
    """
    Filter the given list of model names based on experiment configuration flags.

    Behavior:
    - Removes PRNG models unless explicitly allowed in `xp_config['PRNGs']`.
    - If `xp_config['PRNGs']` is an empty list → keep all PRNG models.
    - If `xp_config['PRNGs']` is None → remove all PRNG models.
    - Removes any models listed in `xp_config['models_to_remove']`.

    Args:
        models: A list of model names.
        xp_config: A configuration dictionary with optional keys:
            - 'PRNGs': List of PRNG models to include, or [] to include all.
            - 'models_to_remove': List of models to remove from the final list.

    Returns:
        A filtered list of model names.
    """
    models = models.copy()  # Avoid mutating the input

    # Extract configuration options safely
    prngs_to_use = xp_config.get("PRNGs", [])
    # Use list() to copy and handle None (schema sets it to None when unspecified)
    models_to_remove = list(xp_config.get("models_to_remove") or [])
    logger.debug("models_to_remove = %s", xp_config.get("models_to_remove"))

    # Removing abliterated models
    models_to_remove.extend(ABLITERATED_MODELS)
    # --- Handle PRNG filtering ---
    non_prng_models = [m for m in models if m not in PRNG_MODELS]

    if prngs_to_use is None:
        # Explicit None means remove all PRNG models
        models = non_prng_models
    elif prngs_to_use:
        # Include only selected PRNG models
        prng_models = [m for m in models if m in prngs_to_use]
        models = non_prng_models + prng_models
    else:
        # Empty list [] means keep all PRNG models (no filtering)
        models = models

    # --- Handle quantized model filtering ---
    include_quantized = xp_config.get("include_quantized")
    if include_quantized is None:
        # None → remove all quantized models (@@-suffixed)
        models = [m for m in models if QUANTIZATION_SEPARATOR not in m]
    elif include_quantized:
        # Non-empty list → keep only specified quantization methods
        models = [
            m for m in models
            if QUANTIZATION_SEPARATOR not in m
            or get_quantization_suffix(m) in include_quantized
        ]
    # else: empty list [] → keep all quantized models (no filtering)

    # --- Remove explicitly listed models ---
    models = [m for m in models if m not in models_to_remove]
    logger.debug("models = %s", models)
    return models


def filter_token_pairs(datasets: List[str], token_pairs_banned_path: Path) -> List[str]:
    """Filter datasets list based on configuration flags."""
    # If the token_pairs_banned_path exists, we load it, otherwise we create an empty dict and save it
    datasets = datasets.copy()
    if token_pairs_banned_path.exists():
        with open(token_pairs_banned_path, "r") as f:
            datasets_banned = json.load(f)
        datasets = [dataset for dataset in datasets if not datasets_banned.get(dataset, False)]
    else:
        datasets_banned = {"setup_example": False}
        with open(token_pairs_banned_path, "w") as f:
            json.dump(datasets_banned, f)

    logger.debug("token_pairs_banned_path = %s", token_pairs_banned_path)
    logger.debug("Datasets banned: %s", datasets_banned)

    return datasets


def remove_closed_source_model(models: list):

    closed_model = [
        "anthropic/claude-3-haiku",
        "deepseek/deepseek-chat",
        "deepseek/deepseek-chat-v3-0324",
        "deepseek/deepseek-r1",
        "google/gemini-2.0-flash-001",
        "google/gemini-2.5-flash-lite",
        "google/gemini-flash-1.5",
        "google/gemini-flash-1.5-8b",
        "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/llama-3.1-405b-instruct",
        "nousresearch/hermes-3-llama-3.1-405b",
        "openai/gpt-3.5-turbo",
        "openai/gpt-4.1-mini",
        "openai/gpt-4.1-nano",
    ]
    return [model for model in models if model not in closed_model]


def group_models_idx_by_var_or_orig(models_idx: Dict[int, str], group_by="orig"):
    """
    Groups model indices by either original model name or variation.
    Returns a dictionary of {unit_name: [(model_idx, model_name), ...]}.
    """
    grouped = {}
    for model_idx, model_name in models_idx.items():
        if group_by == "orig":
            unit_name = full_var_model_name_to_original_model_name(model_name)
        else:
            unit_name = full_var_model_name_to_var_name(model_name)
        grouped.setdefault(unit_name, []).append((model_idx, model_name))
    return grouped


def full_var_model_name_to_original_model_name(full_var_model_name: str):
    if "_ablit" in full_var_model_name:
        original_model_name = ABLITERATED_MODELS_MAP_TO_ORIGINAL[full_var_model_name.split("_")[0]]
    elif full_var_model_name in PRNG_MODELS:
        return full_var_model_name  # Keep the full PRNG name as the row label
    else:
        original_model_name = get_base_model_name(full_var_model_name.rsplit("_")[0])
    return original_model_name


def full_var_model_name_to_var_name(full_var_model_name: str):
    if full_var_model_name in PRNG_MODELS:
        return "PRNG"  # Canonical column name, analogous to "ablit"
    elif "_" in full_var_model_name:
        parts = full_var_model_name.rsplit("_")
        model_part = parts[0]
        var_part = "_".join(parts[1:])
        quant_suffix = get_quantization_suffix(model_part)
        if quant_suffix:
            return f"{quant_suffix}_{var_part}"
        return var_part
    else:
        return full_var_model_name


def original_model_name_to_safe_var_model_idx_mapper(new_models_idx, safe_suffix: str = "temp-1.0_sp--1"):
    """
    Docstring for original_model_name_to_safe_var_model_idx_mapper

    :param model_idx: new_models_idx # {model_idx: model_name}
    """
    model_var_to_safe_model_map = {}
    for model_idx, model_name in new_models_idx.items():
        if safe_suffix in model_name:
            original_model_name = get_base_model_name(model_name.split("_")[0])
            model_var_to_safe_model_map[original_model_name] = model_idx

    return model_var_to_safe_model_map


def full_var_model_name_to_full_safe_var_model_name_mapper(
    new_models_idx: Union[Dict, List], safe_suffix: str = "temp-1.0_sp--1"
):

    full_var_model_name_to_full_safe_var_model_name_map = {}
    if isinstance(new_models_idx, list):
        new_models_idx = {idx: name for idx, name in enumerate(new_models_idx)}
    for model_idx, model_name in new_models_idx.items():
        original_model_name = full_var_model_name_to_original_model_name(model_name)
        full_var_model_name_to_full_safe_var_model_name_map[model_name] = "_".join([original_model_name, safe_suffix])

    return full_var_model_name_to_full_safe_var_model_name_map
