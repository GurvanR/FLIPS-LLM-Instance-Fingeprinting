"""Model variation context and index computation.

This module implements the VariationContext dataclass to consolidate
model variation index remapping logic.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

import polars as pl

from audit_llm.models_management.model_names import ABLITERATED_MODELS_MAP_TO_ORIGINAL
from audit_llm.xp_tools.label_formatting import assemble_iterator_name_and_value


@dataclass
class VariationContext:
    """
    Consolidates model variation index remapping logic.
    
    This dataclass validates that the total number of classes matches the expected count:
    n_classes == len(base_models) * n_variations + len(abliterated_models)
    
    Attributes:
        base_models: List of base model names (without variations)
        variations: Dict mapping variation parameter names to their values
                   e.g., {"temperature": [0.4, 1.0], "system_prompt_idx": [-1, 0]}
        abliterated_models: List of abliterated model names
        model_variations_indices: Dict mapping variation keys to model indices
                                 e.g., {"temp-0.4_sp--1": [0, 1, 2], ...}
        n_classes: Total number of model classes
    
    # TODO: move abliterated model parameters (temperature=1.0, sp=-1) to YAML config.
    """
    base_models: List[str]
    variations: Dict[str, List]
    abliterated_models: List[str]
    model_variations_indices: Dict[str, List[int]]
    n_classes: int
    quantized_model_variations_indices: Dict[str, List[int]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate that n_classes matches expected model count.

        Validation is skipped when ``variations`` is empty, because the
        caller may not yet have the full variation structure available.
        """
        if not self.variations:
            return
        # Calculate expected variations
        n_variations = 1
        for var_list in self.variations.values():
            n_variations *= len(var_list)

        # Expected number of classes
        expected = len(self.base_models) * n_variations + len(self.abliterated_models)

        if self.n_classes != expected:
            raise ValueError(
                f"VariationContext validation failed: "
                f"n_classes ({self.n_classes}) != expected ({expected}). "
                f"base_models={len(self.base_models)}, n_variations={n_variations}, "
                f"abliterated_models={len(self.abliterated_models)}"
            )


def compute_model_variations_indices(
    xp_config: Dict, main_dataset_df, model_variations: Optional[Union[Dict[str, List], List[Dict[str, List]]]] = None
) -> Dict:
    """
    Section 4: Computing index of model variations.
    Args:
        xp_config: Experiment configuration  dictionary
        main_dataset_df: Main dataset DataFrame
        model_variations: Either a single dict or list of dicts specifying variations.
            Single dict example: {'temperature': [0.1, 0.5, 1.0], 'system_prompt_idx': [-1, 0, 9]}
                Creates cartesian product: 3 × 3 = 9 variations

            List of dicts example: [
                {'temperature': [1.0], 'system_prompt_idx': [0, 5, 9]},
                {'temperature': [0.1, 0.5, 0.9], 'system_prompt_idx': [-1]}
            ]
                Creates: 1×3 + 3×1 = 6 variations

    Returns:
        Dictionary of model variation indices
        e.g. {'temp-0.1_sp--1': [0, 1, 2], 'temp-0.5_sp-0': [10, 11, 12], ...}
    """
    model_variations_indices = {}

    if model_variations is None:
        model_variations = xp_config.get("model_variations", None)

    if model_variations is not None:
        # Normalize to list of dicts
        if isinstance(model_variations, dict):
            variations_list = [model_variations]
        else:
            variations_list = model_variations

        # Process each variation group
        for variation_group in variations_list:
            variation_cols = list(variation_group.keys())

            # 1. Validation: Ensure all variation columns exist in the DataFrame
            missing_cols = [col for col in variation_cols if col not in main_dataset_df.columns]
            if missing_cols:
                raise ValueError(f"The following model variation columns are missing from the dataset: {missing_cols}")

            # 2. Filter & Group (Cartesian product within this group)
            filter_condition = pl.lit(True)
            for col_name, allowed_values in variation_group.items():
                filter_condition &= pl.col(col_name).is_in(allowed_values)

            grouped_df = (
                main_dataset_df.filter(filter_condition)
                .group_by(variation_cols)
                .agg(pl.col("Index").alias("indices"))
                .sort(variation_cols)
            )

            # 3. Convert to dictionary {variation_key: [list_of_indices]}
            keys = grouped_df.select(variation_cols).rows()
            group_indices = dict(zip(keys, grouped_df["indices"].to_list()))

            # Merge into main dictionary
            model_variations_indices.update(group_indices)

        # Formatting names of keys in a string
        name_list_dict = {}
        for key, value in model_variations_indices.items():
            name_list = []
            # Get column names from the key tuple
            # Need to figure out which columns these values correspond to
            # We'll use the first variation group's keys as reference
            iterator_names = list(variations_list[0].keys())
            for i, iterator_name in enumerate(iterator_names):
                name_list.append((assemble_iterator_name_and_value(iterator_name, key[i])))
            name_list_dict[key] = "_".join(name_list)

        model_variations_indices = {name_list_dict[key]: value for key, value in model_variations_indices.items()}
        for key, value in model_variations_indices.items():
            logger.debug("%s: %s...%s", key, value[:7], value[-7:])

        # 4. Assert all variations have the same sample count (Balanced classes check)
        if model_variations_indices:
            first_len = len(next(iter(model_variations_indices.values())))
            if not all(len(indices) == first_len for indices in model_variations_indices.values()):
                raise ValueError(
                    f"Model variations are unbalanced. Expected length {first_len} for all combinations."
                )

    return model_variations_indices
