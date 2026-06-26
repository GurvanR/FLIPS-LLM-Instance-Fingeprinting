# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Shared scenario enumerator for the ANALYSIS layer.

:func:`build_instances` is the analysis-layer **source of truth** for *which*
model × variation :class:`~audit_llm.scenarios.variation.Instance` objects a
scenario materialises and what their classification labels are. It reproduces
the union the legacy analysis-side derivation produced:

- ``base_models`` × ``base_variations`` (temperature × system_prompt_idx);
- ``quantized_base_models`` × ``quantized_variations`` (temperature ×
  system_prompt_idx × quantization);
- ``abliterated_models`` (different weights repos) × ``abliteration_variations``
  (pinned temperature=1.0 / system_prompt_idx=-1).

The inference path (``scripts/Run_Inferences.py``) keeps its **own** vLLM
expansion — a documented asymmetry. This module is consumed by the analysis
layer ONLY and never touches inference code.

:func:`build_analysis_variation_structures` is the builder-backed *adapter* that
turns a :class:`~audit_llm.scenarios.schema.Scenario` into the analysis-side
structures (``model_variations_indices`` / ``quantized_model_variations_indices``
/ the abliterated repo list / the abliteration ``(temperature,
system_prompt_idx)`` pin). It supersedes the independent re-derivation in the
legacy ``compute_model_variations_indices``
(``audit_llm.xp_tools.variation_context``) — which stays physically present and
is slated for removal. The adapter therefore owns its own
sample-index grouping and does **not** call the legacy function, so its
removal will not affect this adapter.

This module imports :class:`Instance`, which pulls in the vLLM backend module.
The package ``__init__`` deliberately stays light; import ``build_instances`` /
``build_analysis_variation_structures`` from this module explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import polars as pl

from audit_llm.models_management.model_names import ABLITERATED_MODELS_MAP_TO_ORIGINAL
from audit_llm.scenarios.schema import Scenario, VariationGroup
from audit_llm.scenarios.variation import Instance
from audit_llm.xp_tools.label_formatting import assemble_iterator_name_and_value

# (temperature, system_prompt_idx) cell — the analysis variation key tuple.
VariationCell = Tuple[float, int]


def _expand_grid(groups: Tuple[VariationGroup, ...]) -> List[Dict]:
    """Flatten a variation grid (tuple of groups) to a deduped list of cells.

    Each group expands to its cartesian product (:meth:`VariationGroup.cartesian`);
    the grid is the *union* across groups, preserving first-occurrence order.
    """
    cells: List[Dict] = []
    seen = set()
    for group in groups:
        for cell in group.cartesian():
            key = tuple(sorted(cell.items()))
            if key not in seen:
                seen.add(key)
                cells.append(cell)
    return cells


def build_instances(scenario: Scenario) -> List[Instance]:
    """Enumerate every :class:`Instance` a scenario materialises (union, deduped).

    The analysis-layer source of truth for the classification class set and its
    labels. The order is: all base instances, then all quantized, then all
    abliterated; cells in grid order, models in scenario-declared order.
    """
    instances: List[Instance] = []

    # Base models × base grid.
    for cell in _expand_grid(scenario.base_variations):
        for base in scenario.base_models:
            instances.append(
                Instance(
                    base_model=base,
                    temperature=cell["temperature"],
                    system_prompt_idx=cell["system_prompt_idx"],
                )
            )

    # Quantized base models × quant grid (× quantization levels carried in the cell).
    for cell in _expand_grid(scenario.quantized_variations):
        quant = cell.get("quantization")
        for base in scenario.quantized_base_models:
            instances.append(
                Instance(
                    base_model=base,
                    temperature=cell["temperature"],
                    system_prompt_idx=cell["system_prompt_idx"],
                    quantization=quant,
                )
            )

    # Abliterated repos × abliteration grid (pinned). The scenario lists the
    # abliterated repos directly; map each back to its base so the Instance
    # reproduces ``storage_name``/``label`` = ``{repo}_ablit`` via the resolver.
    for cell in _expand_grid(scenario.abliteration_variations):
        for repo in scenario.abliterated_models:
            instances.append(
                Instance(
                    base_model=_abliterated_repo_to_base(repo),
                    temperature=cell["temperature"],
                    system_prompt_idx=cell["system_prompt_idx"],
                    abliterated=True,
                )
            )

    # Union, not multiset: dedup preserving order (Instance is a frozen dataclass).
    return list(dict.fromkeys(instances))


def _abliterated_repo_to_base(repo: str) -> str:
    """Map an abliterated HF repo id back to its base model (SSOT map lookup)."""
    try:
        return ABLITERATED_MODELS_MAP_TO_ORIGINAL[repo]
    except KeyError:
        raise ValueError(
            f"Unknown abliterated repo {repo!r}; declared abliterated repos are "
            f"{sorted(ABLITERATED_MODELS_MAP_TO_ORIGINAL)}."
        ) from None


# ---------------------------------------------------------------------------
# Analysis adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisVariationStructures:
    """The analysis-side structures derived from a scenario.

    Mirrors what the legacy ``_xp_config_init`` produced from the xp_config keys,
    but sourced from :func:`build_instances`:

    - ``model_variations_indices`` — ``{ "temp-<t>_sp-<sp>": [Index ...] }`` for
      base models (same dict applies to every base model);
    - ``quantized_model_variations_indices`` — same shape for quantized models
      (empty when the scenario has no quantized variations);
    - ``abliterated_models`` — the abliterated repo ids (``storage_name``s);
    - ``abliterated_variation`` — the single ``(temperature, system_prompt_idx)``
      pin shared by all abliterated instances, or ``None`` when there are none.
    """

    instances: Tuple[Instance, ...]
    model_variations_indices: Dict[str, List[int]]
    quantized_model_variations_indices: Dict[str, List[int]]
    abliterated_models: Tuple[str, ...]
    abliterated_variation: Optional[VariationCell]


def _distinct_cells(instances: List[Instance]) -> List[VariationCell]:
    """Distinct ``(temperature, system_prompt_idx)`` cells, first-occurrence order."""
    cells: List[VariationCell] = []
    for inst in instances:
        cell = (inst.temperature, inst.system_prompt_idx)
        if cell not in cells:
            cells.append(cell)
    return cells


def _variation_indices(cells: List[VariationCell], main_dataset_df: pl.DataFrame) -> Dict[str, List[int]]:
    """Map each ``(temperature, system_prompt_idx)`` cell to its sample indices.

    Reproduces the legacy ``compute_model_variations_indices`` contract — the
    ``temp-<t>_sp-<sp>`` key (temperature first), the sorted ``Index`` list, and
    the balanced-classes check — but owns the grouping so the scenario path does
    not depend on the soon-to-be-removed legacy function.
    """
    indices: Dict[str, List[int]] = {}
    for temperature, system_prompt_idx in cells:
        filtered = main_dataset_df.filter(
            (pl.col("temperature") == temperature)
            & (pl.col("system_prompt_idx") == system_prompt_idx)
        )
        label = "_".join(
            [
                assemble_iterator_name_and_value("temperature", temperature),
                assemble_iterator_name_and_value("system_prompt_idx", system_prompt_idx),
            ]
        )
        indices[label] = sorted(filtered["Index"].to_list())

    if indices:
        first_len = len(next(iter(indices.values())))
        if not all(len(v) == first_len for v in indices.values()):
            raise ValueError(
                f"Model variations are unbalanced. Expected length {first_len} for all combinations: "
                f"{ {k: len(v) for k, v in indices.items()} }."
            )
    return indices


def build_analysis_variation_structures(
    scenario: Scenario, main_dataset_df: pl.DataFrame
) -> AnalysisVariationStructures:
    """Derive the analysis-side structures from a scenario, backed by build_instances."""
    instances = build_instances(scenario)

    base = [i for i in instances if not i.abliterated and not i.is_quantized]
    quantized = [i for i in instances if i.is_quantized]
    abliterated = [i for i in instances if i.abliterated]

    model_variations_indices = _variation_indices(_distinct_cells(base), main_dataset_df)
    quantized_model_variations_indices = (
        _variation_indices(_distinct_cells(quantized), main_dataset_df) if quantized else {}
    )

    # Deduped abliterated repo ids, in instance order.
    abliterated_models = tuple(dict.fromkeys(i.storage_name for i in abliterated))

    abliterated_cells = _distinct_cells(abliterated)
    if len(abliterated_cells) > 1:
        raise ValueError(
            f"Abliterated instances must share a single (temperature, system_prompt_idx) "
            f"pin; got {abliterated_cells}."
        )
    abliterated_variation = abliterated_cells[0] if abliterated_cells else None

    return AnalysisVariationStructures(
        instances=tuple(instances),
        model_variations_indices=model_variations_indices,
        quantized_model_variations_indices=quantized_model_variations_indices,
        abliterated_models=abliterated_models,
        abliterated_variation=abliterated_variation,
    )
