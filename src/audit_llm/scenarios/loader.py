# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Loader for declarative scenario YAML files.

``load_scenario(path)`` reads ``config/scenarios/<name>.yaml``, builds a typed
:class:`~audit_llm.scenarios.schema.Scenario`, and validates every model
selection against the ``config/models.yaml`` registry (the model-universe SSOT
produced upstream). Any unknown base model, undeclared
``(quantized_base, quantization)`` pair, or unknown abliteration target raises a
clear :class:`ValueError` naming the offender and the registry path.

The loader reads/validates only. It never emits or rewrites any prompt-request
CSV — the scenario layer governs model × variation selection only (see
``config/scenarios/README.md`` for the CSV invariant).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml

from audit_llm.scenarios.schema import Scenario, VariationGroup

# Repo root: scenarios/ -> audit_llm/ -> src/ -> <root>. Dependency-free so the
# scenarios package does not import config side effects to find its files.
_ROOT = Path(__file__).resolve().parents[3]
_REGISTRY_PATH = _ROOT / "config" / "models.yaml"
_SCENARIOS_DIR = _ROOT / "config" / "scenarios"

# Abliteration is pinned to the paper default; no released data exists for any
# other abliterated cell, so the loader rejects abliteration crossings.
_ABLITERATION_PIN_TEMPERATURE = 1.0
_ABLITERATION_PIN_SYSTEM_PROMPT_IDX = -1


# ---------------------------------------------------------------------------
# Registry (config/models.yaml) reader
# ---------------------------------------------------------------------------


class _Registry:
    """The model universe declared in ``config/models.yaml``."""

    def __init__(self, base_models: Set[str], quant_pairs: Set[Tuple[str, str]],
                 quant_levels: Set[str], abliterated_hf_ids: Set[str]) -> None:
        self.base_models = base_models
        self.quant_pairs = quant_pairs  # {(base_model, quantization)}
        self.quant_levels = quant_levels
        self.abliterated_hf_ids = abliterated_hf_ids


def _load_registry(registry_path: Path) -> _Registry:
    if not registry_path.exists():
        raise ValueError(f"Model registry not found: {registry_path}")
    with open(registry_path, "r") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Malformed model registry (expected a mapping): {registry_path}")

    base_models = set(raw.get("base_models") or [])

    quant_pairs: Set[Tuple[str, str]] = set()
    quant_levels: Set[str] = set()
    for entry in raw.get("quantized_variants") or []:
        base = entry.get("base")
        quant = entry.get("quantization")
        if base and quant:
            quant_pairs.add((base, quant))
            quant_levels.add(quant)

    abliterated_hf_ids = {
        entry.get("abliterated_hf_id")
        for entry in (raw.get("abliterated_variants") or [])
        if entry.get("abliterated_hf_id")
    }

    return _Registry(base_models, quant_pairs, quant_levels, abliterated_hf_ids)


# ---------------------------------------------------------------------------
# YAML -> schema construction
# ---------------------------------------------------------------------------


def _build_groups(raw_groups: Any, *, section: str) -> Tuple[VariationGroup, ...]:
    """Build a tuple of ``VariationGroup`` from a YAML list-of-dicts grid."""
    if raw_groups is None:
        return ()
    if not isinstance(raw_groups, list):
        raise ValueError(
            f"Scenario section '{section}' must be a list of variation groups, "
            f"got {type(raw_groups).__name__}."
        )
    groups: List[VariationGroup] = []
    for i, raw in enumerate(raw_groups):
        if not isinstance(raw, dict):
            raise ValueError(
                f"Scenario section '{section}' group #{i} must be a mapping, "
                f"got {type(raw).__name__}."
            )
        unknown = set(raw) - {"temperature", "system_prompt_idx", "quantization", "abliteration"}
        if unknown:
            raise ValueError(
                f"Scenario section '{section}' group #{i} has unknown axis(es) "
                f"{sorted(unknown)}; allowed axes are "
                f"temperature, system_prompt_idx, quantization, abliteration."
            )
        groups.append(
            VariationGroup(
                temperature=tuple(float(t) for t in raw.get("temperature", []) or []),
                system_prompt_idx=tuple(int(s) for s in raw.get("system_prompt_idx", []) or []),
                quantization=tuple(str(q) for q in raw.get("quantization", []) or []),
                abliteration=tuple(str(a) for a in raw.get("abliteration", []) or []),
            )
        )
    return tuple(groups)


def _as_str_tuple(value: Any, *, key: str) -> Tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"Scenario key '{key}' must be a list of strings, got {type(value).__name__}.")
    return tuple(str(v) for v in value)


# ---------------------------------------------------------------------------
# Validation against the registry
# ---------------------------------------------------------------------------


def _validate(scenario: Scenario, registry: _Registry, registry_path: Path) -> None:
    where = f"(registry: {registry_path})"

    unknown_base = [m for m in scenario.base_models if m not in registry.base_models]
    if unknown_base:
        raise ValueError(f"Unknown base model(s) {unknown_base} not in registry {where}.")

    quant_bases = {base for base, _ in registry.quant_pairs}
    unknown_qbase = [m for m in scenario.quantized_base_models if m not in quant_bases]
    if unknown_qbase:
        raise ValueError(
            f"Unknown quantized base model(s) {unknown_qbase}; "
            f"no quantized_variants declared for them {where}."
        )

    requested_levels = {
        q for g in scenario.quantized_variations for q in g.quantization
    }
    unknown_levels = [q for q in requested_levels if q not in registry.quant_levels]
    if unknown_levels:
        raise ValueError(
            f"Unknown quantization level(s) {unknown_levels}; "
            f"declared levels are {sorted(registry.quant_levels)} {where}."
        )
    # Every (quantized base, requested level) must be a declared registry pair.
    for base in scenario.quantized_base_models:
        for level in requested_levels:
            if (base, level) not in registry.quant_pairs:
                raise ValueError(
                    f"Quantization '{level}' is not declared for base model "
                    f"'{base}' {where}."
                )

    unknown_ablit = [m for m in scenario.abliterated_models if m not in registry.abliterated_hf_ids]
    if unknown_ablit:
        raise ValueError(
            f"Unknown abliteration target(s) {unknown_ablit}; "
            f"declared abliterated repos are {sorted(registry.abliterated_hf_ids)} {where}."
        )

    _validate_abliteration_pin(scenario)


def _validate_abliteration_pin(scenario: Scenario) -> None:
    """Reject any abliteration cell outside the temp=1.0 / sp=-1 pin.

    No released data exists for abliterated cells other than the paper default,
    so abliteration must never be crossed over temperature / system_prompt_idx
    (or over quantization / abliteration axes).
    """
    for i, group in enumerate(scenario.abliteration_variations):
        bad_temp = [t for t in group.temperature if t != _ABLITERATION_PIN_TEMPERATURE]
        bad_sp = [s for s in group.system_prompt_idx if s != _ABLITERATION_PIN_SYSTEM_PROMPT_IDX]
        if bad_temp or bad_sp or group.quantization or group.abliteration:
            raise ValueError(
                f"abliteration_variations group #{i} crosses outside the pinned "
                f"cell (temperature={_ABLITERATION_PIN_TEMPERATURE}, "
                f"system_prompt_idx={_ABLITERATION_PIN_SYSTEM_PROMPT_IDX}): no "
                f"released data exists for other abliterated cells. Got "
                f"temperature={list(group.temperature)}, "
                f"system_prompt_idx={list(group.system_prompt_idx)}, "
                f"quantization={list(group.quantization)}, "
                f"abliteration={list(group.abliteration)}."
            )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _resolve_path(path: Union[str, Path]) -> Path:
    p = Path(path)
    # A bare scenario name (no suffix) resolves under config/scenarios/.
    if p.suffix == "":
        p = _SCENARIOS_DIR / f"{p.name}.yaml"
    if not p.exists():
        raise ValueError(f"Scenario file not found: {p}")
    return p


def load_scenario(
    path: Union[str, Path],
    registry_path: Optional[Union[str, Path]] = None,
) -> Scenario:
    """Load and validate a scenario from ``config/scenarios/<name>.yaml``.

    Parameters
    ----------
    path:
        A scenario name (``"main"`` -> ``config/scenarios/main.yaml``) or an
        explicit path to a YAML file.
    registry_path:
        Override for the ``config/models.yaml`` registry (defaults to the repo's
        ``config/models.yaml``). Primarily a testing seam.

    Returns
    -------
    Scenario
        The validated, typed scenario.

    Raises
    ------
    ValueError
        On a missing file, malformed YAML, or any model selection (base /
        quantization level / abliteration target) absent from the registry, and
        on any abliteration cell outside the temp=1.0 / sp=-1 pin.
    """
    scenario_path = _resolve_path(path)
    with open(scenario_path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Empty or malformed scenario YAML (expected a mapping): {scenario_path}")

    name = raw.get("name") or scenario_path.stem

    scenario = Scenario(
        name=str(name),
        base_models=_as_str_tuple(raw.get("base_models"), key="base_models"),
        base_variations=_build_groups(raw.get("base_variations"), section="base_variations"),
        quantized_base_models=_as_str_tuple(raw.get("quantized_base_models"), key="quantized_base_models"),
        quantized_variations=_build_groups(raw.get("quantized_variations"), section="quantized_variations"),
        abliterated_models=_as_str_tuple(raw.get("abliterated_models"), key="abliterated_models"),
        abliteration_variations=_build_groups(
            raw.get("abliteration_variations"), section="abliteration_variations"
        ),
    )

    registry = _load_registry(Path(registry_path) if registry_path else _REGISTRY_PATH)
    _validate(scenario, registry, registry_path=Path(registry_path) if registry_path else _REGISTRY_PATH)

    return scenario
