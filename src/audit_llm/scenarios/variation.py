# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Typed ``Variation`` (a grid cell) and resolved ``Instance`` (base model + cell).

An :class:`Instance` is a single materialised classification unit: "a transformed
copy of a base model" along the ``quantization`` / ``abliteration`` axes, plus a
sampling request along ``temperature`` / ``system_prompt_idx``. A :class:`Variation`
is the axis tuple *without* a base model bound yet — one cell of a variation grid.

The two ``*_name`` / ``label`` properties reproduce the legacy on-disk and
classification-label strings **byte-for-byte**. These strings are load-bearing:
they are parsed by ``audit_llm.xp_tools.model_filtering``
(``full_var_model_name_to_original_model_name`` /
``full_var_model_name_to_var_name``) and must not drift. The conventions:

- **storage_name** — the on-disk / ``model_idx`` model id (what the variation
  suffix is appended to):
  base ``{base_model}``; quantized ``{base_model}@@{quant}``
  (``@@`` = :data:`audit_llm.file_io.QUANTIZATION_SEPARATOR`); abliterated → the
  resolved *different* repo id.
- **label** — the classification-class label (``new_models_idx`` value):
  base / quantized ``{storage_name}_temp-{t}_sp-{sp}`` (variation suffix built via
  :func:`audit_llm.xp_tools.label_formatting.assemble_iterator_name_and_value`,
  temperature first); abliterated ``{repo}_ablit`` with **no** temp/sp suffix
  (matching ``"_".join([model_name, "ablit"])`` in ``prepare_dataset_features``).

This module imports the resolvers, which pull in the vLLM backend module (whose
import sets the multiprocessing start method). The package ``__init__`` therefore
deliberately exports only the light schema/loader path; import ``Variation`` /
``Instance`` explicitly from this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from audit_llm.file_io import QUANTIZATION_SEPARATOR
from audit_llm.scenarios.resolver import resolve_abliteration, resolve_quantization
from audit_llm.xp_tools.label_formatting import assemble_iterator_name_and_value

# The "no quantization" sentinel — the base-weights case (no ``@@`` suffix).
NO_QUANTIZATION = "no_quantized"


def _reject_abliterated_and_quantized(quantization: str, abliterated: bool) -> None:
    """The released data never combines abliteration with quantization."""
    if abliterated and quantization != NO_QUANTIZATION:
        raise ValueError(
            "An instance cannot be both abliterated and quantized "
            f"(got quantization={quantization!r}, abliterated=True); abliteration "
            "selects a different weights repo and is never crossed with quantization."
        )


@dataclass(frozen=True)
class Variation:
    """One cell of a variation grid, not yet bound to a base model.

    ``quantization`` defaults to :data:`NO_QUANTIZATION` (base weights);
    ``abliterated`` defaults to ``False``. Use :meth:`bind` to attach a base model
    and obtain an :class:`Instance`.
    """

    temperature: float
    system_prompt_idx: int
    quantization: str = NO_QUANTIZATION
    abliterated: bool = False

    def __post_init__(self) -> None:
        # Coerce to the exact types the legacy label strings were produced from,
        # so ``temp-1.0`` keeps its trailing ``.0`` and ``sp--1`` is an int.
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(self, "system_prompt_idx", int(self.system_prompt_idx))
        _reject_abliterated_and_quantized(self.quantization, self.abliterated)

    def bind(self, base_model: str) -> "Instance":
        """Bind this variation cell to ``base_model``, returning an :class:`Instance`."""
        return Instance(
            base_model=base_model,
            temperature=self.temperature,
            system_prompt_idx=self.system_prompt_idx,
            quantization=self.quantization,
            abliterated=self.abliterated,
        )


@dataclass(frozen=True)
class Instance:
    """A resolved base model × variation cell — one classification unit."""

    base_model: str
    temperature: float
    system_prompt_idx: int
    quantization: str = NO_QUANTIZATION
    abliterated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(self, "system_prompt_idx", int(self.system_prompt_idx))
        _reject_abliterated_and_quantized(self.quantization, self.abliterated)

    @property
    def is_quantized(self) -> bool:
        """True if this instance carries a quantization transform."""
        return self.quantization != NO_QUANTIZATION

    @property
    def storage_name(self) -> str:
        """The on-disk / ``model_idx`` model id this instance materialises from."""
        if self.abliterated:
            return resolve_abliteration(self.base_model)
        if self.is_quantized:
            return f"{self.base_model}{QUANTIZATION_SEPARATOR}{self.quantization}"
        return self.base_model

    @property
    def _variation_suffix(self) -> str:
        """The ``temp-<t>_sp-<sp>`` suffix (temperature first), byte-identical to legacy."""
        return "_".join(
            [
                assemble_iterator_name_and_value("temperature", self.temperature),
                assemble_iterator_name_and_value("system_prompt_idx", self.system_prompt_idx),
            ]
        )

    @property
    def label(self) -> str:
        """The classification-class label (the ``new_models_idx`` value)."""
        if self.abliterated:
            # Mirrors data_preparation.py: '_'.join([abliterated_repo, 'ablit']);
            # no temp/sp suffix — abliterated cells are pinned, never crossed.
            return "_".join([self.storage_name, "ablit"])
        return f"{self.storage_name}_{self._variation_suffix}"

    @property
    def quant_engine_kwargs(self) -> Dict[str, str]:
        """The vLLM engine kwarg for this instance's quantization level (possibly ``{}``)."""
        return resolve_quantization(self.base_model, self.quantization)
