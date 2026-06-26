# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Pure resolvers for the two *model-transform* variation axes.

A scenario varies four axes (``temperature``, ``system_prompt_idx``,
``quantization``, ``abliteration``). The first two are sampling-request fields;
the last two **transform the model** — an :class:`~audit_llm.scenarios.variation.Instance`
is "a transformed copy of a base model" along these two axes. The two transforms
differ in *kind*, and that difference is isolated here:

- **quantization** keeps the same weights and changes only a *vLLM engine kwarg*
  (e.g. ``fp8`` ⇒ ``{"quantization": "fp8"}``);
- **abliteration** swaps in a *different weights repo* (e.g.
  ``meta-llama/Llama-3.1-8B-Instruct`` ⇒
  ``failspy/Meta-Llama-3-8B-Instruct-abliterated-v3``).

Both resolvers **reuse** the existing sources of truth rather than re-declaring
them: ``QUANTIZATION_VLLM_PARAMS`` (the vLLM backend's quantization map) and
``ABLITERATED_MODELS_MAP_TO_ORIGINAL`` (the abliterated→original map the released
data was generated against). The kwarg-vs-different-weights distinction lives
*only* in this module.
"""

from __future__ import annotations

from typing import Dict

from audit_llm.LLM_Classes.vLLM_Classes import QUANTIZATION_VLLM_PARAMS
from audit_llm.models_management.model_names import ABLITERATED_MODELS_MAP_TO_ORIGINAL

# Inverse of the SSOT map: {original base model -> abliterated (different) repo}.
# All map values are distinct, so the inverse is well-defined. Built by reusing
# ABLITERATED_MODELS_MAP_TO_ORIGINAL — never re-declared.
_BASE_TO_ABLITERATED: Dict[str, str] = {
    original: abliterated for abliterated, original in ABLITERATED_MODELS_MAP_TO_ORIGINAL.items()
}


def resolve_quantization(base_model: str, quantization: str) -> Dict[str, str]:
    """Resolve a quantization level to its vLLM engine kwarg.

    Reuses ``QUANTIZATION_VLLM_PARAMS`` from the vLLM backend (the SSOT for how a
    quantization level maps to engine parameters). ``"no_quantized"`` yields an
    empty kwarg; ``"fp8"`` → ``{"quantization": "fp8"}``; ``"bitsandbytes_int4"``
    → ``{"quantization": "bitsandbytes"}`` (note the kwarg *value* differs from
    the level *key* — the ``@@`` storage suffix uses the key, the engine the value).

    ``base_model`` is part of the resolver contract (symmetry with
    :func:`resolve_abliteration`, which genuinely depends on it); the resulting
    engine kwarg is the same for any base model carrying that quantization level.

    Returns a fresh dict copy so callers cannot mutate the shared SSOT map.

    Raises
    ------
    ValueError
        If ``quantization`` is not a declared level.
    """
    if quantization not in QUANTIZATION_VLLM_PARAMS:
        raise ValueError(
            f"Unknown quantization level {quantization!r} for base model "
            f"{base_model!r}; declared levels are {sorted(QUANTIZATION_VLLM_PARAMS)}."
        )
    return dict(QUANTIZATION_VLLM_PARAMS[quantization])


def resolve_abliteration(base_model: str) -> str:
    """Resolve a base model to its concrete abliterated (different weights) repo.

    Reuses the inverse of ``ABLITERATED_MODELS_MAP_TO_ORIGINAL``. Unlike
    quantization, abliteration is *not* an engine kwarg: it selects a different
    Hugging Face repository entirely.

    Raises
    ------
    ValueError
        If ``base_model`` has no abliterated counterpart.
    """
    try:
        return _BASE_TO_ABLITERATED[base_model]
    except KeyError:
        raise ValueError(
            f"No abliterated repo for base model {base_model!r}; abliteratable "
            f"bases are {sorted(_BASE_TO_ABLITERATED)}."
        ) from None
