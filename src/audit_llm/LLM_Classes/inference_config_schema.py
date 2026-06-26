"""Pydantic schema for ``scripts/Inference_configs.yaml`` validation.

Provides :func:`validate_all_configs` to validate all dataset entries from the
YAML in a single call.  Key normalisation maps legacy UPPER_CASE YAML keys
(``TOP_K``, ``MAX_TOKENS``, …) to their snake_case Pydantic field names.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class InferenceConfig(BaseModel):
    """Schema for a single dataset entry in Inference_configs.yaml."""

    model_config = ConfigDict(extra="forbid")

    min_seq_length: Optional[int]
    dyn_checking_batch_size: int
    top_k: int
    max_model_len: int
    max_tokens: int
    nb_of_samples: Optional[int] = None
    token_pairs_set: Optional[str] = None
    temperatures: Optional[list[float]] = None
    logprobs: Optional[int] = None


# ------------------------------------------------------------------
# Key normalisation (YAML legacy → Pydantic)
# ------------------------------------------------------------------

_KEY_ALIASES: dict[str, str] = {
    "TOP_K": "top_k",
    "MAX_TOKENS": "max_tokens",
    "TOKEN_PAIRS_SET": "token_pairs_set",
    "TEMPERATURES": "temperatures",
}


def normalize_config_keys(raw: dict[str, object]) -> dict[str, object]:
    """Normalise YAML keys to snake_case Pydantic field names."""
    return {_KEY_ALIASES.get(k, k): v for k, v in raw.items()}


def validate_inference_config(name: str, raw: dict[str, object]) -> InferenceConfig:
    """Validate and normalise a single inference config entry."""
    normalized = normalize_config_keys(raw)
    try:
        return InferenceConfig(**normalized)  # type: ignore[arg-type]
    except Exception as exc:
        available = list(raw.keys())
        raise ValueError(
            f"Invalid inference config for dataset '{name}' " f"(available keys: {available}): {exc}"
        ) from exc


def validate_all_configs(
    configs: dict[str, dict[str, object]],
) -> dict[str, InferenceConfig]:
    """Validate all entries from the YAML, applying ``_default`` as base."""
    configs = dict(configs)  # shallow copy — don't mutate caller's dict
    defaults: dict[str, object] = configs.pop("_default", {})
    return {name: validate_inference_config(name, {**defaults, **raw}) for name, raw in configs.items()}
