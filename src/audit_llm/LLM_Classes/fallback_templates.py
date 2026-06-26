"""Fallback prompt templates for models without a working chat_template.

The fallback chain is: apply_chat_template (with system) -> apply_chat_template
(without system) -> format_with_fallback -> plain concatenation.
"""

import logging
logger = logging.getLogger(__name__)

from collections.abc import Callable
from typing import Optional


def format_for_phi3(
    user_text: str,
    system_prompt: Optional[str] = None,
    bos_token: str = "<s>",
    eos_token: str = "</s>",
) -> str:
    """Format prompt for Phi-3 models."""
    prompt = ""
    if system_prompt:
        prompt += f"<|system|>\n{system_prompt}<|end|>\n"
    prompt += f"<|user|>\n{user_text}<|end|>\n<|assistant|>"
    return f"{bos_token}{prompt}{eos_token}"


def format_for_llama_2_instruct(
    user_text: str,
    system_prompt: Optional[str] = None,
    bos_token: str = "<s>",
) -> str:
    """Format prompt for togethercomputer/Llama-2-7B-32K-Instruct."""
    user_text = user_text.strip()
    if system_prompt:
        return f"{bos_token}[INST] <<SYS>>\n" f"{system_prompt.strip()}\n" f"<</SYS>>\n\n" f"{user_text} [/INST]"
    return f"{bos_token}[INST] {user_text} [/INST]"


def format_for_orca_2(
    user_text: str,
    system_prompt: Optional[str] = None,
    bos_token: str = "<s>",
    eos_token: str = "</s>",
) -> str:
    """Format prompt for microsoft/Orca-2-13b."""
    system_content = f"<|im_start|>system {system_prompt} <|im_end|>" if system_prompt else ""
    user_content = f"<|im_start|>user {user_text} <|im_end|>"
    return f"{bos_token}{system_content}\n{user_content}\n" f"<|im_start|>assistant {eos_token}"


def format_for_llama_2_chat(
    user_text: str,
    system_prompt: Optional[str] = None,
    bos_token: str = "<s>",
    eos_token: str = "</s>",
) -> str:
    """Format prompt for meta-llama/Llama-2-*b-chat-hf models."""
    if system_prompt:
        content = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_text}"
    else:
        content = user_text
    return f"{bos_token}[INST] {content.strip()} [/INST]{eos_token}"


# ------------------------------------------------------------------
# Registry: (substring, formatter) — checked in order, first match wins.
# ------------------------------------------------------------------

_FALLBACK_REGISTRY: list[tuple[str, Callable[..., str]]] = [
    ("phi-3", format_for_phi3),
    ("phi3", format_for_phi3),
    (
        "togethercomputer/llama-2-7b-32k-instruct",
        format_for_llama_2_instruct,
    ),
    ("microsoft/orca-2-13b", format_for_orca_2),
]


def _is_llama_2_chat(model_id_lower: str) -> bool:
    """Check if model is a Llama-2 chat variant (compound condition)."""
    return "meta-llama/llama-2" in model_id_lower and "chat-hf" in model_id_lower


def format_with_fallback(
    model_id: str,
    user_text: str,
    system_prompt: Optional[str],
    verbose: bool = True,
) -> str:
    """Apply the best-matching fallback template for the given model."""
    lower = model_id.lower()

    # Llama-2 chat requires a compound condition — check first
    if _is_llama_2_chat(lower):
        return format_for_llama_2_chat(user_text, system_prompt)

    # Walk the registry
    for substring, formatter in _FALLBACK_REGISTRY:
        if substring in lower:
            return formatter(user_text, system_prompt)

    # Default: plain concatenation
    if verbose:
        logger.warning("No specific fallback template for model %s. Using default formatting.", model_id)
    return f"{system_prompt}\n\n{user_text}" if system_prompt else user_text
