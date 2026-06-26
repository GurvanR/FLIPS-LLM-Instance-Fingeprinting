# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Unit tests for the scenario resolvers and the Variation/Instance label forms.

These pin the load-bearing string conventions (``@@<quant>`` / ``{repo}_ablit`` /
``temp-<t>_sp-<sp>``) byte-for-byte and tie the abliteration resolver to the 5
data-backed repos derived from the released class-identity JSON.
"""

import json
from pathlib import Path

import pytest

from audit_llm.LLM_Classes.vLLM_Classes import QUANTIZATION_VLLM_PARAMS
from audit_llm.models_management.model_names import ABLITERATED_MODELS_MAP_TO_ORIGINAL
from audit_llm.scenarios.resolver import resolve_abliteration, resolve_quantization
from audit_llm.scenarios.variation import NO_QUANTIZATION, Instance, Variation
from audit_llm.xp_tools.model_filtering import (
    full_var_model_name_to_original_model_name,
    full_var_model_name_to_var_name,
)

# Released class-identity file (205 labels = 200 base + 5 ablit + 0 quant).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_RELEASED_LABELS_JSON = (
    _REPO_ROOT
    / "XP_configs"
    / "e3_flips_vs_llmmap"
    / "llmmap_if_data"
    / "checkpoint_dir"
    / "new_var_models_idx.json"
)

# The 5 DATA-BACKED abliterated repos -> their base models, derived authoritatively
# from the released ``_ablit`` labels (re-derived from the JSON in
# ``test_data_backed_abliterated_set_matches_released_json``). NOT the same as
# ``len(ABLITERATED_MODELS_MAP_TO_ORIGINAL)`` (6 here) nor ``ABLITERATED_MODELS`` (8).
DATA_BACKED_ABLITERATED = {
    "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3": "meta-llama/Llama-3.1-8B-Instruct",
    "failspy/Smaug-Llama-3-70B-Instruct-abliterated-v3": "abacusai/Smaug-Llama-3-70B-Instruct",
    "natong19/Qwen2-7B-Instruct-abliterated": "Qwen/Qwen2-7B-Instruct",
    "failspy/Phi-3-medium-4k-instruct-abliterated-v3": "microsoft/Phi-3-medium-4k-instruct",
    "dphn/dolphin-2.9.2-Phi-3-Medium-abliterated": "microsoft/Phi-3-medium-128k-instruct",
}


# ---------------------------------------------------------------------------
# resolve_quantization — vLLM engine kwargs (reuse of QUANTIZATION_VLLM_PARAMS)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "level,expected",
    [
        ("no_quantized", {}),
        ("fp8", {"quantization": "fp8"}),
        ("bitsandbytes_int4", {"quantization": "bitsandbytes"}),
    ],
)
def test_resolve_quantization_returns_vllm_kwarg(level, expected):
    out = resolve_quantization("Qwen/Qwen2-7B-Instruct", level)
    assert out == expected
    # Reuses the SSOT map (not a re-declared copy with drifted values).
    assert out == QUANTIZATION_VLLM_PARAMS[level]


def test_resolve_quantization_is_base_independent():
    # The engine kwarg depends only on the level, not on which base model.
    assert resolve_quantization("Qwen/Qwen2-7B-Instruct", "fp8") == resolve_quantization(
        "meta-llama/Meta-Llama-3-8B-Instruct", "fp8"
    )


def test_resolve_quantization_returns_independent_copy():
    out = resolve_quantization("Qwen/Qwen2-7B-Instruct", "fp8")
    out["quantization"] = "MUTATED"
    # The shared SSOT map must be untouched.
    assert QUANTIZATION_VLLM_PARAMS["fp8"] == {"quantization": "fp8"}


def test_resolve_quantization_rejects_unknown_level():
    with pytest.raises(ValueError):
        resolve_quantization("Qwen/Qwen2-7B-Instruct", "int3_nonsense")


# ---------------------------------------------------------------------------
# resolve_abliteration — base model -> different weights repo (inverse of MAP)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("abliterated_repo,base", list(DATA_BACKED_ABLITERATED.items()))
def test_resolve_abliteration_maps_base_to_repo(abliterated_repo, base):
    assert resolve_abliteration(base) == abliterated_repo
    # Round-trips through the SSOT map the data was generated against.
    assert ABLITERATED_MODELS_MAP_TO_ORIGINAL[abliterated_repo] == base


def test_resolve_abliteration_rejects_non_abliteratable_base():
    with pytest.raises(ValueError):
        resolve_abliteration("CohereForAI/c4ai-command-r-plus")


@pytest.mark.skipif(
    not _RELEASED_LABELS_JSON.exists(), reason="released class-identity JSON absent"
)
def test_data_backed_abliterated_set_matches_released_json():
    """The 5 abliterated repos are exactly those in the released ``_ablit`` labels."""
    labels = set(json.loads(_RELEASED_LABELS_JSON.read_text()).values())
    ablit_repos = {label.removesuffix("_ablit") for label in labels if label.endswith("_ablit")}

    assert len(ablit_repos) == 5
    assert ablit_repos == set(DATA_BACKED_ABLITERATED)
    # Every released abliterated repo resolves back from its base via the resolver.
    for repo in ablit_repos:
        base = ABLITERATED_MODELS_MAP_TO_ORIGINAL[repo]
        assert resolve_abliteration(base) == repo


# ---------------------------------------------------------------------------
# Instance.label / Instance.storage_name — byte-for-byte legacy strings
# ---------------------------------------------------------------------------


def test_base_instance_label_and_storage():
    inst = Instance("CohereForAI/c4ai-command-r-plus", temperature=0.4, system_prompt_idx=-1)
    assert not inst.is_quantized
    assert inst.storage_name == "CohereForAI/c4ai-command-r-plus"
    assert inst.label == "CohereForAI/c4ai-command-r-plus_temp-0.4_sp--1"
    assert inst.quant_engine_kwargs == {}


def test_temperature_one_keeps_trailing_zero():
    inst = Instance("Qwen/Qwen2-7B-Instruct", temperature=1.0, system_prompt_idx=-1)
    assert inst.label == "Qwen/Qwen2-7B-Instruct_temp-1.0_sp--1"


def test_quantized_instance_fp8():
    inst = Instance(
        "Qwen/Qwen2-7B-Instruct", temperature=1.0, system_prompt_idx=-1, quantization="fp8"
    )
    assert inst.is_quantized
    assert inst.storage_name == "Qwen/Qwen2-7B-Instruct@@fp8"
    assert inst.label == "Qwen/Qwen2-7B-Instruct@@fp8_temp-1.0_sp--1"
    assert inst.quant_engine_kwargs == {"quantization": "fp8"}


def test_quantized_instance_bitsandbytes_int4():
    inst = Instance(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        temperature=0.6,
        system_prompt_idx=-1,
        quantization="bitsandbytes_int4",
    )
    # Storage suffix uses the quant KEY ...
    assert inst.storage_name == "meta-llama/Meta-Llama-3-8B-Instruct@@bitsandbytes_int4"
    assert inst.label == "meta-llama/Meta-Llama-3-8B-Instruct@@bitsandbytes_int4_temp-0.6_sp--1"
    # ... while the engine kwarg uses the vLLM VALUE.
    assert inst.quant_engine_kwargs == {"quantization": "bitsandbytes"}


def test_abliterated_instance_label_is_repo_ablit_with_no_var_suffix():
    base = "meta-llama/Llama-3.1-8B-Instruct"
    inst = Instance(base, temperature=1.0, system_prompt_idx=-1, abliterated=True)
    # storage_name / label are the abliterated REPO id, not the base.
    assert inst.storage_name == "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3"
    assert inst.label == "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3_ablit"
    # Byte-for-byte: no temp/sp suffix on abliterated labels.
    assert "temp-" not in inst.label
    assert "sp-" not in inst.label


@pytest.mark.parametrize("abliterated_repo,base", list(DATA_BACKED_ABLITERATED.items()))
def test_all_data_backed_abliterated_labels(abliterated_repo, base):
    inst = Instance(base, temperature=1.0, system_prompt_idx=-1, abliterated=True)
    assert inst.label == f"{abliterated_repo}_ablit"


def test_abliterated_and_quantized_is_rejected():
    with pytest.raises(ValueError):
        Instance(
            "meta-llama/Llama-3.1-8B-Instruct",
            temperature=1.0,
            system_prompt_idx=-1,
            quantization="fp8",
            abliterated=True,
        )


# ---------------------------------------------------------------------------
# Round-trip oracle: produced labels parse back through model_filtering.py
# ---------------------------------------------------------------------------


def test_base_label_round_trips():
    inst = Instance("CohereForAI/c4ai-command-r-plus", temperature=0.4, system_prompt_idx=-1)
    assert (
        full_var_model_name_to_original_model_name(inst.label)
        == "CohereForAI/c4ai-command-r-plus"
    )
    assert full_var_model_name_to_var_name(inst.label) == "temp-0.4_sp--1"


def test_quantized_label_round_trips():
    inst = Instance(
        "Qwen/Qwen2-7B-Instruct", temperature=1.0, system_prompt_idx=-1, quantization="fp8"
    )
    assert full_var_model_name_to_original_model_name(inst.label) == "Qwen/Qwen2-7B-Instruct"
    assert full_var_model_name_to_var_name(inst.label) == "fp8_temp-1.0_sp--1"


def test_abliterated_label_round_trips():
    base = "meta-llama/Llama-3.1-8B-Instruct"
    inst = Instance(base, temperature=1.0, system_prompt_idx=-1, abliterated=True)
    assert full_var_model_name_to_original_model_name(inst.label) == base


# ---------------------------------------------------------------------------
# Variation -> Instance binding
# ---------------------------------------------------------------------------


def test_variation_bind_produces_equivalent_instance():
    var = Variation(temperature=0.8, system_prompt_idx=3)
    inst = var.bind("google/gemma-2-9b-it")
    assert inst == Instance("google/gemma-2-9b-it", temperature=0.8, system_prompt_idx=3)
    assert inst.label == "google/gemma-2-9b-it_temp-0.8_sp-3"
    assert var.quantization == NO_QUANTIZATION
