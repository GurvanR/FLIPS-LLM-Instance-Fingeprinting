# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Parity tests for the scenario-driven analysis path.

The scenario enumerator (``build_instances`` + ``build_analysis_variation_structures``)
must produce the SAME ``new_models_idx`` classification labels as the legacy
``model_variations`` / ``quantized_model_variations`` / ``abliterated_models``
path when fed an equivalent grid — including the byte-identical ``{repo}_ablit``
and ``@@<quant>`` label forms.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import yaml

from audit_llm.scenarios.enumerator import build_analysis_variation_structures, build_instances
from audit_llm.scenarios.loader import load_scenario
from audit_llm.xp_tools.data_preparation import prepare_dataset_features
from audit_llm.xp_tools.variation_context import compute_model_variations_indices

# ---------------------------------------------------------------------------
# Shared fixture: a tiny but representative model × variation universe.
# ---------------------------------------------------------------------------

_TOKEN_PAIR = "tp"

# 2 base + 2 @@-quantized + 1 abliterated, in the order they live in model_idx.
_MODEL_IDX = {
    "CohereForAI/c4ai-command-r-plus": 0,
    "google/gemma-2-9b-it": 1,
    "Qwen/Qwen2-7B-Instruct@@fp8": 2,
    "Qwen/Qwen2-7B-Instruct@@bitsandbytes_int4": 3,
    "natong19/Qwen2-7B-Instruct-abliterated": 4,
}
# The main expansion slices base + quantized positions; the abliterated model is
# concatenated separately (looked up in model_idx by name).
_MODELS_INDICES = [0, 1, 2, 3]

_ABLITERATED_REPO = "natong19/Qwen2-7B-Instruct-abliterated"

# Legacy grids (list-of-dicts), temp/sp only — quantization is encoded in @@ names.
_BASE_GRID = [
    {"temperature": [0.4, 1.0], "system_prompt_idx": [-1]},
    {"temperature": [1.0], "system_prompt_idx": [0]},
]
_QUANT_GRID = [
    {"temperature": [1.0], "system_prompt_idx": [-1]},
    {"temperature": [1.0], "system_prompt_idx": [0]},
]

# The expected 11 classification labels (2 base × 3 var + 2 quant × 2 var + 1 ablit).
_EXPECTED_LABELS = {
    "CohereForAI/c4ai-command-r-plus_temp-0.4_sp--1",
    "CohereForAI/c4ai-command-r-plus_temp-1.0_sp--1",
    "CohereForAI/c4ai-command-r-plus_temp-1.0_sp-0",
    "google/gemma-2-9b-it_temp-0.4_sp--1",
    "google/gemma-2-9b-it_temp-1.0_sp--1",
    "google/gemma-2-9b-it_temp-1.0_sp-0",
    "Qwen/Qwen2-7B-Instruct@@fp8_temp-1.0_sp--1",
    "Qwen/Qwen2-7B-Instruct@@fp8_temp-1.0_sp-0",
    "Qwen/Qwen2-7B-Instruct@@bitsandbytes_int4_temp-1.0_sp--1",
    "Qwen/Qwen2-7B-Instruct@@bitsandbytes_int4_temp-1.0_sp-0",
    "natong19/Qwen2-7B-Instruct-abliterated_ablit",
}


def _main_dataset_df() -> pl.DataFrame:
    """Index/temperature/system_prompt_idx with 2 rows per (temp, sp) cell."""
    return pl.DataFrame(
        {
            "Index": [0, 1, 2, 3, 4, 5],
            "temperature": [0.4, 0.4, 1.0, 1.0, 1.0, 1.0],
            "system_prompt_idx": [-1, -1, -1, -1, 0, 0],
        }
    )


def _feature_inputs():
    """Deterministic feature tensor + index/selection dicts for prepare_dataset_features."""
    n_samples, n_models, n_features = 6, len(_MODEL_IDX), 2
    X = np.arange(n_samples * n_models * n_features, dtype=float).reshape(
        n_samples, n_models, n_features
    )
    intra_samples_features_dict = {_TOKEN_PAIR: X}
    intra_samples_feature_index_dict = {_TOKEN_PAIR: {"f0": 0, "f1": 1}}
    selected_features = {"f0": 0, "f1": 1}
    return intra_samples_features_dict, intra_samples_feature_index_dict, selected_features


def _run_prepare(experiment_config, xp_config) -> dict:
    df = _main_dataset_df()
    intra_features, intra_index, selected = _feature_inputs()
    _, new_models_idx = prepare_dataset_features(
        token_pair=_TOKEN_PAIR,
        calculation_item={"all": None},
        intra_samples_features_dict=intra_features,
        intra_samples_feature_index_dict=intra_index,
        selected_features=selected,
        new_models_idx={},
        models_indices=list(_MODELS_INDICES),
        xp_config=xp_config,
        Experiment_config=experiment_config,
        MainDataset_df_iterators=df,
        Answers_df=df,
    )
    return new_models_idx


def _write_scenario(tmp_path):
    """Write a temp registry + scenario YAML mirroring the legacy grid exactly."""
    registry = {
        "base_models": ["CohereForAI/c4ai-command-r-plus", "google/gemma-2-9b-it"],
        "quantized_variants": [
            {"base": "Qwen/Qwen2-7B-Instruct", "quantization": "fp8"},
            {"base": "Qwen/Qwen2-7B-Instruct", "quantization": "bitsandbytes_int4"},
        ],
        "abliterated_variants": [
            {"base": "Qwen/Qwen2-7B-Instruct", "abliterated_hf_id": _ABLITERATED_REPO},
        ],
    }
    scenario = {
        "name": "test_parity",
        "base_models": ["CohereForAI/c4ai-command-r-plus", "google/gemma-2-9b-it"],
        "base_variations": [
            {"temperature": [0.4, 1.0], "system_prompt_idx": [-1]},
            {"temperature": [1.0], "system_prompt_idx": [0]},
        ],
        "quantized_base_models": ["Qwen/Qwen2-7B-Instruct"],
        "quantized_variations": [
            {"temperature": [1.0], "system_prompt_idx": [-1], "quantization": ["fp8", "bitsandbytes_int4"]},
            {"temperature": [1.0], "system_prompt_idx": [0], "quantization": ["fp8", "bitsandbytes_int4"]},
        ],
        "abliterated_models": [_ABLITERATED_REPO],
        "abliteration_variations": [
            {"temperature": [1.0], "system_prompt_idx": [-1]},
        ],
    }
    registry_path = tmp_path / "models.yaml"
    scenario_path = tmp_path / "scenario.yaml"
    registry_path.write_text(yaml.safe_dump(registry))
    scenario_path.write_text(yaml.safe_dump(scenario))
    return scenario_path, registry_path


# ---------------------------------------------------------------------------
# build_instances — count, split, labels
# ---------------------------------------------------------------------------


def test_build_instances_count_split_and_labels(tmp_path):
    scenario_path, registry_path = _write_scenario(tmp_path)
    scenario = load_scenario(scenario_path, registry_path=registry_path)
    instances = build_instances(scenario)

    base = [i for i in instances if not i.abliterated and not i.is_quantized]
    quant = [i for i in instances if i.is_quantized]
    ablit = [i for i in instances if i.abliterated]
    assert (len(base), len(quant), len(ablit)) == (6, 4, 1)
    assert len(instances) == 11

    assert {i.label for i in instances} == _EXPECTED_LABELS
    # Abliterated label is byte-identical {repo}_ablit with no temp/sp suffix.
    (ablit_inst,) = ablit
    assert ablit_inst.label == f"{_ABLITERATED_REPO}_ablit"


def test_analysis_structures_grids_and_pin(tmp_path):
    scenario_path, registry_path = _write_scenario(tmp_path)
    scenario = load_scenario(scenario_path, registry_path=registry_path)
    structures = build_analysis_variation_structures(scenario, _main_dataset_df())

    assert set(structures.model_variations_indices) == {
        "temp-0.4_sp--1",
        "temp-1.0_sp--1",
        "temp-1.0_sp-0",
    }
    assert set(structures.quantized_model_variations_indices) == {
        "temp-1.0_sp--1",
        "temp-1.0_sp-0",
    }
    assert structures.abliterated_models == (_ABLITERATED_REPO,)
    assert structures.abliterated_variation == (1.0, -1)


# ---------------------------------------------------------------------------
# Parity: scenario path vs legacy path yield identical new_models_idx labels
# ---------------------------------------------------------------------------


def test_scenario_and_legacy_new_models_idx_labels_match(tmp_path):
    df = _main_dataset_df()

    # --- Legacy path: grids via compute_model_variations_indices, hardcoded pin.
    legacy_mvi = compute_model_variations_indices({}, df, model_variations=_BASE_GRID)
    legacy_qmvi = compute_model_variations_indices({}, df, model_variations=_QUANT_GRID)
    legacy_xp_config = {"abliterated_models": [_ABLITERATED_REPO]}
    legacy_experiment_config = {
        "global_samples_indices": [0, 1, 2, 3, 4, 5],
        "model_variations_indices": legacy_mvi,
        "quantized_model_variations_indices": legacy_qmvi,
        "model_idx": dict(_MODEL_IDX),
        # No "abliterated_variation" → legacy hardcoded (temp==1.0)&(sp==-1) filter.
    }
    legacy_labels = _run_prepare(legacy_experiment_config, legacy_xp_config)

    # --- Scenario path: grids + pin from build_analysis_variation_structures.
    scenario_path, registry_path = _write_scenario(tmp_path)
    scenario = load_scenario(scenario_path, registry_path=registry_path)
    structures = build_analysis_variation_structures(scenario, df)
    scenario_xp_config = {"abliterated_models": list(structures.abliterated_models)}
    scenario_experiment_config = {
        "global_samples_indices": [0, 1, 2, 3, 4, 5],
        "model_variations_indices": structures.model_variations_indices,
        "quantized_model_variations_indices": structures.quantized_model_variations_indices,
        "model_idx": dict(_MODEL_IDX),
        "abliterated_variation": structures.abliterated_variation,
    }
    scenario_labels = _run_prepare(scenario_experiment_config, scenario_xp_config)

    # Same count and same exact label strings.
    assert len(scenario_labels) == len(legacy_labels) == 11
    assert set(scenario_labels.values()) == set(legacy_labels.values()) == _EXPECTED_LABELS
    # The abliterated class is present and byte-identical in both paths.
    assert f"{_ABLITERATED_REPO}_ablit" in set(scenario_labels.values())


def test_legacy_path_unaffected_without_scenario(tmp_path):
    """Sanity: the legacy (no-scenario) path still produces the abliterated class."""
    df = _main_dataset_df()
    legacy_experiment_config = {
        "global_samples_indices": [0, 1, 2, 3, 4, 5],
        "model_variations_indices": compute_model_variations_indices({}, df, model_variations=_BASE_GRID),
        "quantized_model_variations_indices": compute_model_variations_indices({}, df, model_variations=_QUANT_GRID),
        "model_idx": dict(_MODEL_IDX),
    }
    labels = _run_prepare(legacy_experiment_config, {"abliterated_models": [_ABLITERATED_REPO]})
    assert set(labels.values()) == _EXPECTED_LABELS
