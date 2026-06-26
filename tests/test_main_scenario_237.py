# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Definition-consistency + real-data tie test for the shipped scenarios.

This is NOT a data-free test. It pins two things at once:

(a) **Definition consistency** — ``config/scenarios/main.yaml`` enumerates exactly
    237 instances with the 200 / 32 / 5 base / quantized / abliterated split, and
    the load-bearing label forms (``@@fp8``, ``@@bitsandbytes_int4``,
    ``{repo}_ablit``, ``temp-<t>_sp-<sp>``) are produced byte-for-byte.

(b) **Real-data tie** — the enumerator's 200 base + 5 abliterated labels equal the
    released **205-class identity** in
    ``XP_configs/e3_flips_vs_llmmap/llmmap_if_data/checkpoint_dir/new_var_models_idx.json``
    **EXACTLY as a set**. That file is ``{index -> label}``, so the
    comparison is against ``set(json.values())`` (the 205 label strings), never
    ``json.keys()`` (the integer indices ``"0".."204"``).

This tie is the deterministic **Layer 1** anchor of the golden repro gate:
the enumerator output is byte-stable, so the assertion is EXACT and must never be
relaxed to a subset / tolerance.

The cross500 / cross1000 scenarios are checked for their documented exact counts
and for the abliteration pin (no crossing beyond temp=1.0 / sp=-1). A schema-guard
test confirms the loader rejects any abliteration cell outside that pin.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from audit_llm.scenarios.enumerator import build_instances
from audit_llm.scenarios.loader import load_scenario

_ROOT = Path(__file__).resolve().parents[1]
_RELEASED_205 = (
    _ROOT
    / "XP_configs"
    / "e3_flips_vs_llmmap"
    / "llmmap_if_data"
    / "checkpoint_dir"
    / "new_var_models_idx.json"
)


def _split(instances):
    base = [i for i in instances if not i.abliterated and not i.is_quantized]
    quant = [i for i in instances if i.is_quantized]
    ablit = [i for i in instances if i.abliterated]
    return base, quant, ablit


# ---------------------------------------------------------------------------
# (a) main.yaml — definition consistency: 237 = 200 / 32 / 5 + label forms
# ---------------------------------------------------------------------------


def test_main_total_and_split_is_237():
    instances = build_instances(load_scenario("main"))
    base, quant, ablit = _split(instances)
    assert len(instances) == 237
    assert (len(base), len(quant), len(ablit)) == (200, 32, 5)


def test_main_label_forms_spot_check():
    instances = build_instances(load_scenario("main"))
    labels = {i.label for i in instances}

    # quantization @@ suffix uses the quant KEY, with a temp/sp variation suffix.
    assert "Qwen/Qwen2-7B-Instruct@@fp8_temp-1.0_sp--1" in labels
    assert "Qwen/Qwen2-7B-Instruct@@bitsandbytes_int4_temp-1.0_sp-0" in labels
    # abliterated label is byte-identical {repo}_ablit with NO temp/sp suffix.
    assert "natong19/Qwen2-7B-Instruct-abliterated_ablit" in labels
    # a base variation label (note sp--1 for system_prompt_idx=-1).
    assert "CohereForAI/c4ai-command-r-plus_temp-1.0_sp--1" in labels


# ---------------------------------------------------------------------------
# (b) Real-data tie: 200 base + 5 abliterated == released 205-class identity
# ---------------------------------------------------------------------------


def test_main_ties_to_released_205_class_identity_exact():
    instances = build_instances(load_scenario("main"))
    base, quant, ablit = _split(instances)

    base_labels = {i.label for i in base}
    ablit_labels = {i.label for i in ablit}
    quant_labels = {i.label for i in quant}

    released = json.loads(_RELEASED_205.read_text())
    # The file maps integer-index strings -> label strings; tie against VALUES.
    assert sorted(released.keys()) != sorted(set(released.values()))  # keys are not labels
    released_labels = set(released.values())
    assert len(released_labels) == 205

    # EXACT set equality on both sides — no extras, no missing (golden Layer 1).
    assert base_labels | ablit_labels == released_labels
    assert len(base_labels) == 200
    assert len(ablit_labels) == 5

    # The 32 quantized (@@…) labels are NOT part of the 205-class identity.
    assert all("@@" in l for l in quant_labels)
    assert quant_labels.isdisjoint(released_labels)


# ---------------------------------------------------------------------------
# Schema guard: abliteration cell outside temp=1.0 / sp=-1 is rejected
# ---------------------------------------------------------------------------


def _write_min_scenario(tmp_path, abliteration_variations):
    """Write a minimal registry + scenario whose abliteration grid is overridable."""
    registry = {
        "base_models": ["Qwen/Qwen2-7B-Instruct"],
        "quantized_variants": [],
        "abliterated_variants": [
            {
                "base": "Qwen/Qwen2-7B-Instruct",
                "abliterated_hf_id": "natong19/Qwen2-7B-Instruct-abliterated",
            }
        ],
    }
    scenario = {
        "name": "ablit_guard",
        "base_models": ["Qwen/Qwen2-7B-Instruct"],
        "base_variations": [{"temperature": [1.0], "system_prompt_idx": [-1]}],
        "abliterated_models": ["natong19/Qwen2-7B-Instruct-abliterated"],
        "abliteration_variations": abliteration_variations,
    }
    registry_path = tmp_path / "models.yaml"
    scenario_path = tmp_path / "scenario.yaml"
    registry_path.write_text(yaml.safe_dump(registry))
    scenario_path.write_text(yaml.safe_dump(scenario))
    return scenario_path, registry_path


def test_abliteration_pin_is_satisfied_when_valid(tmp_path):
    scenario_path, registry_path = _write_min_scenario(
        tmp_path, [{"temperature": [1.0], "system_prompt_idx": [-1]}]
    )
    # The valid pinned cell loads cleanly.
    scenario = load_scenario(scenario_path, registry_path=registry_path)
    assert len(scenario.abliteration_variations) == 1


def test_abliteration_cell_with_bad_temperature_is_rejected(tmp_path):
    scenario_path, registry_path = _write_min_scenario(
        tmp_path, [{"temperature": [0.6], "system_prompt_idx": [-1]}]
    )
    with pytest.raises(ValueError, match="abliteration_variations"):
        load_scenario(scenario_path, registry_path=registry_path)


def test_abliteration_cell_with_bad_system_prompt_is_rejected(tmp_path):
    scenario_path, registry_path = _write_min_scenario(
        tmp_path, [{"temperature": [1.0], "system_prompt_idx": [0]}]
    )
    with pytest.raises(ValueError, match="abliteration_variations"):
        load_scenario(scenario_path, registry_path=registry_path)


# ---------------------------------------------------------------------------
# cross500 / cross1000 — load, build, documented exact counts + abliteration pin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, total, base_n, quant_n, ablit_n",
    [
        # Documented exact counts (see each scenario's header comment).
        ("cross500", 500, 375, 120, 5),
        ("cross1000", 995, 750, 240, 5),
    ],
)
def test_cross_scenarios_documented_counts(name, total, base_n, quant_n, ablit_n):
    instances = build_instances(load_scenario(name))
    base, quant, ablit = _split(instances)
    assert len(instances) == total
    assert (len(base), len(quant), len(ablit)) == (base_n, quant_n, ablit_n)


@pytest.mark.parametrize("name", ["cross500", "cross1000"])
def test_cross_scenarios_abliteration_stays_pinned(name):
    instances = build_instances(load_scenario(name))
    _, _, ablit = _split(instances)
    # Every abliterated instance is the single pinned (temp=1.0, sp=-1) cell.
    assert {(i.temperature, i.system_prompt_idx) for i in ablit} == {(1.0, -1)}
    assert all(i.label.endswith("_ablit") for i in ablit)
