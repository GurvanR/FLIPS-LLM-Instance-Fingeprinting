#!/usr/bin/env python3
"""Gate-B part (a) — the DATA-FREE class-parity firewall (always runs).

This is the firewall that lets the repo ship BOTH the new scenario *enumerator*
(``build_instances``) and the legacy class-materialisation path side by side, and
that the manual headline tier gates the legacy deletion on: it proves the
two paths materialise the **exact same classification label set** — computed FROM
CONFIG ALONE, with NO ``main_dataset`` and NO archive.

  ENUMERATOR side (the new path)
    ``{i.label for i in build_instances(load_scenario("main"))}`` — 237 labels:
    base ``{model}_temp-<t>_sp-<sp>``, quantized ``{base}@@{quant}_temp-<t>_sp-<sp>``,
    abliterated ``{repo}_ablit``.

  LEGACY side (the soon-deletable path), built from genuinely INDEPENDENT sources
    so this is a real cross-check, not a restatement of ``Instance.label``:
      * base models + quantized ``@@`` aliases from the ``config/models.yaml`` registry;
      * abliterated repos from the legacy run config's ``abliterated_models``,
        INTERSECTED with the registry's data-backed ``abliterated_variants`` (so the
        config-listed-but-data-absent ``failspy/Phi-3-mini-128k-...-v3`` is excluded
        the same way the released 205-class identity excludes it — via the SSOT
        registry, not a hardcoded string);
      * the ``temp-<t>_sp-<sp>`` variation KEYS via the **legacy**
        ``compute_model_variations_indices`` fed a tiny config-derived synthetic
        DataFrame (one row per grid cell) — exercising the real legacy key-formatting
        code with no dataset.
    Legacy labels are then joined exactly as ``data_preparation.prepare_dataset_features``
    does: ``{storage}_{variation_key}`` (base/quant) and ``{repo}_ablit`` (abliterated).

EXACT SET equality (count + exact strings) is the firewall — consistent with the
the parity test (``tests/test_scenario_analysis_parity.py``) and the golden
Layer-1 tie. Label ORDER is NOT asserted: nothing consumes the label list by position
(sklearn sorts ``classes_`` alphabetically, so the confusion-matrix axes follow the
sorted label set, which is identical on both paths when the sets are equal); the golden
accuracy band (gate-golden Layer 3) catches any behavioural consequence anyway.

Needs ``audit_llm`` importable — run inside the poetry ``.venv`` (or pass
``PYTHON=.venv/bin/python``); a ``src/`` sys.path fallback also lets a non-editable
install work. Wired by the Makefile as ``make gate-b-parity`` and reused by
``scripts/gates/gate_b.py`` (``make gate-b``).
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Fallback so the gate imports audit_llm even without an editable install
# (mirrors run_experiments.py adding src/ to its own path).
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

MODELS_YAML = REPO_ROOT / "config" / "models.yaml"
# The historical legacy run config the main.yaml grids were ported verbatim from.
LEGACY_XP_CONFIG = (
    REPO_ROOT
    / "XP_configs"
    / "e1_closedset_headline"
    / "FLIPS_mix_tp_full.yaml"
)
SCENARIO = "main"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"gate-b-parity: required config not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"gate-b-parity: malformed YAML (expected a mapping): {path}")
    return data


def _variation_keys_via_legacy(grid: list[dict]) -> set[str]:
    """The ``temp-<t>_sp-<sp>`` keys the LEGACY path derives for a variation grid.

    Exercises the real ``compute_model_variations_indices`` against a synthetic,
    config-derived DataFrame (one row per ``(temperature, system_prompt_idx)`` cell,
    temperature cast to float so keys render ``temp-1.0`` not ``temp-1``). The keys
    depend only on the grid, not on any dataset values — so this is data-free.
    """
    import polars as pl

    from audit_llm.xp_tools.variation_context import compute_model_variations_indices

    cells: list[tuple[float, int]] = []
    seen: set[tuple[float, int]] = set()
    for group in grid:
        for temp, sp in itertools.product(
            group.get("temperature", []) or [], group.get("system_prompt_idx", []) or []
        ):
            cell = (float(temp), int(sp))
            if cell not in seen:
                seen.add(cell)
                cells.append(cell)

    synth_df = pl.DataFrame(
        {
            "Index": list(range(len(cells))),
            "temperature": [c[0] for c in cells],
            "system_prompt_idx": [c[1] for c in cells],
        }
    ).with_columns(
        pl.col("temperature").cast(pl.Float64),
        pl.col("system_prompt_idx").cast(pl.Int64),
    )
    return set(compute_model_variations_indices({}, synth_df, model_variations=grid).keys())


def legacy_label_set() -> tuple[set[str], tuple[int, int, int]]:
    """The label set the LEGACY path materialises, from config alone.

    Returns ``(labels, (n_base, n_quant, n_ablit))``.
    """
    models = _load_yaml(MODELS_YAML)
    legacy = _load_yaml(LEGACY_XP_CONFIG)

    base_models = list(models.get("base_models") or [])
    quant_aliases = [v["alias"] for v in (models.get("quantized_variants") or []) if v.get("alias")]
    data_backed_ablit = {
        v["abliterated_hf_id"]
        for v in (models.get("abliterated_variants") or [])
        if v.get("abliterated_hf_id")
    }

    base_keys = _variation_keys_via_legacy(legacy["model_variations"])
    quant_keys = _variation_keys_via_legacy(legacy["quantized_model_variations"])

    base_labels = {f"{m}_{k}" for m in base_models for k in base_keys}
    quant_labels = {f"{alias}_{k}" for alias in quant_aliases for k in quant_keys}
    # Legacy config lists 6 abliterated repos; only those data-backed in the registry
    # become classes (the 6th has no released data). Filter via the SSOT registry.
    ablit_labels = {
        f"{repo}_ablit"
        for repo in (legacy.get("abliterated_models") or [])
        if repo in data_backed_ablit
    }

    labels = base_labels | quant_labels | ablit_labels
    return labels, (len(base_labels), len(quant_labels), len(ablit_labels))


def enumerator_label_set() -> tuple[set[str], tuple[int, int, int]]:
    """The label set the new ENUMERATOR materialises for the ``main`` scenario.

    Returns ``(labels, (n_base, n_quant, n_ablit))``.
    """
    from audit_llm.scenarios.enumerator import build_instances
    from audit_llm.scenarios.loader import load_scenario

    instances = build_instances(load_scenario(SCENARIO))
    base = [i for i in instances if not i.abliterated and not i.is_quantized]
    quant = [i for i in instances if i.is_quantized]
    ablit = [i for i in instances if i.abliterated]
    return {i.label for i in instances}, (len(base), len(quant), len(ablit))


def check_parity() -> tuple[bool, str]:
    """Run the data-free firewall. Returns ``(ok, detail)`` (reusable by gate_b.py)."""
    enum_labels, enum_split = enumerator_label_set()
    legacy_labels, legacy_split = legacy_label_set()

    missing = sorted(enum_labels - legacy_labels)   # in enumerator, absent from legacy
    extra = sorted(legacy_labels - enum_labels)     # in legacy, absent from enumerator

    print(f"  enumerator : {len(enum_labels):>3} labels  "
          f"(base={enum_split[0]}, quant={enum_split[1]}, ablit={enum_split[2]})")
    print(f"  legacy     : {len(legacy_labels):>3} labels  "
          f"(base={legacy_split[0]}, quant={legacy_split[1]}, ablit={legacy_split[2]})")

    if enum_labels == legacy_labels:
        return True, f"EXACT set parity: {len(enum_labels)} labels identical (200 base + 32 quant + 5 ablit = 237 expected)"

    detail = []
    if missing:
        detail.append(f"{len(missing)} only in enumerator e.g. {missing[:5]}")
    if extra:
        detail.append(f"{len(extra)} only in legacy e.g. {extra[:5]}")
    return False, "LABEL-SET DRIFT — " + "; ".join(detail)


def main() -> int:
    print("=== gate-B part (a) : data-free scenario-vs-legacy class-parity firewall ===")
    print(f"  scenario={SCENARIO}  registry={MODELS_YAML.relative_to(REPO_ROOT)}  "
          f"legacy={LEGACY_XP_CONFIG.relative_to(REPO_ROOT)}")
    ok, detail = check_parity()
    print(f"  [{'PASS' if ok else 'FAIL'}] {detail}")
    print("=" * 60)
    print(f"gate-B-parity: {'PASS' if ok else 'FAIL'}")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
