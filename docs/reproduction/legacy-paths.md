# Legacy vs. scenario analysis paths — both ship

**Status: BOTH paths ship.** The class-materialization logic that drives the analysis
layer exists in two forms, and the public repo deliberately keeps both. This page
explains what they are, why both are present, and the single condition under which the
legacy one is removed.

## The two paths

The analysis layer needs to turn the run's `(model, temperature, system_prompt,
quantization, abliteration)` cells into the labelled classes a classifier trains on.
Two implementations produce that mapping:

1. **Scenario enumerator (new, preferred).** A declarative scenario file
   (`config/scenarios/main.yaml`, etc.) is expanded by `build_instances` /
   `build_analysis_variation_structures` in
   [`src/audit_llm/scenarios/enumerator.py`](../../src/audit_llm/scenarios/enumerator.py).
   It is the single source of truth for the model×variation cross-product (e.g. `main` =
   237 instances: 200 base + 32 quantized + 5 data-backed abliterated).

2. **Legacy hardcoded path (retained, bypassed on the scenario path).** The historical
   `compute_model_variations_indices`
   ([`src/audit_llm/xp_tools/variation_context.py`](../../src/audit_llm/xp_tools/variation_context.py))
   for the base/quant grid, plus a **hardcoded abliteration pin** in
   [`src/audit_llm/xp_tools/data_preparation.py`](../../src/audit_llm/xp_tools/data_preparation.py)
   (`abliterated_samples_indices = main_dataset.filter((temperature == 1.0) &
   (system_prompt_idx == -1))`, near `data_preparation.py:193-207`).

## How they coexist (no duplication of effect)

The selector lives in `experiment_runner.py` (`_xp_config_init`):

- If the `xp_config` carries a **`scenario:`** key, the enumerator runs and the legacy
  branch is **not** taken. The scenario also supplies the abliterated
  `(temperature, system_prompt_idx)`, so `data_preparation.py` takes its
  `abliterated_variation is not None` branch and the hardcoded `temp=1.0 / sp=-1` pin is
  bypassed.
- If there is **no** `scenario:` key, the run falls back to the legacy
  `model_variations` / `quantized_model_variations` / `abliterated_models` keys, and the
  hardcoded abliteration pin resolves the abliterated samples.

The two paths are kept in exact parity by the **data-free firewall**
[`scripts/gates/gate_b_parity.py`](../../scripts/gates/gate_b_parity.py) (`make
gate-b-parity`), which asserts the enumerator and the legacy path emit the **identical
label set** (237 = 200 base + 32 quant + 5 ablit) from config alone — no archive needed.
That gate runs on every clone and guarantees the two paths cannot silently diverge while
both ship.

## Why the legacy path is still here

The plan is to collapse to the single scenario path. That collapse is **gated on gate-B
passing on the real headline run**, and gate-B has two parts:

- **part (a)** — the data-free class-parity firewall above — passes everywhere, including
  on a fresh clone with no data.
- **part (b)** — a coarse-banded accuracy sanity check anchored to the headline
  `results.json` — needs the **off-disk headline export `FLiPS_ICML_run`**. That export
  is not present on the automated CPU build machine, so part (b) **SKIPs** (loudly, never
  a false pass) in the orchestrated flow.

Deleting the legacy path before part (b) has actually passed on real headline data would
remove the fallback while the new path is still only *parity-verified*, not
*accuracy-verified end-to-end against the headline numbers*. So the deletion is
deliberately **deferred to a manual step**, and until then **both paths ship**.

## When the legacy path gets deleted

The deletion is a **by-hand operator step**. The operator runs it only after:

1. ingesting the HPC `FLiPS_ICML_run` headline export locally (flattened under
   `Productions/FLiPS_ICML_run/`), and
2. `make gate-b` passing **both** part (a) **and** part (b) on that real headline data.

At that point the operator deletes the hardcoded abliteration branch in
`data_preparation.py`, the legacy `compute_model_variations_indices`, and any now-dead
legacy helpers, leaving `build_instances` as the single path — then re-runs `make smoke
&& make gate-a && make gate-b-parity && make gate-b` to confirm nothing regresses and
updates this page to say the legacy path was removed.

If gate-B part (b) does **not** pass, the operator does **not** delete: both paths stay
shipped (the default state described here).
