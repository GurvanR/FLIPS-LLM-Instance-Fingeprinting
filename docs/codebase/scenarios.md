# Scenarios guide

Scenarios (`config/scenarios/*.yaml`) are declarative files that select which model × variation instances the **analysis layer** materialises as classification classes. Three scenarios are currently shipped.

For the YAML field contract, see `config/scenarios/README.md`. For the xp_config `scenario:` key, see [`configuration-advanced.md`](configuration-advanced.md).

---

## The three shipped scenarios

### `main` — the FLIPS paper default

**File:** `config/scenarios/main.yaml`

Enumerates **237 instances** in three groups:

| Group | Count | How |
|---|---|---|
| base | 200 | 25 base models × 8 variation cells |
| quantized | 32 | 4 quantized base models × 2 quant levels × 4 variation cells |
| abliterated | 5 | 5 data-backed abliterated repos × 1 cell (pinned `temp=1.0, sp=-1`) |
| **Total** | **237** | |

**Base variation grid** (cartesian within group, union across groups):
- Group A: `temp[0.4, 0.6, 0.8, 1.0] × sp[-1]` → 4 cells
- Group B: `temp[1.0] × sp[0, 3, 6, 7]` → 4 cells
- Union: 8 unique cells per base model → 25 × 8 = 200 instances

**Quantized variation grid:**
- Group A: `temp[0.6, 1.0] × sp[-1] × {fp8, bitsandbytes_int4}` → 4 cells × 2 = 4 (per model)
- Group B: `temp[1.0] × sp[0, 3] × {fp8, bitsandbytes_int4}` → 4 cells (per model)
- 4 quant base models × 8 quant cells = 32 instances

**25 base models:** the full universe from `config/models.yaml`.

**4 quantized base models:**
- `Qwen/Qwen2-7B-Instruct`
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `microsoft/Phi-3-mini-4k-instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

**Reproducibility anchor:** the 200 base + 5 abliterated instance labels produced by `build_instances(load_scenario("config/scenarios/main.yaml"))` match EXACTLY the released 205-class identity in `XP_configs/e3_flips_vs_llmmap/llmmap_if_data/checkpoint_dir/new_var_models_idx.json` (asserted by `tests/test_main_scenario_237.py`). The 32 quantized labels are disjoint from that 205-set.

---

### `cross500` — extended crossing (~500 instances)

**File:** `config/scenarios/cross500.yaml`

A broader scenario crossing more temperature and system-prompt combinations. Targets 500 instances with a realistic temp/sp grid and the same 25 base models + 4 quant bases. The exact count and variation grid are documented in the file's header comment.

---

### `cross1000` — wide crossing (~1000 instances)

**File:** `config/scenarios/cross1000.yaml`

A scenario extending the temperature axis with additional values (including illustrative non-data-backed temperatures) to approach 1000 instances. This scenario is **not a reproduction target** — its illustrative temperatures have no released inference data. The exact count and caveats are documented in the file's header comment.

---

## Abliterated repos

### 5 data-backed repos

These five repos have released inference data and are listed in `config/scenarios/main.yaml`:

| HF repo | Base model |
|---|---|
| `failspy/Meta-Llama-3-8B-Instruct-abliterated-v3` | `meta-llama/Meta-Llama-3-8B-Instruct` |
| `failspy/Smaug-Llama-3-70B-Instruct-abliterated-v3` | `abacusai/Smaug-Llama-3-70B-Instruct` |
| `natong19/Qwen2-7B-Instruct-abliterated` | `Qwen/Qwen2-7B-Instruct` |
| `failspy/Phi-3-medium-4k-instruct-abliterated-v3` | `microsoft/Phi-3-medium-4k-instruct` |
| `dphn/dolphin-2.9.2-Phi-3-Medium-abliterated` | `microsoft/Phi-3-medium-4k-instruct` |

### The documented 6th repo (config-listed but absent)

`failspy/Phi-3-mini-128k-instruct-abliterated-v3` appears in the historical experiment config (`XP_configs/e1_closedset_headline/FLIPS_mix_tp_full.yaml`) and in `config/models.yaml`, but **no inference data was ever generated for it**. It is therefore excluded from `main.yaml` (and all three shipped scenarios). A header comment in `main.yaml` documents its absence so the offline build self-consistently lands on 5 abliterated instances with no data-universe filter needed.

---

## The no-crossing rule for abliteration

Abliterated instances are **pinned** to `temperature=1.0, system_prompt_idx=-1`. No released inference data exists for any other `(temperature, system_prompt_idx)` cell for abliterated models. The scenario schema rejects any abliteration variation group that specifies `temperature != 1.0` or `system_prompt_idx != -1` with a clear error.

This means abliterated instances never contribute multiple variation cells per repo — each abliterated repo produces exactly **one** instance in any scenario. The abliteration label is `{abliterated_repo_hf_id}_ablit` with **no** `temp-x_sp-y` suffix (see [Label stability](#label-stability) below).

---

## CSV invariant

The scenario layer governs **model × variation selection only**. It never emits, regenerates, or owns any prompt-request CSV.

`datasets/Bits_Datasets/*.csv` files are consumed **unchanged** by the analysis pipeline. Their SHA-256 hash (computed by `compute_file_hash` in `audit_llm.file_io`, exposed as `source_csv_hash` in `audit_llm.data_loader`) gates the feature cache — any byte change would silently invalidate the released cache. Scenarios must never write to that directory.

---

## Label stability

Instance labels produced by the scenario enumerator are consumed by the downstream parsing layer (`xp_tools.model_filtering`). They must remain byte-stable across scenarios. The three load-bearing forms are:

| Form | Example | Notes |
|---|---|---|
| `@@<quant_key>` (storage suffix) | `Qwen/Qwen2-7B-Instruct@@fp8` | `@@` = `QUANTIZATION_SEPARATOR`; suffix is the quant *key*, not the vLLM engine kwarg value |
| `{abliterated_repo}_ablit` (full label) | `failspy/Meta-Llama-3-8B-Instruct-abliterated-v3_ablit` | `{abliterated_repo}` is the abliterated HF id, not the base; **no** `temp-x_sp-y` suffix |
| `temp-{t}_sp-{sp}` (variation suffix) | `temp-1.0_sp--1` | `sp--1` for `sp=-1`; trailing `.0` in temperature is preserved |

See also the conventions note in [`conventions.md`](conventions.md).
