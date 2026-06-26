# Provenance

Where the numbers, models, prompts, and released archives in this repository come from. If you only
want to *run* something, see [`reproduction.md`](reproduction.md); this document explains what the
pipeline is built on and why the headline count is **237**.

---

## 1. The provenance chain — 34 → 205 → 237

The three counts you will see throughout the paper and the configs are stages of one funnel, not
independent figures.

**34 — the candidate vocabulary.**
`Productions/Intersection_vocab/Intersection_of_34_models.json` lists **34 candidate model names**.
This is the *universe* from which the FLIPS experiment pool was drawn — the set of models whose
tokenizers were intersected to build the shared token-pair vocabulary the method probes. It is tracked
in-repo (not archived) because it is an **import-time dependency** of the analysis layer; a fresh clone
must have it to enumerate instances. Not every one of the 34 made it into the released runs.

**205 — the LLMmap reference identity.**
The LLMmap baseline run (`LLMmap_ICML_run`) classifies a **205-class identity**. This 205-class set is
the reference evaluation frame for the Mode A analyses: the FLIPS base + abliterated label set is
reconciled against it. Concretely, the 200 base + 5 abliterated instance labels produced by the
scenario enumerator match **exactly** the released 205-class identity in
`XP_configs/e3_flips_vs_llmmap/llmmap_if_data/checkpoint_dir/new_var_models_idx.json` (asserted by
`tests/test_main_scenario_237.py`).

**237 — the FLIPS headline instance count.**
The paper's headline pool is **237 instances**, decomposed as **200 base + 32 quantized + 5
abliterated** (see the [decomposition table](#3-the-237-instance-decomposition) below). The 32
quantized labels are *disjoint* from the 205-class identity — that is why 205 (= 200 base + 5
abliterated) and 237 (= 205 + 32 quantized) are both correct, at different layers.

### Soft per-run filter — `Black_list_models_in_runs.json`

`Productions/Black_list_models_in_runs.json` is a **per-run exclusion list applied at analysis time**,
keyed by run name (`FLiPS_ICML_run`, `LLMmap_ICML_run`, …). Some runs omit a
handful of models — typically because inference failed, was incomplete, or the model fell out of the
intersection vocabulary for that run. This is a **soft filter**: it is data-driven and per-run, not a
fixed, hard-coded exclusion baked into the model universe. The same model may be present in one run and
black-listed in another. Treat it as a record of *which models a given run could not use*, not as a
permanent removal from the candidate set.

---

## 2. The 25 base LLMs

The reproducible benchmark set is **25 base models**, addressed by Hugging Face Hub id. The model
universe source of truth is `config/models.yaml`.

| Model (short name) | HF id | License | Notes |
|---|---|---|---|
| Command-R+ | `CohereForAI/c4ai-command-r-plus` | CC-BY-NC | **Outputs CC-BY-NC** (non-commercial) — see the per-model output constraints below |
| Command-R v01 | `CohereForAI/c4ai-command-r-v01` | CC-BY-NC | **Outputs CC-BY-NC** (non-commercial) |
| Aya-23-35B | `CohereForAI/aya-23-35B` | CC-BY-NC | Cohere family — **outputs CC-BY-NC** |
| Aya-23-8B | `CohereForAI/aya-23-8B` | CC-BY-NC | Cohere family — **outputs CC-BY-NC** |
| Zephyr-7B-β | `HuggingFaceH4/zephyr-7b-beta` | MIT | Fine-tune of Mistral-7B |
| Nous-Hermes-2-Mixtral-DPO | `NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO` | Apache 2.0 | Mixtral-8x7B base |
| Qwen2-1.5B-Instruct | `Qwen/Qwen2-1.5B-Instruct` | Apache 2.0 | |
| Qwen2-7B-Instruct | `Qwen/Qwen2-7B-Instruct` | Apache 2.0 | Also a quantized + abliterated base |
| Qwen2-72B-Instruct | `Qwen/Qwen2-72B-Instruct` | Apache 2.0 | |
| Qwen3-30B-A3B-Instruct | `Qwen/Qwen3-30B-A3B-Instruct` | Apache 2.0 | MoE |
| Smaug-Llama-3-70B-Instruct | `abacusai/Smaug-Llama-3-70B-Instruct` | Apache 2.0 | Built on Llama-3; consult Meta Llama terms for derivatives |
| Gemma-2-9B-it | `google/gemma-2-9b-it` | Gemma Terms of Use | **Outputs under the Gemma Terms of Use** — see the per-model output constraints below |
| Gemma-2-27B-it | `google/gemma-2-27b-it` | Gemma Terms of Use | **Outputs under the Gemma Terms of Use** |
| Llama-3-8B-Gradient-1048k | `gradientai/Llama-3-8B-Instruct-Gradient-1048k` | Meta Llama License | Llama-3 long-context fine-tune |
| Meta-Llama-3-8B-Instruct | `meta-llama/Meta-Llama-3-8B-Instruct` | Meta Llama License | Also a quantized + abliterated base |
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | Meta Llama License | |
| Meta-Llama-3.1-70B-Instruct | `meta-llama/Meta-Llama-3.1-70B-Instruct` | Meta Llama License | |
| Phi-3-mini-4k-instruct | `microsoft/Phi-3-mini-4k-instruct` | MIT | Also a quantized base |
| Phi-3-mini-128k-instruct | `microsoft/Phi-3-mini-128k-instruct` | MIT | |
| Phi-3-medium-4k-instruct | `microsoft/Phi-3-medium-4k-instruct` | MIT | Also an abliterated base |
| Phi-3-medium-128k-instruct | `microsoft/Phi-3-medium-128k-instruct` | MIT | |
| Mistral-7B-Instruct-v0.1 | `mistralai/Mistral-7B-Instruct-v0.1` | Apache 2.0 | |
| Mistral-7B-Instruct-v0.2 | `mistralai/Mistral-7B-Instruct-v0.2` | Apache 2.0 | |
| Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | Apache 2.0 | Also a quantized base |
| SOLAR-10.7B-Instruct | `upstage/SOLAR-10.7B-Instruct-v1.0` | Apache 2.0 | |

### The 5 data-backed abliterated instances

Each abliterated instance is a community "uncensored" fork of one of the base models above. Listed as
**base model row + abliterated weights (HF id)**:

| Abliterated weights (HF id) | Base model | License (inherits base) |
|---|---|---|
| `failspy/Meta-Llama-3-8B-Instruct-abliterated-v3` | `meta-llama/Meta-Llama-3-8B-Instruct` | Meta Llama License |
| `failspy/Smaug-Llama-3-70B-Instruct-abliterated-v3` | `abacusai/Smaug-Llama-3-70B-Instruct` | Apache 2.0 (Llama-3 base) |
| `natong19/Qwen2-7B-Instruct-abliterated` | `Qwen/Qwen2-7B-Instruct` | Apache 2.0 |
| `failspy/Phi-3-medium-4k-instruct-abliterated-v3` | `microsoft/Phi-3-medium-4k-instruct` | MIT |
| `dphn/dolphin-2.9.2-Phi-3-Medium-abliterated` | `microsoft/Phi-3-medium-4k-instruct` | MIT |

> A 6th abliterated repo, `failspy/Phi-3-mini-128k-instruct-abliterated-v3`, appears in the historical
> config and in `config/models.yaml` but **no inference data was ever generated for it** — so it is
> excluded from all shipped scenarios and the count lands on **5**. See
> [`codebase/scenarios.md`](codebase/scenarios.md) for details.

### Per-model output-use constraints

The released model **OUTPUTS** carry each model's own terms **on top of** the blanket data license.
State this plainly:

- **CC-BY-4.0** (the repository's data license) covers **our** annotations and derived features —
  feature tables, scenario labels, classification scores, figures.
- The **raw generations** themselves are still bound by **each source model's own terms**. Most are
  permissive (Apache 2.0 / MIT), but two families are stricter:
  - **Cohere Command-R / Aya outputs are CC-BY-NC** — *non-commercial use only*.
  - **Gemma outputs are governed by the Gemma Terms of Use.**
- Reusers must consult the **License** column of the table above before redistributing or building on
  the raw generations. The CC-BY-4.0 tag on the data archives does **not** override these per-model
  output terms.

---

## 3. The 237-instance decomposition

| Tier | Count | Description |
|------|-------|-------------|
| Base | 200 | 25 models × 8 variations = **(4 temps 0.4/0.6/0.8/1.0 at sp=-1) ∪ (4 system prompts sp=0/3/6/7 at temp=1.0)** — a UNION of 4+4, **NOT** a 4×4 cross |
| Quantized | 32 | 4 base models × 2 quant levels (`fp8`, `bitsandbytes_int4`) × 4 variations = (temps 0.6/1.0 at sp=-1) ∪ (sp 0/3 at temp=1.0). |
| Abliterated | 5 | 5 community-abliterated instances, temp=1.0, sp=-1 |
| **Total** | **237** | |

The 8 base variations are a **union of two groups of 4**, not a 4-temperature × 4-system-prompt cross
(which would be 16). The 4 quantized base models are `Qwen/Qwen2-7B-Instruct`,
`meta-llama/Meta-Llama-3-8B-Instruct`, `microsoft/Phi-3-mini-4k-instruct`, and
`mistralai/Mistral-7B-Instruct-v0.3`.

> **Note — the smoke fixture uses a *different* grid.** The in-repo `FLiPS_ICML_light_subset` smoke
> fixture is carved from a small dev slice that uses temps `0.4/0.7/1.0` and system-prompt indices
> `0–3/6/7`. That grid does **not** appear in the 237-instance paper decomposition; smoke-fixture
> outputs are indicative, not paper-replicating.

---

## 4. The three scenarios

Scenarios (`config/scenarios/*.yaml`) select which model × variation instances the analysis layer
materializes as classification classes. See [`codebase/scenarios.md`](codebase/scenarios.md) for the
YAML contract; the catalogue:

**`main` — the paper setting (237 instances).** The default scenario. Enumerates exactly the
200 base + 32 quantized + 5 abliterated instances of the [decomposition above](#3-the-237-instance-decomposition).
This is the configuration the headline E1 (96% closed-set) and E2 (90% open-set) numbers are computed
on, and the one the released `FLiPS_ICML_run` data backs.

**`cross500` — extended crossing (~500 instances).** A broader scenario that crosses more
temperature and system-prompt combinations over the *same* real grid (≈375 base + 120 quantized + 5
abliterated). It stresses the method on a wider instance set while staying within data-backed
temperatures.

**`cross1000` — wide crossing (~1000 instances).** A scaling-stress scenario that extends the
temperature axis with additional, *illustrative* values (e.g. `0.2`/`1.2`) that have **no released
inference data**. It is **not a reproduction target** — it exists to exercise the enumerator and
classifier at ~1000 classes, not to reproduce a paper result.

---

## 5. Dataset and prompt provenance

The prompt-side inputs are versioned in-repo under `datasets/`, generated by the codebase.

- **`datasets/Bits_Datasets/*.csv` are prompt-*request* tables — not feature tables.** Each row is one
  inference *request*, with columns `Index, prompt_idx, system_prompt_idx, temperature,
  frequency_penalty`. (`FLiPS_ICML.csv` / `FLiPS_ICML_uncensored.csv` for the FLIPS prompt set;
  `LLMmap_ICML.csv` for the LLMmap baseline.) They describe *what to ask*, not *what came back*.
- **Prompt assembly** is governed by `datasets/prompt_config_index.yaml` (the prompt-index → prompt
  mapping) together with `datasets/system_prompts.json` (the system-prompt bank, indexed by
  `system_prompt_idx`; `sp=-1` means *no system prompt*).
- These files are **consumed unchanged** by the analysis pipeline. Each CSV's SHA-256 hash gates the
  feature cache (exposed as `source_csv_hash`); any byte change silently invalidates the released
  cache, so the scenario layer never writes to `datasets/`.

---

## 6. Data hosting

The released archives mirror `data/manifest.sha256`. They are **heterogeneous
sources** of different sizes, not one monolithic bundle:

| Member | Size | Role |
|---|---|---|
| `Productions/FLiPS_ICML_light_subset/` | ~92 MB | In-repo zero-download CPU smoke subset (5 token pairs; committed; backs `make smoke` / `gate-golden`) |
| `Productions/LLMmap_ICML_run/` | 6.5 GB | 205-class LLMmap traces (backs LLMmap E3) |
| `Productions/FLiPS_ICML_run/` | **off-disk** | **Headline run** (E1 96% / E2 90% + FLIPS E3 arms). User HPC export; **absent on the build machine**, gates the data-gated tier; placeholder (64 zeros) in the manifest, filled once the HPC export + Zenodo upload exist. |

Fetch via `python scripts/fetch_data.py` (placeholder Zenodo DOI today — see
[`../data/README.md`](../data/README.md) for how the DOI is wired once minted).

**Verification — SHA-256, not MD5.** Our `data/manifest.sha256` is a **SHA-256** manifest. Zenodo's
own per-file checksums are **MD5**. `fetch_data.py` therefore verifies downloads against **our
SHA-256 manifest**, *not* Zenodo's MD5 checksums. Off-disk entries (hash = 64 zeros) are skipped as
external — they never PASS or FAIL.

> The old "~67 MB Answers / ~11 GB generations" size figures are **retired** — they were wrong. The
> real release is several GB spread across the heterogeneous members above.

---

## 7. LLMmap baseline

The LLMmap query-based fingerprinting baseline is **vendored** from
[`pasquini-dario/LLMmap`](https://github.com/pasquini-dario/LLMmap) under
`src/Fingerprinting_methods/LLMmap/`. Its SPDX header records the upstream license (**MIT**); verify it
against the source repo if you redistribute.

**The in-repo E3 curve needs no LLMmap install.** The E3 comparison replots from cached LLMmap result
data in `XP_configs/e3_flips_vs_llmmap/llmmap_if_data/` — `make repro-llmmap-e3` works out of the box on
CPU with no LLMmap install and no download.

**The 2 shipped adapter files are documented clone-only.**
`src/Fingerprinting_methods/LLMmap/cross_classif_task/make_dataset_from_Answers_format.py` and
`.../main_classif.py` are **import-broken on their own** (they depend on an old package layout, the
uncopied LLMmap core, and torch) and are **not imported by the out-of-box path**. Reproducing the
LLMmap cross-classification *from scratch* requires `git clone`-ing upstream `pasquini-dario/LLMmap` and
following its recipe:

```bash
git clone https://github.com/pasquini-dario/LLMmap.git
# then point its training scripts at the FLIPS Answers tables, per the upstream README.
```

The in-repo E3 curve uses the cached `llmmap_if_data/` — you only need the clone if you want to
**retrain** the LLMmap classifier rather than replot the cached result.

---

## See also

- [`reproduction.md`](reproduction.md) — step-by-step Mode A / Mode B.
- [`codebase/scenarios.md`](codebase/scenarios.md) — the scenario enumerator and label contract.
- [`../data/README.md`](../data/README.md) — archive layout, manifest regeneration, DOI wiring.
