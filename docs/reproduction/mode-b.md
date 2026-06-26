# Mode B — reproduction from scratch (GPU)

Mode B is the **from-scratch GPU path**: it re-runs *inference* to regenerate the model traces,
then hands the parsed runs to [Mode A](mode-a.md) for feature extraction, classification, and
plotting. Use it when you want to reproduce the data itself (or anything that is **not on disk in
any archive tier**, e.g. the D1 log-prob diagnostics), rather than re-analysing released traces.

> **Mode A vs Mode B.** Mode A ([`mode-a.md`](mode-a.md)) consumes already-generated traces on
> CPU. Mode B regenerates those traces on a GPU box and then becomes Mode A. If you only need the
> paper numbers from the released run, you want Mode A Tier 2, **not** Mode B.

**Fidelity caveat.** Inference at `temperature > 0` is **non-deterministic** — regenerated traces
differ run-to-run even at a fixed seed (`--seed 42`), and across vLLM / CUDA / driver versions.
Mode B reproduces the **methodology and the model universe**, not bit-identical traces. The
classification seed downstream is still 42, and every run snapshots its environment to
`Productions/<run>/run_config.json`.

---

## Prerequisites

- A CUDA GPU box. Single-GPU is enough for the ≤9B models; the large models need multi-GPU
  (see the VRAM table below).
- A working **vLLM** install (see *Install* — vLLM is **not** pulled by the default extras).
- The Hugging Face model weights, downloaded on demand through the standard HF cache
  (`HF_HOME`). Weights are **not** redistributed by this repo.

> **Run from a git clone (not a packaged install).** Path resolution
> (`audit_llm.path_utils.get_repository_level_path`) walks parent directories looking for a
> `.git` directory to locate the repo root (and therefore `Productions/`, `scripts/`,
> `config/`). A non-editable `pip install` into `site-packages` has **no `.git`**, so the walk
> falls through to the filesystem root and paths break. Reproduce from a cloned working tree (or
> an editable `pip install -e .` inside the clone). The same constraint applies to Mode A.

---

## Install

```bash
poetry install --extras generation        # torch, transformers, huggingface-hub, openai
```

The `generation` extra (`pyproject.toml` → `[tool.poetry.extras]`) installs `torch`,
`transformers`, `huggingface-hub`, and `openai`. It deliberately does **not** install **vLLM**:
vLLM is tightly coupled to the CUDA / torch build, so it is pinned and installed **separately**,
on its own, to match your GPU stack — for example:

```bash
# Pick the vLLM build that matches your CUDA + torch; pin it explicitly.
pip install "vllm==<the-version-matching-your-torch/cuda>"
```

> **Honest note.** `audit_llm.system_utils._import_vllm_components` raises an ImportError that
> says *"Install the generation extras: poetry install --extras generation"* when vLLM is missing.
> That message is slightly misleading: the `generation` extra does **not** contain vLLM. vLLM is a
> separate install, by design — install it yourself as above.

The code supports several vLLM versions and adapts at import time
(`audit_llm.system_utils.vllm_version_import_manager` handles `<=0.4.0`, `0.4–0.7`, and `0.7+`).

### Environment

| Variable | Purpose | Notes |
|---|---|---|
| `HF_HOME` | Hugging Face cache root (weights land in `$HF_HOME/hub`) | standard HF variable; no cluster-specific paths required |
| `AUDIT_LLM_MODEL_CACHE_DIR` | optional low-level override of the hub cache dir | see `.env.example`; prefer `HF_HOME` |
| `OPENROUTER_API_KEY` | required only for `--openrouter` (OpenRouter-hosted models) | read by `system_utils.get_openrouter_key`; copy `.env.example` → `.env` |

Cluster users who keep weights on a shared read-only mirror can overlay paths via
`config/models.hpc.example.yaml` (`model_cache_root` / per-model `overrides`) — see the
[Appendix](#appendix--hpc--cluster-cluster-use).

### Hardware / per-model GPU sizing

`--gpu N` sets vLLM's **tensor-parallel size** (`Run_Inferences.py` →
`tensor_parallel_size`). `gpu_memory_utilization` is fixed at `0.9`; `max_model_len` comes from
`scripts/Inference_configs.yaml` (`FLiPS_ICML.max_model_len: 2048`).

| Model class (examples) | Typical sizing |
|---|---|
| ≤9B (`Qwen2-7B`, `gemma-2-9b-it`, `Mistral-7B-*`, `Phi-3-mini/medium`) | single GPU, `--gpu 1` |
| 27–35B (`gemma-2-27b-it`, `aya-23-35B`, `command-r-v01`) | 1–2 GPUs depending on VRAM |
| 70B+ and Mixtral (`Meta-Llama-3.1-70B`, `Smaug-Llama-3-70B`, `command-r-plus`, `Nous-Hermes-2-Mixtral-8x7B`) | **multi-GPU** — set `--gpu` to the tensor-parallel size your node supports |

> Sizing is hardware-dependent; treat the table as a starting point and raise `--gpu` if a model
> OOMs at load.

---

## The pipeline (three stages)

### Stage 1 — inference (vLLM)

Run one model at a time against the FLiPS prompt set:

```bash
python scripts/Run_Inferences.py \
    --dataset FLiPS_ICML \
    --model Qwen/Qwen2-7B-Instruct \
    --sub_run qwen2_7b \
    --gpu 1 \
    --seed 42
```

- `--dataset` keys live in `scripts/Inference_configs.yaml` (`FLiPS_ICML` for the FLiPS prompt
  set with `ICML_SET_OF_TOKEN_PAIRS`; `LLMmap_ICML` for the LLMmap baseline). `--model` is a
  Hugging Face Hub id from the registry; `--gpu` is the tensor-parallel size; `--sub_run` names
  this model's slice of the run; `--openrouter` routes through OpenRouter instead of local vLLM
  (needs `OPENROUTER_API_KEY`).
- **Output path.** Each invocation writes under
  `Productions/Graph_Productions/Normal_Runs/FLiPS_ICML_run/<sub_run>/` (the run name is
  `<dataset>_run/<sub_run>`; `--test` redirects to `Test_Runs/`, `Toy_example` to `Toy_Runs/`).
  *(The shorthand `Productions/<run>/<model>/*.parquet` is a simplification of this
  layout.)* The script snapshots `Inference_configs.yaml` and a `run_config` into the run dir for
  reproducibility.

**The model universe.** The 25 base LLMs (plus quantized and abliterated variants) are declared
in **`config/models.yaml`** — the model-universe source of truth, addressed by Hugging Face Hub
id. Iterate `--model` over them to build a full run.

> **Heads-up.** When `--model` is omitted, `Run_Inferences.py` still falls back to the legacy
> `OR_4` model group (there is a `# TODO: load model groups from config/models.yaml` at the top of
> the script). Always pass `--model` explicitly until that TODO lands.

**What defines the cross-product (and the documented asymmetry).** The analysis layer's instance
universe — *which* model × {temperature, system_prompt, quantization, abliteration} cells become
classification classes — is defined by the **scenario enumerator**
[`build_instances`](../../src/audit_llm/scenarios/enumerator.py), driven by the scenario YAMLs in
[`config/scenarios/`](../../config/scenarios/):

| Scenario | Instances | Composition |
|---|---|---|
| `main` (paper default) | **237** | 200 base (25 models × 8 variations) + 32 quantized (4 models × 2 levels × 4 variations) + **5 data-backed abliterated** (pinned `temp=1.0`/`sp=-1`) |
| `cross500` | 500 | wider crossing of the same real grid (375 base + 120 quantized + 5 abliterated) |
| `cross1000` | 995 | scaling stress only — extends the temperature axis with illustrative `0.2`/`1.2` cells with **no released data**; *not* a reproduction target |

> **Documented asymmetry (important).** Inference keeps its **own** vLLM expansion inside
> `Run_Inferences.py` — it decides how each model is sampled (temperatures, system prompts,
> quantization engine kwargs) at generation time. The **enumerator** is a separate,
> **analysis-layer** mechanism: it drives *class materialization* (the label set the classifier
> sees) and **supersedes** both the legacy `compute_model_variations_indices`
> (`src/audit_llm/xp_tools/variation_context.py`) and the legacy hardcoded abliteration branch
> (`abliterated_samples_indices = main_dataset.filter(...)`, near
> `src/audit_llm/xp_tools/data_preparation.py:193`). The two
> paths are deliberately **not** unified: inference owns generation; the enumerator owns analysis.

> **Running your own variation grid?** Once you have generated the cells you want here, select them
> for analysis with a scenario — the end-to-end walkthrough (both layers) is
> [`custom-scenario.md`](custom-scenario.md).

### Stage 2 — parse + merge

Inference writes raw generations; parsing turns them into the per-model Parquet tables the
analysis layer reads.

```bash
# (a) parse inline right after a run …
python scripts/Run_Inferences.py --dataset FLiPS_ICML --model <hf_id> --sub_run <tag> --parse_gen

# … or (b) consolidate all sub_runs of a run into merged_sub_runs/
python scripts/parsing_generations.py --run_name FLiPS_ICML_run --merge_sub_run

# (c) optionally fold a separately-parsed run into another
python scripts/merge_runs.py --dest_run FLiPS_ICML_run --source_run FLiPS_ICML_run_batch2
```

- `--parse_gen` parses the run that just finished; `parsing_generations.py --merge_sub_run`
  consolidates a run's sub_runs into `Productions/<run>/merged_sub_runs/`; `merge_runs.py` folds a
  *separate* parsed run in (it auto-detects and writes into `merged_sub_runs/` when present, and
  refuses to overwrite existing per-model parquets).

> **Run-path convention — do this before Mode A.** The analysis entry point
> `AuditionsAnalysis(run_path)` requires `run_config` to sit **directly under** `run_path`. After
> a `--merge_sub_run`, the data lives one level down in `Productions/<run>/merged_sub_runs/`. So
> either:
> - point Mode A at it explicitly: `make repro-closedset RUN_NAME=FLiPS_ICML_run/merged_sub_runs`, or
> - **flatten** so `run_config` + `Analysis/`/`Experiments/` sit directly under
>   `Productions/FLiPS_ICML_run/`.
>
> Flattening is the **same one-run-path layout** that `scripts/fetch_data.py` produces for the
> downloaded headline run and the committed light subset — picking it keeps the default
> `RUN_NAME=FLiPS_ICML_run` working across all of Mode A.

### Stage 3 — hand off to Mode A

Once `Productions/FLiPS_ICML_run/run_config.{json,pickle}` is in place (flattened, per above),
the data-gated Mode A targets just work:

```bash
make repro-closedset    # E1 closed-set (headline 96%)
make repro-openset      # E2 open-set (headline 90%)
make repro-comparison   # 3-curve E3 (FLiPS arms vs LLMmap)
make figures            # replot from the produced caches
```

Everything else — the tier model, the experiment→command and figure→script tables, the
real-vs-synthetic distinction, where the in-experiment PDFs land — is documented in
[`mode-a.md`](mode-a.md).

---

## D1 — log-prob diagnostics are Mode-B-only

The D1 log-prob diagnostics are **not on disk in any archive tier**: the released runs
(`FLiPS_ICML`, `LLMmap_ICML`) generate text only — `logprobs` is left unset in their
`scripts/Inference_configs.yaml` entries (only a few non-released experiment keys, e.g.
`TempsLogprobs2`/`SPImpact_toy`, request `logprobs`). To reproduce anything D1, you must
**regenerate from scratch** with a logprob-capturing inference config; there is no shortcut
through Mode A.

---

## Smoke target

A guarded one-model end-to-end smoke is wired into the Makefile:

```bash
make repro-modeb-smoke        # alias: make infer-smoke
```

It first checks that both `torch` and `vllm` import. If either is missing (e.g. on a CPU-only
build box) it prints a clear **"needs GPU + vLLM"** message and **exits 0** — so it is safe to run
anywhere and never breaks CI. When the GPU stack is present it runs inference for one small model
end-to-end:

```bash
make repro-modeb-smoke MODEB_SMOKE_MODEL=Qwen/Qwen2-1.5B-Instruct MODEB_SMOKE_DATASET=FLiPS_ICML
```

Override `MODEB_SMOKE_MODEL` / `MODEB_SMOKE_DATASET` / `RUN_NAME` to point it elsewhere.

---

## Appendix — HPC / cluster use

Both are **optional** — document, not require. Nothing here is read unless you opt in.

- **Model-cache overlay:** [`config/models.hpc.example.yaml`](../../config/models.hpc.example.yaml)
  (under `config/`). The public `config/models.yaml` resolves
  every model through the standard HF cache; copy the example to a gitignored
  `config/models.hpc.local.yaml` and set `model_cache_root` (and optional per-model `overrides`)
  to point at a shared read-only mirror.
- **SLURM launcher:** [`scripts/run_large_LLMs.slurm`](../../scripts/run_large_LLMs.slurm) — a
  Jean-Zay-style example. The private allocation has been scrubbed to a placeholder
  (`--account=YOUR_HPC_ACCOUNT`); set `--account` to your own allocation and adjust
  `--gres=gpu:N` / `-C` to your partition before submitting, e.g.:

  ```bash
  sbatch --time=01:00:00 --gres=gpu:1 scripts/run_large_LLMs.slurm <module> \
      scripts/Run_Inferences.py --dataset FLiPS_ICML --model <hf_id> --sub_run <tag> --gpu 1
  ```
