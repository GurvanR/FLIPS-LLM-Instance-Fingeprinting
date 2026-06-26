# Reproduction

Step-by-step instructions to reproduce FLIPS results. This is the concise entry point; the exhaustive
per-tier command/figure tables live in [`reproduction/mode-a.md`](reproduction/mode-a.md) and
[`reproduction/mode-b.md`](reproduction/mode-b.md), and the model/data provenance in
[`provenance.md`](provenance.md).

Two modes:

- **Mode A** — no GPU. Re-runs the *analysis* (features → classification → figures) on
  already-generated traces. Three honestly-labelled tiers.
- **Mode B** — GPU. Re-runs *inference* to regenerate traces from scratch, then becomes Mode A.

---

## 1. Prerequisites

- **Python ^3.11**
- **Poetry** (dependency + virtualenv management)
- **git** — reproduce from a *cloned working tree*. Path resolution walks parent directories for a
  `.git` dir to locate the repo root; a non-editable `pip install` into `site-packages` has no `.git`
  and paths break. Use the clone (or `pip install -e .` inside it).
- **A GPU with ≥24 GB VRAM — Mode B only.** Not needed for any Mode A tier. The large models (70B+,
  Mixtral, Command-R+) need multi-GPU; see the sizing table in
  [`reproduction/mode-b.md`](reproduction/mode-b.md).

---

## 2. Clone and install

```bash
git clone https://github.com/GurvanR/FLIPS-LLM-Instance-Fingerprinting.git
cd FLIPS-LLM-Instance-Fingerprinting

poetry install                       # Mode A (CPU analysis stack) — default, no GPU deps
# poetry install --extras generation # Mode B only (torch, transformers, huggingface-hub, openai)
```

The default install pulls the CPU analysis stack (numpy, pandas, scikit-learn, xgboost, pyarrow) so
every Mode A tier, the smoke test, and the golden gate run CPU-only. The `generation` extra adds the
inference dependencies for Mode B. **vLLM is installed separately** (it is pinned to your CUDA/torch
build) — see [`reproduction/mode-b.md`](reproduction/mode-b.md).

Copy `.env.example` → `.env` if you need any `AUDIT_LLM_*` overrides or an `OPENROUTER_API_KEY`.

> Run the `make` targets inside the poetry env, or pass an interpreter that has `audit_llm` importable,
> e.g. `make smoke PYTHON=.venv/bin/python`. A bare system `python` makes the classification child
> process die with `ModuleNotFoundError: No module named 'audit_llm'`.

---

## 3. Mode A — step by step, by tier

### Tier 1 — out-of-box (zero download, CPU)

Runs against the committed light subset and the shipped LLMmap cache. No upload, no download, no GPU.

```bash
make repro-llmmap-e3   # LLMmap baseline E3 curve, replotted from the in-repo llmmap_if_data/ cache
make smoke             # dev-slice FLIPS smoke classification on the 12-model light subset, emits PDFs
make gate-golden       # golden regression: re-run smoke, assert label set + feature parity + accuracy band
```

The out-of-box target is **`make repro-llmmap-e3`** — *not* `repro-comparison`, which is the
data-gated 3-curve target. It regenerates the LLMmap accuracy-vs-queries curve plus the figure-styling
machinery, and the FLIPS arm is drawn from the **12-model dev slice**.

> **Label the output as dev-slice, not paper numbers.** The dev slice uses a *reduced* variation grid
> (temps `0.4/0.7/1.0`, system-prompt indices `0–3/6/7`) — **not** the paper's 8-variation grid
> (4 temps `0.4/0.6/0.8/1.0` at sp=-1, ∪ 4 system prompts at temp 1.0). Tier 1 is a wiring / regression
> check; its accuracies are **indicative, not paper-replicating**.

### Tier 2 — data-gated headline (needs the uploaded run)

The headline **96% (E1) / 90% (E2)** numbers and the FLIPS arms of E3 live in the off-disk
`FLiPS_ICML_run` (HPC export → Zenodo). Until you fetch it, the data-gated targets **SKIP loudly
(exit 0)** — they never crash.

```bash
python scripts/fetch_data.py        # download the archives; verifies each file against data/manifest.sha256 (SHA-256)
# … then run the headline experiments (commands from Experiment_Pipeline / the Makefile CLI table):
make repro-closedset                # E1 closed-set (headline 96%)
make repro-openset                  # E2 open-set (headline 90%)
make repro-comparison               # E3 3-curve: FLiPS closed/open vs LLMmap
make repro-ablations                # ablation sweep (ships with the headline; SKIPs if absent)
```

The underlying experiment dispatcher (equivalent invocations, from the `Experiment_Pipeline.md` CLI
table):

```bash
python XP_configs/run_experiments.py --config-dir e1_closedset_headline   # E1
python XP_configs/run_experiments.py --config-dir e2_openset_headline     # E2
python XP_configs/run_experiments.py --config-dir e3_flips_vs_llmmap                       # E3 (needs FLiPS_ICML_run)
```

Verification: `fetch_data.py` checks every downloaded file's **SHA-256** against `data/manifest.sha256`
(Zenodo's own checksums are MD5 — we do **not** use those). Off-disk entries (hash = 64 zeros)
are skipped as external. Run `python scripts/fetch_data.py --check-only` (or `make fetch-check`) to
verify the committed subset offline.

Each data-gated target checks for `Productions/$(RUN_NAME)/run_config.{json,pickle}`. If absent it
prints a `needs the user-uploaded FLiPS_ICML_run … SKIPPING` message and exits 0. Override with
`make repro-closedset RUN_NAME=my_run`.

**Tuning CPU/RAM (feature computation).** The first headline run recomputes NIST features from the
run's `Answers.parquet`, parallelized across token-pairs. Worker count defaults to `min(8, cpu-2)`
(clamped to `[1, 4*cpu]`); override with `FLIPS_FEATURE_WORKERS`. Each in-flight worker holds one
token-pair slice, so lowering the count trims the *slice* portion of peak RAM — on top of a fixed
baseline (the full frames stay resident regardless), so it reduces but does not eliminate the
footprint. Lower it on memory-tight machines, and on HPC set it to match your job's core allocation
rather than the node's total cores:

```bash
FLIPS_FEATURE_WORKERS=4 make repro-closedset    # 4 workers — gentler on RAM/CPU
```

Full tier model, experiment→command and figure→script tables, and the real-vs-synthetic-preview
distinction: [`reproduction/mode-a.md`](reproduction/mode-a.md).

---

## 4. Mode B — step by step (GPU)

Regenerate the traces themselves, then hand off to Mode A.

1. **Configure the model universe.** Models are declared in **`config/models.yaml`** (the
   source of truth, addressed by Hugging Face Hub id).
   Cluster users with a shared weights mirror can overlay paths via `config/models.hpc.example.yaml`.

2. **Run inference, one model at a time:**

   ```bash
   python scripts/Run_Inferences.py \
       --dataset FLiPS_ICML \
       --model Qwen/Qwen2-7B-Instruct \
       --sub_run qwen2_7b \
       --gpu 1 \
       --seed 42
   ```

   `--dataset` keys live in `scripts/Inference_configs.yaml` (`FLiPS_ICML` for the FLIPS prompt set,
   `LLMmap_ICML` for the LLMmap baseline). `--gpu N` is the tensor-parallel size. Always pass `--model`
   explicitly. Iterate over the 25 base (+ quantized + abliterated) models to build a full run.

3. **Parse + merge:**

   ```bash
   python scripts/Run_Inferences.py --dataset FLiPS_ICML --model <hf_id> --sub_run <tag> --parse_gen
   python scripts/parsing_generations.py --run_name FLiPS_ICML_run --merge_sub_run
   ```

   Then **flatten** so `run_config` + `Analysis/`/`Experiments/` sit directly under
   `Productions/FLiPS_ICML_run/` (the same one-run-path layout `fetch_data.py` produces) — the analysis
   entry point requires `run_config` directly under `run_path`.

4. **Hand off to Mode A** — once `Productions/FLiPS_ICML_run/run_config.{json,pickle}` is in place, the
   data-gated targets (`make repro-closedset` / `repro-openset` / `repro-comparison`) just work.

### SLURM pattern for HPC users

A Jean-Zay-style launcher ships at `scripts/run_large_LLMs.slurm`. The private allocation has been
scrubbed to a placeholder — **set `--account` to your own allocation** and adjust `--gres=gpu:N` / `-C` for your partition before submitting:

```bash
sbatch --time=01:00:00 --gres=gpu:1 --account=<your-account> \
    scripts/run_large_LLMs.slurm <module> \
    scripts/Run_Inferences.py --dataset FLiPS_ICML --model <hf_id> --sub_run <tag> --gpu 1
```

Full GPU prerequisites, the vLLM install note, per-model VRAM sizing, and the D1 log-prob caveat:
[`reproduction/mode-b.md`](reproduction/mode-b.md).

---

## 5. Seed and environment snapshot

- The analysis/classification seed is **42** by default (`--seed 42` for inference).
- Every run snapshots its environment to **`Productions/<run>/run_config.json`** at run start: the
  Python / vLLM / torch / CUDA versions and the GPU type, plus the models, paths, and grid. It is the
  leak-scrubbed authoritative record of exactly what produced a given set of figures (with
  `run_config.pickle` alongside). Consult it to see which grid and versions a result came from.

---

## 6. Smoke test

A zero-download CPU smoke runs on the committed light subset:

```bash
make smoke                          # or:
python -m pytest tests/smoke/       # in-repo light-subset tests; skip cleanly if the subset is absent
```

`make smoke` re-classifies the carved FLIPS-group token-pairs of the dev slice and emits `F01/F03/F04/
F06` PDFs under `Productions/FLiPS_ICML_light_subset/Experiments/`. The `tests/smoke/` tests use the
same committed subset and **skip** (not fail) when it is absent, so CI stays green on forks that never
carved it.

---

## 7. Non-determinism caveat

Two distinct sources of run-to-run variation — both expected, neither a bug:

**(a) Mode B GPU sampling.** Inference at `temperature > 0` is non-deterministic: regenerated traces
differ run-to-run even at a fixed `--seed 42`, and across vLLM / CUDA / driver versions. Mode B
reproduces the **methodology and the model universe**, not bit-identical traces.

**(b) Analysis-side run-to-run variance.** Even on the *fixed released data* with seed=42, the headline
`mix_tp` accuracy varies by **~±3.4 pp**, because `token_pair_mixing.py` samples token-pair
combinations with an **unseeded `random.sample`** (combined with `n_splits=2`). The decision was to
**preserve this behavior** (so the published numbers reflect exactly the code the paper ran) rather
than re-seed, and to **gate with a wide tolerance** instead: `make gate-golden` allows **±5 pp for
`mix_tp`** and **±2 pp for `tp_wise`** against `golden_refs.json`.

**Reported numbers are a representative single run.** Treat the headline 96% / 90% as the center of a
band, not a value you should expect to hit to the decimal.
