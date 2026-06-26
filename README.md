# FLIPS: Instance-Fingerprinting for LLMs via Pseudo-random Sequences

[![Code license: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data license: CC-BY-4.0](https://img.shields.io/badge/data-CC--BY--4.0-lightgrey.svg)](LICENSES/CC-BY-4.0.txt)
[![Python ≥3.11](https://img.shields.io/badge/python-%E2%89%A53.11-3776ab.svg)](pyproject.toml)
[![arXiv](https://img.shields.io/badge/arXiv-2606.03330-b31b1b.svg)](https://arxiv.org/abs/2606.03330)

> **Reproduction scope**
> This repository supports three reproduction tiers:
> - **Out-of-box (no download):** LLMmap baseline E3 replot + a 12-model FLIPS *dev slice*.
>   The dev slice uses a reduced variation grid (temperatures 0.4 / 0.7 / 1.0; system-prompt
>   indices 0–3 / 6 / 7) — this is NOT the paper's 8-variation grid (temperatures
>   0.4 / 0.6 / 0.8 / 1.0 at sp=-1, ∪ 4 system prompts at temp 1.0 = 8 variations, NOT a 4×4 cross). Results are indicative, not paper-replicating.
> - **Data-gated (upload `FLiPS_ICML_run` from HPC):** full 237-instance headline (E1 96% / E2 90%) + FLIPS E3 arms.
> - **Mode B (GPU, from scratch):** re-run inference; reproduces methodology, not exact paper numbers (GPU non-determinism at temp > 0).

## Abstract

Literature reveals that a Large Language Model's (LLM) behavior is not only conditioned by its original weights but also its instance-level parameters, such as instructional prompt, sampling configuration or quantization. A model that generates safe outputs under one configuration may produce toxic content under another. However, current LLM identification techniques (such as fingerprinting) focus on intellectual property protection, and their design favors robustness to changes in these instance-level parameters. This poses a critical challenge for AI regulation in which compliance assessments target actual deployed behaviors, not model provenance. In this paper, we introduce instance-level fingerprinting, a regulator-oriented paradigm that distinguishes configurations of the same LLM. Our method FLIPS exploits biases in generated binary random sequences to reach **96% closed-set** accuracy (E1) and **90% open-set** accuracy (E2, where some targets are unknown) identification accuracy across 237 model instances, versus 35% for the adapted LLMmap baseline, and outperforms it in the head-to-head comparison (E3). This shows that instance-level fingerprinting is both necessary for regulation and practically feasible.

## A note on naming

The repository and all documentation are branded **FLIPS**, but the installable Python package is
`audit_llm` (its historical name) — so you `import audit_llm` even though everything else says FLIPS.

## Repository map

| Path | Role |
|------|------|
| `src/audit_llm/` | The installable package: inference backends (`LLM_Classes/`), feature computation, scenario enumerator, classification analysis. |
| `scripts/` | CLI entry points: `Run_Inferences.py` (Mode B), `fetch_data.py` (data download), `gates/` (out-of-box gates), `fig_scripts/preview_*.py` (figure regen). |
| `XP_configs/` | Declarative experiment configs + `run_experiments.py` dispatcher (E1/E2/E3 + smoke). |
| `datasets/` | Prompt-request tables (`Bits_Datasets/*.csv`) and prompt-assembly config (`prompt_config_index.yaml`, `system_prompts.json`). |
| `config/` | `models.yaml` model-universe registry and scenario YAMLs (the cluster-config SSOT). |
| `docs/` | Codebase guides (`docs/codebase/`) and reproduction guides (`docs/reproduction/`). |
| `data/` | Download target for fetched archives; verified against `data/manifest.sha256`. |
| `Productions/` | The committed light subset (`FLiPS_ICML_light_subset/`) + the intersection vocabulary; fetched/generated runs also land here. |

## Environment setup

Prerequisites: **Python ^3.11** and **[Poetry](https://python-poetry.org/)**.

```bash
git clone https://github.com/GurvanR/FLIPS-LLM-Instance-Fingerprinting.git
cd FLIPS-LLM-Instance-Fingerprinting
poetry install                 # analysis stack (CPU); add --extras generation for Mode B (GPU)
cp .env.example .env           # see .env.example for the AUDIT_LLM_* cache/path keys
```

The plain `poetry install` is CPU-only and is all you need for the out-of-box tier and every analysis
step. Mode B (GPU inference) additionally needs `--extras generation` plus a matching vLLM build —
see [`docs/reproduction/mode-b.md`](docs/reproduction/mode-b.md).

## Mode A — analysis quickstart (no GPU)

Mode A consumes existing generations and runs feature computation + classification on CPU. It comes
in three tiers; see [`docs/reproduction/mode-a.md`](docs/reproduction/mode-a.md) for the full
step-by-step.

### (a) Out-of-box — zero download

```bash
make repro-llmmap-e3      # LLMmap baseline E3 curve + dev-slice FLIPS, from the committed light subset
```

This produces an E3 comparison PDF. The **LLMmap arm** is the real cached baseline; the **FLIPS arm**
is computed from the 12-model *dev slice*, whose reduced variation grid (temps 0.4/0.7/1.0,
sp idx 0–3/6/7) differs from the paper's 8-variation grid. Treat this curve as *indicative of the
method*, **not** as a reproduction of the paper's headline numbers. (Out-of-box is `repro-llmmap-e3`,
not `repro-comparison` — the latter is the data-gated 3-curve target.)

> **Why the LLMmap baseline is downloaded, not retrained.** Unlike FLIPS (a lightweight **XGBoost**
> classifier), the LLMmap arm is a trained **neural-net** classifier (PyTorch Lightning checkpoints), and
> retraining it from scratch is roughly **~10–20× the FLIPS XGBoost analysis runtime**. So it ships
> pre-computed two ways: the small committed `llmmap_if_data/` cache (used by `repro-llmmap-e3` /
> `repro-comparison` — no download needed), and the full run `Productions/LLMmap_ICML_run/` (trained
> checkpoints + traces) hosted on Zenodo and fetched by `scripts/fetch_data.py` for anyone who wants to
> inspect or regenerate the baseline itself.

### (b) Data-gated — full 237-instance headline

Requires the HPC-exported `FLiPS_ICML_run` (the headline feature store + closed/open checkpoints,
which is off-disk in this repo). The data-gated `make` targets **SKIP loudly** (exit 0) when it is
absent — they never crash.

```bash
python scripts/fetch_data.py          # download released archives; verifies against data/manifest.sha256

make repro-closedset                  # E1 closed-set headline (96%)
make repro-openset                    # E2 open-set headline (90%)
make repro-comparison                 # E3 3-curve comparison (FLiPS closed/open vs LLMmap)
```

Underlying calls (the `make` targets wrap these):

```bash
python XP_configs/run_experiments.py --config-dir e1_closedset_headline --run-name FLiPS_ICML_run   # E1
python XP_configs/run_experiments.py --config-dir e2_openset_headline   --run-name FLiPS_ICML_run   # E2
python XP_configs/run_experiments.py --config-dir e3_flips_vs_llmmap                     --run-name FLiPS_ICML_run   # E3
```

> **Compute & disk budget.** Each data-gated experiment is a multi-hour, ~16–20 GB-RAM, no-GPU job that
> writes tens of GB under `Productions/<run>/Experiments/` (E1 alone: ~6 h, ~13 GB peak, 44 GB on disk;
> disk scales ~linearly — budget ~120–150 GB for all three).
> See [`docs/reproduction/compute-resources.md`](docs/reproduction/compute-resources.md) for measured
> per-experiment wall-clock, RAM, CPU and disk (hosts the E1/E2/E3 reports).

### (c) Re-render figures from a cached run

```bash
make figures     # redraw the closed-set micro-P/R and open-set ROC/PR figures from any run's caches
```

`make figures` scans `Productions/` for the `micro_pr_curve_cache.pkl` / `roc_figscore_cache.pkl` a real
run writes (`make smoke` or any `repro-*`) and re-plots without repeating the multi-GB analysis; with no
cache present it prints a hint and exits 0. Per-figure cache detail is in
[`docs/reproduction/mode-a.md`](docs/reproduction/mode-a.md).

## Mode B — inference from scratch (GPU required)

Mode B re-runs inference to regenerate the raw data, then feeds Mode A. It reproduces the
*methodology*, not the exact paper numbers (GPU sampling is non-deterministic at temp > 0). Needs a
GPU (≥24 GB VRAM), `poetry install --extras generation`, and a matching vLLM build.

```bash
python scripts/Run_Inferences.py --dataset FLiPS_ICML --model <hf-id> --gpu 1 --parse_gen --seed 42
```

Then run the Mode A tier-(b) analysis steps on the run you produced. For HPC/SLURM dispatch, see
[`docs/reproduction/mode-b.md`](docs/reproduction/mode-b.md) — do **not** put cluster-specific flags
into the quickstart above.

## Run your own scenario

The reproduction experiments are not a closed list — the same pipeline runs **your own scenario**
(your own set of model × variation instances), end to end. The model variations split across two
layers: **inference generates** the cells (`Run_Inferences.py` + a dataset CSV), and a small
**scenario YAML selects** which generated cells become classification classes. A runnable,
zero-download template ships in the repo:

```bash
make example-scenario     # selects a 4-class subset of the committed light subset (CPU, no download)
make figures
```

Copy [`config/scenarios/example_custom.yaml`](config/scenarios/example_custom.yaml) +
[`XP_configs/example_custom/`](XP_configs/example_custom/), edit the model list / variation grid,
and run. The full walkthrough — including generating *new* variations from inference — is in
[`docs/reproduction/custom-scenario.md`](docs/reproduction/custom-scenario.md).

## Experiment → command

| Exp | What | Command | Tier |
|-----|------|---------|------|
| **E1** | Closed-set headline (96%) | `make repro-closedset` → `run_experiments.py --config-dir e1_closedset_headline --run-name FLiPS_ICML_run` | data-gated |
| **E2** | Open-set headline (90%) | `make repro-openset` → `run_experiments.py --config-dir e2_openset_headline --run-name FLiPS_ICML_run` | data-gated |
| **E3** | FLiPS vs LLMmap comparison | `make repro-comparison` → `run_experiments.py --config-dir e3_flips_vs_llmmap --run-name FLiPS_ICML_run` (full 3-curve) | data-gated |
| **E3** | LLMmap baseline + dev-slice FLIPS replot | `make repro-llmmap-e3` | out-of-box |
| **E0** † | DCA / abliteration arm | part of the 237-instance headline (5 abliterated instances); no dedicated config dir — regenerate via Mode B, then the DCA showcase figure is emitted by the analysis pipeline. | data-gated † |

† **E0** has no standalone experiment config; the abliteration arm lives inside the headline run and
its showcase figure is produced by the analysis pipeline from that run.

## Figure / table → command

Figures are emitted by the analysis pipeline itself: running an experiment (**E1/E2/E3** or
[your own scenario](docs/reproduction/custom-scenario.md)) computes the real scores and writes the
corresponding figures under `Productions/<run>/…`.

## Non-determinism caveats

Reproduced numbers will not be bit-identical. Two independent sources of variance:

1. **Mode-B GPU sampling.** Two independent effects compound. First, generation runs at
   temperature > 0 (0.4–1.0) with an **unseeded** sampler, so each run draws different completions —
   that alone changes the outputs. Second, even at a fixed seed (or greedy decoding) GPU
   floating-point reductions are not bit-reproducible across runs, because kernel scheduling and batch
   composition vary. Re-running inference reproduces the *methodology* and accuracy band, not the
   exact generations or paper numbers.
2. **Analysis-side run-to-run variance.** Even on the *fixed released data* with `seed=42`, the
   headline `mix_tp` accuracy varies by **~±3.4 pp** because `token_pair_mixing.py` samples token-pair
   combinations with an **unseeded** `random.sample`.

## Citation

If you use FLIPS, please cite the paper. The preprint is on arXiv:
[arXiv:2606.03330](https://arxiv.org/abs/2606.03330). Metadata also lives in
[`CITATION.cff`](CITATION.cff).

```bibtex
@misc{richardeau2026flipsinstancefingerprintingllmspseudorandom,
      title={FLIPS: Instance-Fingerprinting for LLMs via Pseudo-random Sequences},
      author={Gurvan Richardeau and Gohar Dashyan and Erwan Le Merrer and Gilles Tredan},
      year={2026},
      eprint={2606.03330},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.03330},
}
```

## License

- **Code:** MIT — see [`LICENSE`](LICENSE).
- **Data:** CC-BY-4.0 — see [`LICENSES/CC-BY-4.0.txt`](LICENSES/CC-BY-4.0.txt). Released
  model *outputs* additionally inherit each model's own terms (e.g. Cohere Command-R outputs are
  CC-BY-NC, Gemma outputs follow the Gemma Terms of Use); see [`docs/`](docs/) provenance for the
  per-model constraints.
