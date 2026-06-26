# Codebase overview

## Architecture summary

FLIPS runs in two phases. The **inference phase** (`scripts/Run_Inferences.py`) drives vLLM to
generate bit sequences from a prompt CSV dataset and writes one Parquet checkpoint per model per
sampling key. A parse/merge step (`scripts/parsing_generations.py`) assembles those checkpoints
into per-model `Answers.parquet` files. The **analysis phase** starts from those Parquet files:
`_compute_save_load_experiments()` computes NIST-based intra-sample and inter-sample features and
caches them as `.npy`/`.npz` files; the classification layer
(`batch_classification_across_token_pairs` / `classify` / `classify_cross_token_pairs`) trains
XGBoost classifiers across token pairs; and figure scripts under `scripts/fig_scripts/` render the
paper figures. The experiment pipeline (`XP_configs/run_experiments.py`) orchestrates the analysis
phase end-to-end from declarative YAML configs.

## Repository map

| Directory | Role |
|-----------|------|
| `src/audit_llm/` | Installable Python package (inference backends, analysis pipeline, classification, scenarios) |
| `scripts/` | CLI entry points: inference, parsing, figure generation, data fetching, gates |
| `XP_configs/` | Experiment YAML configs + per-experiment entry point (`XP_script_global.py`) |
| `datasets/` | Prompt-request CSVs (`Bits_Datasets/*.csv`), prompt config index, system-prompt JSONs |
| `config/` | Model universe registry (`models.yaml`), scenario YAMLs (`scenarios/`) |
| `docs/` | User-facing documentation portal |
| `data/` | Downloaded Zenodo archives land here (fetched by `scripts/fetch_data.py`) |
| `Productions/` | All run outputs: raw Parquet checkpoints, parsed answers, feature caches, figures |

## Sub-guides

- **Inference pipeline**: [`inference.md`](inference.md)
- **Experiment pipeline**: [`experiment.md`](experiment.md)
- **Feature computation**: [`features.md`](features.md)
- **Classification**: [`classification.md`](classification.md)
- **Dataset format**: [`dataset.md`](dataset.md)
- **Quantized models**: [`quantized-models.md`](quantized-models.md)
- **Configuration**: [`configuration.md`](configuration.md) (simple) · [`configuration-advanced.md`](configuration-advanced.md) (full reference)
- **Conventions**: [`conventions.md`](conventions.md)
- **Testing strategy**: [`testing.md`](testing.md)
- **Shipped scenarios**: [`scenarios.md`](scenarios.md)
- **Reproduction (no-GPU, 3 tiers)**: [`../reproduction/mode-a.md`](../reproduction/mode-a.md)
- **Reproduction (GPU, from scratch)**: [`../reproduction/mode-b.md`](../reproduction/mode-b.md)
- **Analysis paths (legacy + scenario)**: [`../reproduction/legacy-paths.md`](../reproduction/legacy-paths.md)

---

---

## Package layout (`src/audit_llm/`)

The package has three layers:

```
config/scenarios/*.yaml          ← declarative scenario files (model × variation selection)
config/models.yaml               ← model universe registry (HF ids, quant levels, abliterated repos)
        │
        ▼  (analysis layer only)
audit_llm.scenarios              ← scenario package
  ├── schema.py / loader.py      ← Scenario dataclass + YAML loader (validates against registry)
  ├── variation.py               ← Variation + Instance (resolved model × variation cell)
  ├── resolver.py                ← resolve_quantization → vLLM kwarg; resolve_abliteration → HF repo
  └── enumerator.py              ← build_instances(scenario) → list[Instance]  [analysis SSOT]
        │
        ▼
audit_llm.experiment_runner      ← analysis entry point
  └── _xp_config_init()          ← prefers scenario: key; falls back to legacy keys
        │
        ▼
audit_llm.xp_tools.data_preparation
  └── prepare_dataset_features() ← materialises classification classes from Instance list
```

The inference path (`scripts/Run_Inferences.py`) keeps its **own vLLM expansion** and is **not** connected to the scenario enumerator. This is a deliberate asymmetry: `build_instances` is the analysis-layer source of truth for which instances exist and their labels; the inference path has its own logic for launching vLLM jobs. Do not describe them as a shared or unified source of truth.

---

## Scenario layer

A `Scenario` (loaded from `config/scenarios/*.yaml`) selects which model × variation instances the analysis layer materialises as classification classes. It varies over **four axes**:

| Axis | Nature |
|---|---|
| `temperature` | sampling-request field |
| `system_prompt_idx` | sampling-request field |
| `quantization` | model transform → `{base}@@{quant_key}` via a vLLM engine kwarg |
| `abliteration` | model transform → `{abliterated_repo}_ablit` using a *different weights repo* on HF Hub |

`build_instances(scenario)` (in `audit_llm.scenarios.enumerator`) enumerates the full union of variation cells:
- base models × base variation grid
- quantized base models × quant variation grid × quant levels (`fp8`, `bitsandbytes_int4`)
- abliterated models × abliteration group (pinned to `temp=1.0, sp=-1`)

The resulting `Instance` list is the analysis-layer source of truth for the classification class set. Labels are byte-stable strings parsed by `xp_tools.model_filtering`; see the label stability note in [`conventions.md`](conventions.md).

For the three shipped scenarios (`main`, `cross500`, `cross1000`) and the abliteration rules, see [`scenarios.md`](scenarios.md).

For the xp_config `scenario:` field and legacy fallback keys, see [`configuration-advanced.md`](configuration-advanced.md).
