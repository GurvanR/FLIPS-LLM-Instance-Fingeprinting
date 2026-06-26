# Experiment Pipeline

---

## Overview

The experiment pipeline provides a reproducible, validated workflow for running analysis experiments
from YAML configuration files. It handles config discovery and validation (Pydantic), dispatches
execution across multiple modes, loads production run data, computes and caches per-model features,
and calls the appropriate experiment function (classification, visualization, or data export).

---

## File structure

| File | Description |
|------|-------------|
| `XP_configs/run_experiments.py` | CLI entry point (Click): discovers YAML configs, validates, dispatches via execution modes |
| `XP_configs/XP_script_global.py` | Per-experiment script: loads validated config, loads blacklisted models, calls `AuditionsAnalysis.run_xp()` |
| `src/audit_llm/experiment_config_schema.py` | Pydantic schema (`ExperimentConfig`) with sub-models; `load_experiment_config()` validates YAML |
| `src/audit_llm/Analysis_Classes.py` | `AuditionsAnalysis` facade: loads `run_config.json`, calls `_analysis_mode()` then `run_xp()` |
| `src/audit_llm/experiment_runner.py` | `run_xp()` orchestration: feature computation, config init, experiment dispatch |
| `src/audit_llm/data_loader.py` | `_analysis_mode()` (loads DataFrames), `_compute_save_load_experiments()` (feature computation & caching) |
| `src/audit_llm/xp_init_fun.py` | `_xp_config_init()`: prepares iterators, models, token pairs, model variation indices |
| `src/audit_llm/xp_tools/logging_utils.py` | Per-experiment file + console logging setup/teardown |

---

## End-to-end flow

```
run_experiments.py (CLI)
  |
  |-- 1. _find_yaml_configs()     : discover *.yaml in config-dir
  |-- 2. _validate_configs()      : Pydantic validation via load_experiment_config()
  |-- 3. _build_command()         : build XP_script_global.py commands
  |-- 4. Dispatch by mode:
  |       inline | slurm | single-screen | separate-screens | dry-run
  |
  v
XP_script_global.py (per experiment)
  |
  |-- argparsing(run_name, xp_suffix, xp_config_path)
  |-- load_experiment_config(path)           -> xp_config dict
  |-- Load Black_list_models_in_runs.json    -> xp_config['models_to_remove']
  |-- AuditionsAnalysis(run_path)
  |       |
  |       +-- Loads run_config.json (models, paths, scrapping_rule, token_pairs_set)
  |
  +-- AuditionsAnalysis.run_xp(xp_config)
        |
        |-- _analysis_mode()                 -> Answers_df, TokenIDs_df, MainDataset_df
        |
        +-- run_xp() [experiment_runner.py]
              |
              |-- [FLiPS path]
              |     |-- _compute_save_load_experiments()  -> cached features
              |     +-- _xp_config_init()                 -> iterators, models, token_pairs
              |
              |-- [LLMmap path]
              |     |-- _prepare_models_with_PRNGs()
              |     +-- _xp_config_init(feature_index=None)
              |
              |-- Build Experiment_config dict
              |-- setup_experiment_logging()
              |-- EXPERIMENT_FUNCTION_MAP[experiment_fun](Experiment_config)
              +-- teardown_experiment_logging()
```

---

## CLI options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--config-dir` | yes | — | Directory relative to `XP_configs/` containing YAML files |
| `--run-name` | yes | — | Production run name (maps to `Productions/<RUN_NAME>`) |
| `--xp-suffix` | no | `"default"` | Experiment suffix, used in output paths and `xp_name` |
| `--mode` | no | `"inline"` | Execution mode (see below) |
| `--recursive` | no | flag | Search YAMLs recursively |
| `--sleep` | no | `0.0` | Hours to sleep before starting |
| `--screen-name` | no | xp_suffix | GNU screen session name (single-screen mode) |
| `--slurm-time` | no | `"24:00:00"` | SLURM `--time` value |
| `--slurm-gpus` | no | `1` | SLURM GPUs per task |
| `--slurm-cpus` | no | `4` | SLURM CPUs per task |

---

## Execution modes

| Mode | Behavior |
|------|----------|
| `inline` | Sequential subprocess in current shell; logs stdout+stderr per experiment |
| `slurm` | One `sbatch` job per config |
| `single-screen` | All configs chained with `&&` in one GNU screen session |
| `separate-screens` | One GNU screen session per config |
| `dry-run` | Validate all configs and print commands without executing |

All modes log combined stdout+stderr to:
```
xp_logs/<run_name>/<xp_suffix>/xp_{i}_log_{timestamp}.log
```

> **Note:** `run_experiments.py` returns exit code 0 even when an inner XP fails — it only prints
> `Command failed with exit code ...`. Always check the log files for `Traceback` /
> `FileNotFoundError` when verifying a run.

---

## Config discovery

`_find_yaml_configs()` collects `*.yaml` files from the config directory, excluding
`XP_config_libs/` and `Old_configs/` subdirectories.

---

## Config example

```yaml
experiment_fun: Batch_Classification_across_token_pairs
alpha: 0.05
features: SmallNist
min_seq_length: 100
PRNGs: None
models: []                    # empty = all available
token_pairs: []               # empty = all in token_pairs_set

sampling_parameters:
  system_prompt_idx: []       # empty = all
  temperature: []
  frequency_penalty: [0.0]

model_variations:
  - {temperature: [0.4, 0.6, 0.8, 1.0], system_prompt_idx: [-1]}
  - {temperature: [1.0], system_prompt_idx: [0, 3, 6, 7]}

calculations:
  token_pairs: token_pairs

classification_config:
  classifiers: [XGBoost]
  splitter_type: StratifiedShuffleSplit
  n_splits: 10
  test_size: 64
  batch_prediction_sizes: [1,2,3,4,5,6,7,8]
  batch_types: [tp_wise]
```

---

## Data loading

`_analysis_mode()` loads three DataFrames:

| DataFrame | Source | Content |
|-----------|--------|---------|
| `Answers_df` | Per-model partitioned Parquet | All collected LLM answers: `Model`, `Token_pair`, `Answer`, `Dataset_Question Index`, sampling params |
| `TokenIDs_df` | Parquet (currently disabled for size) | Token ID sequences per answer |
| `MainDataset_df` | Source CSV (`Dataset_path`) | The prompt dataset with sampling parameter columns and `Index` |

---

## Feature computation branch

After data loading, `run_xp()` branches on experiment type:

- **FLiPS path**: calls `_compute_save_load_experiments()` — feature computation and caching.
  See [features.md](features.md) for details.
- **LLMmap path**: detected via `"LLMmap" in Dataset_path`. Calls `_prepare_models_with_PRNGs()`
  directly (no NIST feature computation). Uses embedding-based classification.

---

## Experiment init (`_xp_config_init`)

| Step | Function | Purpose |
|------|----------|---------|
| 1 | `integrate_nist_test_parameters()` | Merge NIST test config based on `features` key |
| 2 | `prepare_token_pairs()` | Filter token pairs: config subset + remove banned pairs |
| 3 | `prepare_models()` | Filter models: config subset, blacklist, closed-source removal |
| 4 | `build_calculation_iterators()` | Build `{iterator_idx: value_list}` from `calculations` |
| 5 | `extract_global_samples_indices()` | Filter `MainDataset_df` by sampling parameters |
| 6 | `compute_model_variations_indices()` | Compute model grouping indices from `model_variations` |

---

## Experiment function map

| Key | Purpose |
|-----|---------|
| `Nist_perf_chart` | NIST statistical test performance visualization |
| `Seq_Length_visualization` | Sequence length CCDF plots |
| `Valid_count_chart` | Valid answer count bar charts |
| `Save_pv_in_parquet` | Export p-values to Parquet format |
| `classify` | Single token-pair classification |
| `classify_cross_token_pairs` | Cross-token-pair NxN evaluation |
| `batch_classification_across_token_pairs` | Multi-token-pair batch classification |
| `LLMmap_classification` | LLMmap embedding-based classification |
| `feature_space_visualization` | UMAP/t-SNE scatter plots of the feature space |

Legacy alias: `Batch_Classification_across_token_pairs` → `batch_classification_across_token_pairs`.

---

## Output structure

```
Productions/<RUN_NAME>/
  +-- run_config.json
  +-- Black_list_models_in_runs.json
  +-- <model_name>/                      (raw inference outputs)
  +-- Analysis/<scrapping_rule>/
  |     +-- answers/*.parquet
  |     +-- token_ids/*.parquet
  +-- Experiments/
        +-- model_index.json
        +-- feature_computation_data/
        |     +-- <token_pair>/
        |           +-- intra/<model>.npy
        |           +-- inter/<model>.npz
        |           +-- manifest.json
        +-- <experiment_fun>/
              +-- <xp_name>/
                    +-- Experiment_config.txt
                    +-- summary.log
                    +-- detailed.log
                    +-- (figures, checkpoints, results...)
```

---

## Logging

Two log levels per experiment run:

| File | Level | Content |
|------|-------|---------|
| `summary.log` | INFO | High-level milestones |
| `detailed.log` | DEBUG | Full diagnostics, shapes, per-split info |
| Console | INFO | For screen session monitoring |

---

## Related documentation

| File | Content |
|------|---------|
| [features.md](features.md) | Feature computation: NIST tests, caching format, PRNG baselines |
| [classification.md](classification.md) | Classification module internals |
| [configuration.md](configuration.md) | Complete Pydantic schema reference for `ExperimentConfig` |
| [dataset.md](dataset.md) | Dataset format, CSV schema, prompt assembly |
