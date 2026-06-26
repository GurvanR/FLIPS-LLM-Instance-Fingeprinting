# Configuration Reference (Advanced)

Complete schema reference for experiment YAML configuration files, validated by Pydantic in
`src/audit_llm/experiment_config_schema.py`.

> New here? Start with the simple **[configuration.md](configuration.md)** — it covers the few
> keys most experiments actually set. This page documents *every* field, including the ones that
> are defaulted automatically.

---

## Loading & validation

```python
from audit_llm.experiment_config_schema import load_experiment_config

config = load_experiment_config("path/to/config.yaml")  # returns dict
```

- Custom `_UniqueKeyLoader` raises `ValueError` on duplicate YAML keys
- Pydantic `ExperimentConfig.model_validate()` validates structure and types
- All models use `extra="allow"` for forward compatibility
- Returns a plain `dict` via `model_dump()` for backward compatibility

---

## Auto-applied defaults

These `classification_config` fields are filled in automatically — omit them unless you want a
non-default value:

| Field | Default applied | Notes |
|-------|-----------------|-------|
| `classifiers` | `["XGBoost"]` | Add more names to train several classifiers |
| `splitter_type` | `"StratifiedShuffleSplit"` | CV splitter from `SPLITTER_MAP` |
| `default_normalization` | `"auto"` | Per-feature method auto-selected unless overridden |
| `normalization_methods` | `{seq_length: none}` | Per-feature overrides; `seq_length` is left un-normalized |
| `batch_types` | **derived from `batch_prediction_sizes`** | size `1` → `tp_wise`; any size `> 1` → `mix_tp_at_pred`. Set explicitly to override. |

---

## `ExperimentConfig` — top-level fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `experiment_fun` | `str` | **required** | Experiment function name; validated against `VALID_EXPERIMENT_FUNCTIONS` |
| `alpha` | `float` | `0.05` | Statistical significance threshold |
| `features` | `str\|None` | `None` | Feature set name (e.g. `SmallNist`, `SmallNist_only_pv`) |
| `min_seq_length` | `int` | `100` | Minimum bit sequence length filter |
| `PRNGs` | `list\|str\|None` | `None` | PRNG models: `None` = skip, `[]` = all, `[names]` = specific |
| `models` | `list[str]` | `[]` | Model filter: `[]` = all available models |
| `token_pairs` | `list[str]` | `[]` | Token pair filter: `[]` = all in `token_pairs_set` |
| `models_to_remove` | `list[str]\|None` | `None` | Blacklisted models (set at runtime from `Black_list_models_in_runs.json`) |
| `sampling_parameters` | `SamplingParameters` | — | Sampling parameter filters (see below) |
| `scenario` | `str\|None` | `None` | Declarative model × variation selection (`config/scenarios/*.yaml`); the forward path (see [variation selection](#variation-selection-scenario-vs-model_variations)) |
| `model_variations` | `str\|dict\|list[dict]\|None` | `None` | Direct model × variation grid; used (with `quantized_model_variations` / `abliterated_models`) when `scenario` is unset — the path the shipped configs currently take |
| `abliterated_models` | `list[str]\|None` | `None` | `None` = skip, `[]` = all, `[names]` = specific |
| `include_quantized` | `list[str]\|None` | `None` | `None` = exclude all `@@`-suffixed models; `[]` = include all; `["fp8"]` = specific methods |
| `calculations` | `dict[str,str]\|None` | `None` | Iterator definitions |
| `figures` | `dict[str,FigureConfig]\|None` | `None` | Named figure configurations |
| `train_sizes` | `list[int]\|None` | `None` | Training sizes for ablation studies |
| `classification_config` | `ClassificationConfig\|None` | `None` | Classification pipeline settings |
| `set_constant_seq_length` | `int\|None` | `None` | Truncate all bit sequences to exactly this length before NIST computation |
| `dr_method` | `str\|None` | `"umap"` | Dimensionality reduction for `feature_space_visualization`: `"umap"` or `"tsne"` |
| `xp_name` | `str\|None` | `None` | Auto-set at runtime: `{config_name}_{xp_suffix}` |

### Valid experiment functions

```
Nist_perf_chart
Seq_Length_visualization
Valid_count_chart
classify
classify_cross_token_pairs
batch_classification_across_token_pairs
Batch_Classification_across_token_pairs    (legacy alias)
LLMmap_classification
Save_pv_in_parquet
feature_space_visualization
```

---

## `SamplingParameters`

Controls which sampling parameter values to include. Empty list = all available values.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `system_prompt_idx` | `list[int]` | `[]` | System prompt indices to include |
| `temperature` | `list[float]` | `[]` | Temperature values to include |
| `frequency_penalty` | `list[float]` | `[0.0]` | Frequency penalty values |

`extra="allow"`: any additional sampling parameter key is accepted.

```yaml
sampling_parameters:
  system_prompt_idx: [-1, 0, 3]
  temperature: [0.4, 0.6, 0.8, 1.0]
  frequency_penalty: [0.0]
```

---

## `ClassificationConfig`

Fields marked **(auto)** are filled in automatically (see
[Auto-applied defaults](#auto-applied-defaults)); omit them unless overriding.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `classifiers` | `list[str]` | `["XGBoost"]` **(auto)** | Classifier names (e.g. `XGBoost`, `Random Forest`, `Logistic Regression`) |
| `splitter_type` | `str` | `"StratifiedShuffleSplit"` **(auto)** | CV splitter |
| `n_splits` | `int` | `2` | Number of CV splits |
| `test_size` | `float\|int` | `0.5` | Test set size (float=fraction, int=absolute per-class count) |
| `force_class_size` | `int\|"auto"\|None` | `"auto"` | Per-class cap. `"auto"`=balance to the smallest class (undersample only, no oversampling/leakage); `int`=cap each class to ≤N (must be `> test_size`, checked at load); `None`=no balancing |
| `classifier_metrics` | `list[str]` | `["accuracy"]` | Metrics to compute |
| `batch_prediction_sizes` | `list[int]\|None` | `None` | Batch sizes for batch prediction |
| `batch_types` | `list[str]\|None` | derived **(auto)** | Batch modes: `tp_wise`, `mix_tp_at_pred`, `mix_tp_at_train`. When unset, derived from `batch_prediction_sizes` (size `1` → `tp_wise`, size `> 1` → `mix_tp_at_pred`) |
| `openset` | `bool` | `false` | Enable open-set classification |
| `m_test_size` | `float\|int` | `5` | Number of unknown models for open-set |
| `openset_m_splits` | `int` | `10` | Number of open-set splits |
| `default_normalization` | `str` | `"auto"` **(auto)** | Default normalization: `auto`, `none`, `standard`, etc. |
| `normalization_methods` | `dict[str,str]` | `{seq_length: none}` **(auto)** | Per-feature normalization overrides |
| `unique_tp_in_mix` | `int\|list[int]\|str` | `['max']` **(auto)** | Distinct token pairs per mixed batch. `'max'` (default) = `bs` distinct pairs (needs `C(n_tp, bs) >= max_nb_of_uplet` for each plotted `bs`); an `int`/`[int]` caps it — pin a small value when few token pairs ship |
| `max_nb_of_uplet` | `int\|None` | `None` | Maximum number of token-pair tuples in mixing |
| `compute_confusion_matrices` | `bool` | `false` | Compute and plot confusion matrix heatmaps (tables always computed) |
| `store_prediction_probas` | `bool` | `true` **(auto)** | Keep per-sample predicted probabilities (needed for PR curves / probability tables) |
| `compute_micro_pr_curve` | `bool` | `true` **(auto)** | Compute the micro-averaged precision–recall curve |
| `is_closed` | `bool\|None` | `None` | LLMmap cross-classification: force closed-set (vs open-set) dataset construction |
| `train_size_dict_2_checkpoint_path` | `str\|None` | `None` | DCA "compare mode": second checkpoint to overlay against the main curve |
| `train_size_dict_map` | `dict[str,str]\|None` | `None` | "Merged-wrapper mode": load named per-source checkpoints for F01 comparison figure |

```yaml
classification_config:
  n_splits: 10
  test_size: 64
  # force_class_size omitted -> "auto" (balance to the smallest class). Set an explicit
  # int (must be > test_size) only to cap classes below the data size, e.g. for speed.
  batch_prediction_sizes: [1,2,3,4,5,6,7,8]
  # classifiers / splitter_type / default_normalization / normalization_methods / batch_types
  # all omitted -> auto-applied defaults. Override any of them here when needed, e.g.:
  classifiers: [XGBoost, Random Forest]
  batch_types: [tp_wise]            # force a single mode instead of the derived pair
```

---

## `FigureConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str\|None` | `None` | Figure title |
| `type` | `str` | `"lineplot"` | Plot type: `barplot`, `lineplot` |
| `x_axis` | `str` | `""` | X-axis variable (YAML key: `x-axis`) |
| `y_axis` | `str` | `"metric"` | Y-axis variable (YAML key: `y-axis`) |
| `metrics` | `list[str]` | `[]` | Metrics to plot |
| `group_by` | `str` | `"none"` | Grouping |
| `aggregation` | `str` | `"mean"` | Aggregation function |
| `error_bar` | `str` | `"std"` | Error bar type |
| `repeat_for_each` | `str` | `"none"` | Generate separate figure per value of this iterator |

Hyphenated YAML keys `x-axis` and `y-axis` are remapped to `x_axis` and `y_axis`.

---

## Variation selection: `scenario` vs `model_variations`

There are **two interchangeable ways** to declare which model × variation cells become
classification classes, and **both ship by design**:

- **`scenario:`** — a declarative `config/scenarios/*.yaml` file expanded by `build_instances`
  (the forward, single-source path).
- **`model_variations` + `quantized_model_variations` + `abliterated_models`** — the direct
  grid keys. **The shipped configs currently use these**; they are not deprecated.

The two paths are held to the *identical* label set by `make gate-b-parity` (a data-free firewall
run on every clone), so they cannot silently diverge. The collapse to a scenario-only path is a
deliberate, **gated** manual release step (it requires accuracy-verifying the new path against the
HPC-only headline export) — not something to do per-config. Full rationale:
[`../reproduction/legacy-paths.md`](../reproduction/legacy-paths.md).

```yaml
# Single variation group
model_variations:
  temperature: [0.4, 0.6, 0.8, 1.0]
  system_prompt_idx: [-1]

# Multiple variation groups (list)
model_variations:
  - {temperature: [0.4, 0.6, 0.8, 1.0], system_prompt_idx: [-1]}
  - {temperature: [1.0], system_prompt_idx: [0, 3, 6, 7]}
```

---

## `calculations`

Defines iteration axes. Empty or absent = no explicit iteration axis.

```yaml
calculations:
  token_pairs: token_pairs    # iterate over all token pairs
```

Supported iterator names: `token_pairs`, `models`, `features`, or any sampling parameter column.

---

## `model_groups`

Group-based classification: models are grouped into named categories; the classifier predicts the
category. Models not listed in any group are excluded.

```yaml
# Hardcoded groups
model_groups:
  abliterated:
    - failspy/Meta-Llama-3-8B-Instruct-abliterated-v3
  non-abliterated:
    - meta-llama/Llama-3.1-8B-Instruct

# Parameter-based groups
model_groups:
  group_by: temperature   # one group per temperature value
```

Supported `group_by` values: `temperature`, `system_prompt_idx`, `abliteration`.

---

## Config organization on disk

```
XP_configs/
  +-- run_experiments.py          # CLI dispatcher
  +-- XP_script_global.py        # Per-experiment entry point
  +-- Final_FLiPS_ICML/          # Production ICML experiments
  +-- LLMmap/                    # LLMmap-specific experiments
  +-- e3_flips_vs_llmmap/        # E3 comparison (3-curve merge wrapper)
  +-- e3_llmmap_baseline/        # Out-of-box LLMmap-only E3 curve
  +-- Smoke_light/               # Light-subset smoke test config
  +-- XP_config_libs/            # Shared config fragments (excluded from discovery)
  +-- Old_configs/               # Legacy configs (excluded from discovery)
```

---

## Related documentation

| File | Content |
|------|---------|
| [configuration.md](configuration.md) | Simple config guide — the common fields only |
| [classification.md](classification.md) | Classification pipeline internals |
| [features.md](features.md) | Feature preset names (`SmallNist`, etc.) |
| [quantized-models.md](quantized-models.md) | `include_quantized` field details |
| [experiment.md](experiment.md) | How configs are discovered, validated, and dispatched |
</content>
