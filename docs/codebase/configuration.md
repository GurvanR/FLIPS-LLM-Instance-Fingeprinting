# Configuration

Experiments are declared in YAML and validated by Pydantic
(`src/audit_llm/experiment_config_schema.py`). A working config only needs a handful of
keys — sensible defaults cover everything else.

> **Need the full parameter set** (CV splitters, normalization, batch modes, open-set internals,
> scenario & variation keys)? See **[configuration-advanced.md](configuration-advanced.md)**.

---

## Minimal example

```yaml
experiment_fun: batch_classification_across_token_pairs
features: SmallNist
models: []            # [] = all available models
token_pairs: []       # [] = all token pairs
train_sizes: [40]

calculations:
  token_pairs: token_pairs

classification_config:
  n_splits: 10
  test_size: 64                          # samples per class held out
  batch_prediction_sizes: [1, 2, 3, 4, 5, 6, 7, 8]
  # force_class_size defaults to "auto" (balance to the smallest class) — omit it
```

That is enough to run. The classifier, CV splitter, normalization and batch modes are all
defaulted — see [Auto-applied defaults](#auto-applied-defaults) below.

---

## Common top-level fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `experiment_fun` | `str` | **required** | What to run (e.g. `batch_classification_across_token_pairs`) |
| `features` | `str` | `None` | Feature set name (e.g. `SmallNist`) |
| `models` | `list[str]` | `[]` | Model filter; `[]` = all available |
| `token_pairs` | `list[str]` | `[]` | Token-pair filter; `[]` = all |
| `train_sizes` | `list[int]` | `None` | Training sizes to sweep |
| `scenario` | `str` | `None` | Declarative model × variation selection (`config/scenarios/*.yaml`) |
| `sampling_parameters` | mapping | — | Which sampling values to include (see advanced) |
| `calculations` | mapping | `None` | Iteration axes, e.g. `token_pairs: token_pairs` |

---

## Common `classification_config` fields

These control the train/test split — and therefore the scientific result — so they stay explicit.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_splits` | `int` | `2` | Number of CV splits |
| `test_size` | `float\|int` | `0.5` | Test set size (`int` = samples **per class**, `float` = fraction) |
| `force_class_size` | `int\|"auto"\|None` | `"auto"` | Balance every class to this many rows (`"auto"` = smallest class) |
| `batch_prediction_sizes` | `list[int]` | `None` | Batch sizes to evaluate |
| `openset` | `bool` | `false` | Switch to open-set classification |
| `classifier_metrics` | `list[str]` | `["accuracy"]` | Metrics to compute |

---

## Auto-applied defaults

You do **not** need to set these — they are filled in automatically. Add them to the YAML only to
override (full details in [configuration-advanced.md](configuration-advanced.md)):

| Field | Default applied | |
|-------|-----------------|--|
| `classifiers` | `["XGBoost"]` | |
| `splitter_type` | `"StratifiedShuffleSplit"` | |
| `default_normalization` | `"auto"` | |
| `normalization_methods` | `{seq_length: none}` | |
| `batch_types` | **derived from `batch_prediction_sizes`** | size `1` → `tp_wise`; any size `> 1` → `mix_tp_at_pred` |

---

## Related documentation

| File | Content |
|------|---------|
| [configuration-advanced.md](configuration-advanced.md) | Every config field, defaults, open-set, normalization, scenario/variation keys |
| [classification.md](classification.md) | Classification pipeline internals |
| [features.md](features.md) | Feature preset names (`SmallNist`, etc.) |
| [experiment.md](experiment.md) | How configs are discovered, validated, and dispatched |
</content>
