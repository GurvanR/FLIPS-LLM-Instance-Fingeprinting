# Quantized Models Support

Support for quantized model variants (fp8, bitsandbytes_int4, …) as additional classification
columns alongside existing non-quantized models.

---

## Overview

Quantized model variants are identified by the `@@` separator in their model names (e.g.,
`Qwen/Qwen2-7B-Instruct@@fp8`). They appear as **new variation columns**, not new rows, in
heatmaps and result tables. The base model row is shared with its non-quantized counterpart.

Key design choices:
- Same heatmap row as the base model (both map to `Qwen/Qwen2-7B-Instruct`)
- New variation columns (e.g., `fp8_temp-0.6_sp--1`) alongside standard columns
- Only sp=-1 data expected for quantized models; sp≠-1 columns are all-NaN and filtered out
- Controlled per-config via the `include_quantized` field

---

## Config field: `include_quantized`

Added to `ExperimentConfig` in `src/audit_llm/experiment_config_schema.py`.

| Value | Behavior |
|-------|----------|
| `null` (default) | Exclude all quantized (`@@`-suffixed) models |
| `[]` | Include all quantized models |
| `["fp8", "bitsandbytes_int4"]` | Include only these specific quantization methods |

```yaml
# Include specific quantization methods
include_quantized: [fp8, bitsandbytes_int4]

# Exclude all (explicit)
include_quantized: null
```

---

## Model name convention

The `@@` separator (defined as `QUANTIZATION_SEPARATOR` in `src/audit_llm/file_io.py`) separates
the base model from the quantization method:

```
Qwen/Qwen2-7B-Instruct@@fp8_temp-0.6_sp--1
│                       │   │
│                       │   variation suffix (temp-X_sp-Y)
│                       quantization_suffix (after @@)
base_model_name (before @@)
```

This `@@` form is the **analysis/results-facing name**. During inference, `#` is used internally
(e.g. `model#fp8`) when vLLM is invoked; the `#` suffix carries the quantization key that keys
into `QUANTIZATION_VLLM_PARAMS`.

Two helpers in `src/audit_llm/file_io.py`:

```python
from audit_llm.file_io import get_base_model_name, get_quantization_suffix

get_base_model_name("Qwen/Qwen2-7B-Instruct@@fp8")   # -> "Qwen/Qwen2-7B-Instruct"
get_quantization_suffix("Qwen/Qwen2-7B-Instruct@@fp8") # -> "fp8"
```

---

## Pipeline changes

### `src/audit_llm/xp_tools/model_filtering.py`

`filter_models()` respects `include_quantized`:

```python
include_quantized = xp_config.get("include_quantized")
if include_quantized is None:
    models = [m for m in models if QUANTIZATION_SEPARATOR not in m]
elif include_quantized:
    models = [m for m in models if QUANTIZATION_SEPARATOR not in m
              or get_quantization_suffix(m) in include_quantized]
# else [] → keep all
```

`full_var_model_name_to_original_model_name()` calls `get_base_model_name()` so quantized variants
map to the **same heatmap row** as their base model.

`full_var_model_name_to_var_name()` prepends the quantization suffix so quantized variants appear
as **distinct columns**:

```
fp8_temp-0.6_sp--1    # quantized column
temp-0.6_sp--1        # standard column
```

---

## Supported quantized models

| Model | Quantizations available |
|-------|------------------------|
| `Qwen/Qwen2-7B-Instruct` | `fp8`, `bitsandbytes_int4` |
| `meta-llama/Meta-Llama-3-8B-Instruct` | `fp8`, `bitsandbytes_int4` |
| `microsoft/Phi-3-mini-4k-instruct` | `fp8`, `bitsandbytes_int4` |
| `mistralai/Mistral-7B-Instruct-v0.3` | `fp8`, `bitsandbytes_int4` |

The paper's 32 quantized instances (sp=-1 at temps 0.6/1.0; sp 0/3 at temp 1.0) ship as part of the
headline `Productions/FLiPS_ICML_run/` — see [`../provenance.md`](../provenance.md#3-the-237-instance-decomposition).

---

## Underscore ambiguity in `bitsandbytes_int4`

`bitsandbytes_int4` contains an underscore, which interacts with `rsplit("_")` used elsewhere.
This is handled correctly because:
- `get_base_model_name()` splits on `@@` first, cleanly stripping the quantization suffix
- `_parse_variation_label()` splits on `_temp-` to extract the quantization prefix

---

## Parse bug fix: `model_analysis.py`

A historical bug caused `SingleModelAuditionAnalysis.analysis()` to call `_save()` after each
generation file, overwriting previous data. The fix accumulates all generations first, then calls
`_save()` once. If parsing was already run with partial output, delete the affected parquets and
re-run `scripts/parsing_generations.py`.

---

## Related documentation

| File | Content |
|------|---------|
| [inference.md](inference.md) | `QUANTIZATION_VLLM_PARAMS` and backend routing |
| [configuration.md](configuration.md) | `include_quantized` field in `ExperimentConfig` |
| [scenarios.md](scenarios.md) | Scenario-driven selection of quantized instances and variation grids |
| [conventions.md](conventions.md) | `@@` and `#` separator conventions |
