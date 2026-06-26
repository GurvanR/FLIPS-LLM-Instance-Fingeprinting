# Dataset Guide

---

## Overview

A "dataset" in FLIPS is a **CSV** (Polars-readable) where each row describes one inference request
sent to an LLM. Rows supply sampling parameters (`temperature`, `frequency_penalty`) and prompt
identifiers (`prompt_idx`, `system_prompt_idx`); the actual prompt text is assembled at inference
time from those indices and auxiliary config files under `datasets/`.

Sampling parameters live in the CSV rows — **not** in `Inference_configs.yaml`. The inference loop
groups rows by unique `(temperature, frequency_penalty)` pairs extracted directly from the CSV.

---

## File structure

| File | Description |
|------|-------------|
| `datasets/Bits_Datasets/*.csv` | The datasets themselves; consumed by `Run_Inferences.py` via `Dataset_relative_path` |
| `datasets/prompt_config_index.yaml` | Int-keyed registry mapping `prompt_idx` values to prompt configurations |
| `datasets/system_prompts.json` | System prompts for standard datasets. Indexed by `system_prompt_idx`; `-1` means no system prompt |
| `datasets/system_prompts_llmmap.json` | System prompts for LLMmap-style datasets |
| `src/audit_llm/Bits_Generation/Bits_Dataset_Making/Bits_Dataset_Maker.ipynb` | Canonical dataset-creation notebook |
| `src/audit_llm/Bits_Generation/Bits_Dataset_Making/Prompts.py` | Prompt-assembly helpers: `prompt_idx_to_actual_prompt()`, `build_full_prompt()`, `create_single_csv()` |

---

## Required CSV schema

| Column | Type | Purpose |
|--------|------|---------|
| `Index` | int, unique | Row identifier; assert `n_unique == height` when building |
| `prompt_idx` | int | Key into `datasets/prompt_config_index.yaml` |
| `system_prompt_idx` | int | Key into `system_prompts*.json`. Use `-1` for no system prompt |
| `temperature` | float | Sampling temperature; drives `(temperature, frequency_penalty)` grouping |
| `frequency_penalty` | float | Second half of the sampling-key grouping |

Downstream code tolerates additional columns; any extra sampling-parameter column automatically
becomes an available iteration axis in experiments.

---

## Prompt assembly flow

```
CSV row: (prompt_idx=M, system_prompt_idx=N, temperature, frequency_penalty)
  |
  |-- datasets/prompt_config_index.yaml[M]  -> prompt configuration dict
  |-- datasets/system_prompts*.json[N]       -> system prompt string (or none if N == -1)
  |
  v
Prompts.build_full_prompt(prompt_config)
  |
  v
General_LLM_Class._format_prompts_for_model()
  |-- tokenizer.apply_chat_template([system, user], add_generation_prompt=True)
  v
formatted_prompt  -> vLLM.generate()
```

`prompt_idx_to_actual_prompt()` is the inference-time entry point — it looks up
`prompt_config_index.yaml[prompt_idx]` and delegates to `build_full_prompt()`.

### `prompt_config_index.yaml` entry example

```yaml
1:
  instruction_prompt: ip2
  nb_of_bits: 1000
  learning_shot: 1
  seed: 42
  pre_prompt: sp0
  end_context: None
```

Add a new entry to introduce a new prompt template without changing any code.

---

## Dataset variants

| Aspect | Standard | LLMmap |
|--------|----------|--------|
| `prompt_config_index.yaml` entry | `instruction_prompt`, `nb_of_bits`, `learning_shot`, … | `llmmap: K` (index into `llmmap_queries.json`) |
| System prompts | `datasets/system_prompts.json` | `datasets/system_prompts_llmmap.json` |
| Filename convention | any name | must contain `"LLMmap"` — the pipeline uses this substring to branch to the embedding-based analysis path |
| Downstream analysis | NIST-based feature classification | Embedding-based classification (`LLMmap_classification`) |

---

## Creating a new dataset

Datasets are authored in `Bits_Dataset_Maker.ipynb`. Use existing cells for `FLiPS_ICML`,
`LLMmap_ICML`, or `FLiPS-Monochar-0-1` as starting points.

**Recipe:**

1. Define the parameter grid (temperatures, system prompts, prompt indices, repetitions).
2. Add a `prompt_config_index.yaml` entry if the prompt template does not yet exist.
3. Author rows in the notebook:
   ```python
   import polars as pl
   from audit_llm.path_utils import get_repository_level_path

   NAME, N = "MyDataset", 500
   rows, k = [], 0
   for sp_idx in [-1, 0]:
       for temp in [0.4, 1.0]:
           for _ in range(N):
               rows.append({"Index": k, "prompt_idx": 1,
                             "system_prompt_idx": sp_idx, "temperature": temp,
                             "frequency_penalty": 0.0})
               k += 1
   df = pl.DataFrame(rows)
   assert df.select("Index").n_unique() == df.height
   df.write_csv(get_repository_level_path() / "datasets" / "Bits_Datasets" / f"{NAME}.csv")
   ```
4. Register in `scripts/Inference_configs.yaml`:
   ```yaml
   MyDataset:
     min_seq_length: 100
     dyn_checking_batch_size: 500
     TOP_K: 50
     max_model_len: 2048
     MAX_TOKENS: 500
     TOKEN_PAIRS_SET: ICML_SET_OF_TOKEN_PAIRS
   ```
5. Run inference: `python scripts/Run_Inferences.py --dataset MyDataset --model <hf_id> --sub_run run1`

→ To run the full classification pipeline on the variations you just generated, see
[`../reproduction/custom-scenario.md`](../reproduction/custom-scenario.md) (this dataset recipe is
its inference-side "Path B").

---

## Consumption surface

- **Inference**: `LLM_for_audit.multi_dataset_infer` reads the CSV with Polars and dispatches one
  generation batch per unique `(temperature, frequency_penalty)` pair.
- **Analysis**: `_analysis_mode()` reloads the CSV into `MainDataset_df`;
  `extract_global_samples_indices()` filters rows by sampling parameters from the experiment YAML.
- **Variant branching**: the substring `"LLMmap"` in `Dataset_path` switches `run_xp()` to the
  LLMmap classification path.

---

## Related documentation

| File | Content |
|------|---------|
| [inference.md](inference.md) | How the CSV is consumed: backend routing, vLLM generation loop, Parquet outputs |
| [experiment.md](experiment.md) | How `MainDataset_df` is used downstream, sampling-parameter filtering |
| [configuration.md](configuration.md) | `sampling_parameters` schema that keys off the CSV sampling columns |
