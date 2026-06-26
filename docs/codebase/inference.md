# Inference Pipeline

---

## Overview

The inference pipeline produces raw LLM generations from a prompt CSV dataset. It is CLI-driven
(`scripts/Run_Inferences.py`), builds a validated `run_config.json` that describes the run (models,
paths, sampling defaults, environment), dispatches one backend instance per model, iterates over
unique `(temperature, frequency_penalty)` sampling keys extracted from the dataset, and writes one
Parquet checkpoint per model per sampling key under `Productions/.../<model>/`.

vLLM is the primary (fully-implemented) backend. HuggingFace and OpenRouter are scaffolded in the
config and dispatch layers but are **not** the active execution path — see
[Backend coverage](#backend-coverage) below.

---

## File structure

| File | Description |
|------|-------------|
| `scripts/Run_Inferences.py` | CLI entry point: parses args, loads `Inference_configs.yaml`, expands model × quantization, builds vLLM/HF/OpenRouter configs, calls `make_run_config()` then `run_inferences()` → `multi_model_infer()` |
| `scripts/Inference_configs.yaml` | Per-dataset inference defaults (`TOKEN_PAIRS_SET`, `MAX_TOKENS`, `TOP_K`, `max_model_len`, `min_seq_length`, `dyn_checking_batch_size`, optional `TEMPERATURES`, `logprobs`) |
| `config/models.yaml` | Model universe registry: HF ids, quantization levels, abliterated repos |
| `src/audit_llm/LLM_Classes/inference_config_schema.py` | `InferenceConfig` Pydantic validator + legacy key aliases |
| `src/audit_llm/LLM_Classes/run_config.py` | `RunConfigDict` + `make_run_config()` / `update_run_config()` |
| `src/audit_llm/LLM_Classes/inference_runner.py` | `run_inferences()` and `multi_model_infer()` |
| `src/audit_llm/LLM_Classes/General_LLM_Class.py` | Abstract base `LLM_for_audit` + `LLM_Generation` dataclass |
| `src/audit_llm/LLM_Classes/vLLM_Classes.py` | `vLLM` backend: instantiation, `QUANTIZATION_VLLM_PARAMS`, `_generate()` with dynamic batching + retry |
| `src/audit_llm/LLM_Classes/fallback_templates.py` | Chat-template fallbacks when `tokenizer.apply_chat_template` fails |
| `src/audit_llm/LLM_Classes/generation_parser.py` | `parsing_generations()`: post-process Parquet checkpoints into `Answers.csv` |
| `src/audit_llm/Bits_Generation/parsing_bits_tools.py` | `validate_seq()`: determines if a generation is valid |

---

## End-to-end flow

```
Run_Inferences.py (CLI)
  |
  |-- 1. argparsing(args)
  |-- 2. Load Inference_configs.yaml[dataset]        -> Inf_config
  |-- 3. Model name resolution:
  |        --model          -> [parsed_args['model']]
  |        (else)           -> default model set
  |-- 4. Build vllm_model_config / hf_model_config / open_router_config
  |-- 5. Snapshot Inference_configs.yaml into run_path
  |-- 6. make_run_config(...)                        -> run_config.json (+ run_metadata.csv)
  |
  +-- 7. run_inferences(run_name, Productions_path, run)
          |
          +-- multi_model_infer(run_name, Productions_path)
                |
                |-- Load run_config.json (or pickle fallback)
                |
                +-- For each model in run_config['vllm_models'] where is_done==False:
                      |
                      |-- save_tokenizer_vocab_from_model(base_name, model_path)
                      |-- audited_llm = vLLM(model_name, run_config, run_path, model_path)
                      |
                      |-- audited_llm.multi_dataset_infer(dataset_path)
                      |     |
                      |     |-- polars.read_csv(Dataset_path)
                      |     |-- unique (temperature, frequency_penalty) -> sampling_keys
                      |     |
                      |     +-- For each sampling_key:
                      |           |-- _format_prompts_for_model()       -> List[LLM_Generation]
                      |           |-- _generate(sampling_key)            -> fill outputs, retry, validate
                      |           +-- _save_generations(terminated=True) -> parquet
                      |
                      |-- run_config['vllm_models'][model_name] = True
                      |-- update_run_config(...)
                      +-- destroy_model_parallel(); del audited_llm; gc; torch.cuda.empty_cache()
  |
  +-- 8. (optional) parsing_generations(run_name, Productions_path) -> Answers.csv
```

---

## CLI arguments

| Arg | Default | Purpose |
|-----|---------|---------|
| `--dataset` | `Toy_example` | Key into `Inference_configs.yaml`; derives `Dataset_relative_path = datasets/Bits_Datasets/<dataset>.csv` |
| `--model` | `""` | Full HF repo id. Empty ⇒ default model set |
| `--sub_run` | `default` | Per-model-set suffix appended to `run_name` as `<dataset>_run/<sub_run>` |
| `--gpu` | `1` | Tensor-parallel size for vLLM |
| `--bs` | `20` | HF-only batch size (HF backend is scaffolded only) |
| `--openrouter` | `False` | Route all models to OpenRouter |
| `--erase_previous_run` | `False` | Remove the run folder before starting instead of resuming |
| `--parse_gen` | `False` | After inference, run `parsing_generations()` to produce `Answers.csv` |
| `--hours_delay` | `0.0` | Sleep inside `multi_model_infer` before work starts |
| `--device` | `0` | `CUDA_VISIBLE_DEVICES` |
| `--test` | `False` | Routes outputs under `Productions/Graph_Productions/Test_Runs/` |
| `--seed` | `42` | Reserved; passed through for reproducibility |

For HPC/SLURM usage, see [`docs/reproduction.md`](../reproduction.md).

---

## `Inference_configs.yaml` fields

Validated by `InferenceConfig`. Legacy aliases map to canonical names.

| Key (YAML) | Canonical | Purpose |
|------------|-----------|---------|
| `min_seq_length` | `min_seq_length` | Minimum scrapped bit-string length for a generation to be valid |
| `dyn_checking_batch_size` | `dyn_checking_batch_size` | Dynamic batch size fed to `vllm.LLM.generate()` per step |
| `TOP_K` | `top_k` | Sampling top-k |
| `max_model_len` | `max_model_len` | Context window passed to vLLM |
| `MAX_TOKENS` | `max_tokens` | Max tokens generated per request |
| `TOKEN_PAIRS_SET` | `token_pairs_set` | Key into `TOKEN_PAIRS_SETS_DICT` (see [dataset.md](dataset.md)) |
| `TEMPERATURES` | `temperatures` | Optional temperature sweep (deprecated: temperatures come from the dataset) |
| `logprobs` | `logprobs` | Top-k logprobs to capture (optional) |
| `nb_of_samples` | `nb_of_samples` | Deprecated/metadata only |

---

## `run_config.json` schema

| Group | Fields |
|-------|--------|
| Identity | `run_name`, `scrapping_rule`, `created_at`, `Initial_checkpoint` |
| Sampling / dataset | `min_seq_length`, `dyn_checking_batch_size`, `TOKEN_PAIRS_SET`, `MAX_TOKENS`, `Dataset_relative_path`, `hours_delay` |
| Backend configs | `vllm_model_config`, `hf_model_config`, `openrouter_model_config` |
| Model maps | `vllm_models`, `hf_models`, `openrouter_models` (done-flags); `*_model_path`; `quantization_map` |
| Progression | `vllm_models_progression`, `hf_models_progression`, `openrouter_models_progression` |
| Environment | `environment`: `python_version`, `vllm_version`, `torch_version`, `cuda_version`, `gpu_type` |

`run_config.json` is the authoritative record of a run. `update_run_config()` flips
`Initial_checkpoint=False`, updates done-flags, and refreshes `Productions/run_metadata.csv`.

---

## Prompt formatting

`LLM_for_audit._format_prompts_for_model()` converts each dataset row into one or more
`LLM_Generation` objects:

1. Load `system_prompts*.json` once.
2. For each row: resolve the raw prompt via `prompt_idx_to_actual_prompt(prompt_idx, token_pair)`.
3. Apply tokenizer chat template with fallback chain:
   1. `tokenizer.apply_chat_template([{system}, {user}], add_generation_prompt=True)`
   2. Merge system into user turn and retry.
   3. `format_with_fallback()` in `fallback_templates.py` (manual template per model family).
   4. Plain concatenation as last resort.
4. Instantiate one `LLM_Generation` per `(row, token_pair)`.

---

## Generation loop

`vLLM._generate(sampling_key)` consumes the pending `LLM_Generation` buffer in dynamic batches:

1. Fill a working batch up to `dyn_checking_batch_size` from the pending queue.
2. Call `vllm.llm.generate(prompts, SamplingParams(**sampling_config, **sampling_key))`.
3. `validate_seq()` checks: `len(answer_to_bit_string(text, token_pair)) >= min_seq_length`.
4. Per-generation retry counter increments on every attempt; max is `max_gen_counter = 5`. When
   exceeded, the generation is marked `fail=True` and the last output is kept.
5. `_save_generations()` checkpoints every 250 completions to `{temp}-{fp}_generations.parquet`,
   and once more at the end to `{temp}-{fp}_generations_terminated.parquet`.

---

## Output structure

```
Productions/Graph_Productions/<Production_folder>/<dataset>_run/<sub_run>/
  +-- run_config.json
  +-- Inference_configs.yaml              (snapshot at run start)
  +-- <model_name>/
  |     +-- {temp}-{fp}_generations.parquet           (in-progress)
  |     +-- {temp}-{fp}_generations_terminated.parquet (final)
  |     +-- {temp}-{fp}_execution_time.txt
  +-- Output_logs/generations/
  |     +-- output.log / errors.log
  +-- Answers.csv                         (only if --parse_gen)

Productions/Graph_Productions/<Production_folder>/
  +-- run_metadata.csv
```

`Production_folder` is one of `Normal_Runs`, `Test_Runs` (`--test`), or `Toy_Runs` (dataset
`Toy_example`).

### Parquet row schema

`formatted_prompt`, `token_pair`, `dataset_idx`, `output_text`, `scrapped_output`,
`output_token_ids` (JSON), `output_logprobs` (JSON), `fail`, `success`, `gen_counter`.

### Filename trap

Raw generation files are named `{temperature}-{frequency_penalty}_generations_terminated.parquet`.
**The second number is the frequency penalty, not the system prompt index.** `system_prompt_idx` is
not encoded in the filename — it lives inside dataset rows. One file covers all `system_prompt_idx`
values that share the same `(temperature, frequency_penalty)`.

---

## Backend coverage

| Backend | State | Notes |
|---------|-------|-------|
| vLLM | Fully implemented | `vLLM._generate()` is the only concrete backend used in `multi_model_infer()` |
| HuggingFace (`transformers`) | Scaffolded only | `run_config.hf_models` is populated, but `multi_model_infer()` contains no HF dispatch loop |
| OpenRouter | Scaffolded only | `_format_prompts_for_model()` raises `NotImplementedError` for OpenRouter |

---

## Related documentation

| File | Content |
|------|---------|
| [dataset.md](dataset.md) | CSV schema, prompt configs, system prompts, token-pair sets |
| [experiment.md](experiment.md) | What happens after inference: feature computation, classification |
| [quantized-models.md](quantized-models.md) | `QUANTIZATION_VLLM_PARAMS` and per-quantization caveats |
| [configuration.md](configuration.md) | Full Pydantic reference for experiment configs |
| [../reproduction.md](../reproduction.md) | HPC/SLURM usage, Mode B step-by-step |
