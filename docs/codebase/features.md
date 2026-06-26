# Feature Computation

Detailed documentation of the feature computation and caching layer in
`_compute_save_load_experiments()` (`src/audit_llm/data_loader.py`).

---

## Overview

Feature computation transforms raw LLM-generated bit sequences (and PRNG baselines) into numerical
feature matrices suitable for classification. Features are computed per model, per token pair, and
cached as individual files to allow incremental computation and fast loading.

---

## Entry point

```python
_compute_save_load_experiments(
    Experiments_path, Answers_df, TokenIDs_df, MainDataset_df,
    max_tokens, compute_config, Dataset_path
) -> (intra_features_dict, inter_features_map, model_index,
      intra_feature_index_dict, inter_feature_index_dict, compute_config)
```

Called from `run_xp()` in `experiment_runner.py` for FLiPS experiments only (not LLMmap).

---

## Pipeline steps

```
1. _prepare_models_with_PRNGs()
   |-- Collect unique models from Answers_df
   |-- Append PRNG models (numpy_default, numpy_mt19937, ...)
   +-- Build model_index: {model_name: int}

2. Per token_pair (tqdm loop):
   |
   |-- Check cache: skip if all <model>.npy + <model>.npz + manifest exist
   |
   |-- Per model (nested tqdm):
   |     |
   |     |-- [PRNG model]
   |     |     +-- make_random_bit_sequences(model, max_tokens, N=N_iter, seed)
   |     |
   |     |-- [LLM model]
   |     |     +-- Filter Answers_df by (Model, Token_pair)
   |     |     +-- Sort by Dataset_Question Index
   |     |     +-- Extract bit sequences aligned with MainDataset indices
   |     |
   |     |-- Compute inter-sample features -> save <model>.npz
   |     +-- Compute intra-sample features -> save <model>.npy
   |
   +-- Write manifest.json

3. Loading phase:
   |-- Per token_pair: load all <model>.npy, stack into 3D array
   |-- Per token_pair: load all <model>.npz, reconstruct inter-features dict
   +-- Validate source CSV hash against manifest
```

---

## Feature types

### Intra-sample features

Computed per individual bit sequence via `compute_intra_samples_bit_feature_matrix()` in
`Bits_Generation/bits_tools.py`. Output: 2D array `(N_samples, N_features)` per model.

| Feature category | Description | Index range |
|-----------------|-------------|-------------|
| NIST test results | Statistical randomness tests (test statistic and/or p-value per test) | `0 .. 2*nb_tests - 1` |
| Intra-sample features | Per-sequence statistics (e.g. `seq_length`) | `2*nb_tests .. 2*nb_tests + len(intra_features) - 1` |
| Token stats features | Per-token statistics (legacy, currently disabled) | Appended if `token_stats_dict` is set |

**NIST tests available** (configured in `XP_configs/XP_config_libs/FeaturesConfigs.yaml`):

| Test | Description |
|------|-------------|
| `monobit` | Proportion of 0s and 1s |
| `run` | Oscillation between 0s and 1s |
| `block frequency` | Frequency within blocks |
| `non overlapping` | Non-overlapping template matching |
| `overlapping patterns` | Overlapping template matching |
| `longest one block` | Longest run of ones |
| `binary matrix rank` | Rank of binary sub-matrices |
| `spectral` | Discrete Fourier transform test |
| `statistical` | Statistical test |
| `linear complexity` | Linear complexity |
| `approximate entropy` | Approximate entropy |
| `cumulative sums 1s` | Forward cumulative sums |
| `cumulative sums 2s` | Backward cumulative sums |
| `random excursion` | Random walk excursion |

Each test produces results of type `ts` (test statistic) and/or `pv` (p-value), configured via
`nist_tests_result_type`.

### Inter-sample features

Computed across all sequences for a model via `fill_inter_samples_features_map()`. Output:
`{(model_idx, feature_name): value}`.

| Feature | Description |
|---------|-------------|
| `Bits_Seqs_Bernoulli_probs` | Per-position probability of seeing a 1; shape `(seq_length,)` |
| `Bits_Block_Entropy` | Block entropy for block sizes 1–5 |
| `Similarities_on_bit_sequences` | Multiset Jaccard similarity scores (optional, resource-intensive) |
| `nb_of_sequences_after_seq_length_filter` | Count of sequences surviving the minimum length filter |

Sequences shorter than `seq_length_for_inter_sample_features` (default: 300) are dropped; longer
ones are truncated.

---

## Feature presets

Defined in `XP_configs/XP_config_libs/FeaturesConfigs.yaml`, referenced by name via `features` key.

| Preset | NIST tests | Result types | Intra features | Inter features |
|--------|-----------|--------------|----------------|----------------|
| `SmallNist` | All 14 | `ts` only | `seq_length` | — |
| `LongNist` | All 14 (more pattern sizes) | `ts` only | `seq_length` | — |
| `SmallNist_only_pv` | All 14 | `pv` only | — | — |
| `DefaultConfig` | All 14 | `ts` + `pv` | `seq_length` | `Bits_Seqs_Bernoulli_probs`, `Bits_Block_Entropy` |

---

## PRNG baseline models

| Model | Generator |
|-------|-----------|
| `numpy_default` | `np.random.randint` |
| `numpy_mt19937` | `np.random.Generator(MT19937)` |
| `numpy_pcg64` | `np.random.Generator(PCG64)` |
| `numpy_sfc64` | `np.random.Generator(SFC64)` |
| `python_random` | `random.randint` |
| `secrets` | `secrets.randbits` |
| `xor_shift` | Custom XorShift64 |

Each generates `N_iter` sequences of `max_tokens` bits. Seeds are incremented per model to ensure
independence.

---

## Caching format

### Directory structure

```
Experiments/
  +-- model_index.json                    # {model_name: idx} for all models
  +-- feature_computation_data/
        +-- <token_pair>/
              +-- intra/
              |     +-- <sanitized_model>.npy   # shape (N_samples, N_features)
              +-- inter/
              |     +-- <sanitized_model>.npz   # inter-sample features
              +-- manifest.json
```

### Per-model files

| File | Format | Content |
|------|--------|---------|
| `intra/<model>.npy` | NumPy `.npy` | 2D array `(N_samples, N_features)` — one row per dataset question |
| `inter/<model>.npz` | NumPy `.npz` | Named arrays; nested dicts use `__` separator |

### Manifest (`manifest.json`)

```json
{
  "models": ["model_a", "model_b"],
  "feature_index": {"monobit_ts": 0, "run_ts": 1},
  "inter_feature_index": {"Bits_Block_Entropy": 0},
  "source_csv_hash": "sha256_hex_digest"
}
```

- `models`: ordered list of sanitized model names (determines stacking order)
- `source_csv_hash`: SHA-256 of the source dataset CSV for staleness detection

### Cache invalidation

- If all `.npy` + `.npz` files and `manifest.json` exist for a token pair, the entire token pair
  is skipped — never regenerated when the manifest already marks them complete.
- Individual model files are checked: if both `.npy` and `.npz` exist, that model is skipped.
- Source CSV hash mismatch logs a warning but does not prevent loading.

Feature caches are keyed per `(token_pair, model)` as `<model>.npy` / `<model>.npz`; the
`<model>` component is the sanitized model name from `sanitize_model_name()` in `file_io.py`
(replaces `/` and other filesystem-unsafe characters).

---

## Loading phase

1. **Intra-sample features**: loads per-model `.npy` files in manifest order, stacks along axis 1
   to produce 3D array `(N_samples, N_models, N_features)` per token pair.
2. **Inter-sample features**: loads per-model `.npz` files, reconstructs
   `{(model_idx, feature_name): value}` dicts.
3. **Index dicts**: reads `feature_index` and `inter_feature_index` from manifest.

---

## Key files

| File | Key functions |
|------|---------------|
| `src/audit_llm/data_loader.py` | `_compute_save_load_experiments()`, `_prepare_models_with_PRNGs()` |
| `src/audit_llm/Bits_Generation/bits_tools.py` | `compute_intra_samples_bit_feature_matrix()`, `fill_inter_samples_features_map()`, `make_random_bit_sequences()` |
| `src/audit_llm/xp_tools/feature_selection.py` | `get_features_from_xp_config()`, `integrate_nist_test_parameters()` |
| `src/audit_llm/file_io.py` | `sanitize_model_name()`, `compute_file_hash()`, `write_manifest()`, `load_manifest()` |
| `XP_configs/XP_config_libs/FeaturesConfigs.yaml` | Feature preset definitions |

---

## Related documentation

| File | Content |
|------|---------|
| [experiment.md](experiment.md) | How `_compute_save_load_experiments()` is called from `run_xp()` |
| [classification.md](classification.md) | How the 3D feature arrays are consumed by the classification layer |
