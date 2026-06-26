# Conventions

Project-wide conventions. When you learn or confirm a non-trivial convention (naming, layering,
error-handling pattern, stack quirk), append it here rather than leaving it implicit.

Keep this file readable in one sitting (~150 lines). When it outgrows itself, split into
`docs/codebase/conventions/<topic>.md` and update the link in `overview.md`.

## Model name separators

- **`@@` is the analysis/results-facing separator** between base model name and quantization method
  (e.g. `Qwen/Qwen2-7B-Instruct@@fp8`). Defined as `QUANTIZATION_SEPARATOR` in
  `src/audit_llm/file_io.py`. Used in classification class labels, feature cache filenames, and
  heatmap columns.
- **`#` is the internal inference separator** used during vLLM dispatch (e.g. `model#fp8`); the
  `#` suffix keys into `QUANTIZATION_VLLM_PARAMS` to resolve the vLLM engine kwarg. This form
  never appears in analysis outputs or user-facing labels — always convert to `@@` before storing.

## Secrets and paths

- **Environment variables only** — never hardcode model paths, API keys, or cluster paths in
  source code. Every secret or environment-specific path is read from the environment; `.env.example`
  at the repo root enumerates the full set.
- **`config/models.yaml` is the single source of truth** for the model universe (25 base LLMs,
  quantized aliases, abliterated repos) and any cluster-specific filesystem paths. Cluster users
  overlay paths via `config/models.hpc.example.yaml` (copy locally, never commit real paths).
  Nothing in source code hardcodes the model list or local cache paths.

## Run record

- **`run_config.json` is the authoritative record of a run**: it captures the models used, their
  resolved paths, sampling defaults, and an environment snapshot (Python/vLLM/torch/CUDA versions,
  GPU type). Always present under a run root; `update_run_config()` keeps it current as models
  complete.

## Feature caches

- **Feature caches are keyed per `(token_pair, model)`** as `<sanitized_model>.npy` /
  `<sanitized_model>.npz` under
  `Experiments/feature_computation_data/<token_pair>/intra|inter/`. The manifest (`manifest.json`)
  records which models are cached and the source CSV SHA-256. A cache is **never regenerated** if
  the manifest already marks all model files complete — delete the relevant files to force
  recomputation.
- **Feature computation parallelism** (`data_loader._compute_save_load_experiments`) fans out one
  worker per token-pair via a `forkserver` `ProcessPoolExecutor`, using **bounded streaming
  submission**: at most `max_workers + 1` per-token-pair slices are in flight at once (backfilled as
  workers finish). Peak RAM is `(two resident full frames) + (workers+1 in-flight slices)`; the
  streaming bounds only the second, *transient* term (was ~`len(token_pairs)` slices when all pairs
  were submitted up front). It does **not** shrink the resident-frame floor — `Answers_df`/`TokenIDs_df`
  stay live for on-demand slicing and the caller reuses them afterward — so lowering the worker count
  trims only the slice window; an OOM on the resident frames is a separate, larger problem. Do **not**
  revert to submitting all token pairs up front (that re-creates the transient ~full-copy spike).
  Worker count comes from `_resolve_feature_workers()`: default `min(8, cpu-2)`, clamped to
  `[1, 4*cpu]`, overridable via the `FLIPS_FEATURE_WORKERS` env var. Worker count affects only
  speed/RAM, never feature values.

## Path and secrets conventions

- **AUDIT_LLM_MODEL_CACHE**: root of the local Hugging Face model cache; when unset, model loading falls back to `$HF_HOME/hub` (and `$HF_HOME` itself defaults to `~/.cache/huggingface`). Always resolved as `Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"` — the `/hub` suffix is mandatory because weights live under `$HF_HOME/hub`, not `$HF_HOME` directly.
- **OPENROUTER_API_KEY**: the OpenRouter key, read from the environment by `audit_llm.system_utils.get_openrouter_key()`; there is no key file — the function raises `KeyError` pointing at `.env.example` if the variable is missing; one variable covers both test and prod (the old `test_mode` file-read split is gone).
- **AUDIT_LLM_ env-prefix**: `audit_llm.config.PathConfig` is a Pydantic settings model with `env_prefix="AUDIT_LLM_"`; every path field is overridable from the environment by upper-casing the field name with that prefix (e.g. `AUDIT_LLM_MODEL_CACHE_DIR`, `AUDIT_LLM_HF_CACHE_DIR`); `.env.example` at the repo root enumerates the full set; `PathConfig.hf_cache_dir` defers to `model_cache_dir` when the latter is set.
- **config/models.yaml**: the single source of truth for the model universe (the 25 base LLMs plus quantized and abliterated variants, each by Hugging Face Hub id); the analysis-layer instance enumerator reads it and nothing hardcodes the model list; cluster users overlay filesystem paths via `config/models.hpc.example.yaml` (copy + adapt locally, never commit real paths); quantized aliases use the `@@` `QUANTIZATION_SEPARATOR` from `audit_llm.file_io`; abliterated variants carry the base -> abliterated_hf_id mapping preserved from `models_management.model_names.ABLITERATED_MODELS_MAP_TO_ORIGINAL`.

- **config/scenarios/**: declarative scenario YAML files selecting which model × variation instances the analysis layer materialises, over four axes (`temperature`, `system_prompt_idx`, `quantization`, `abliteration`) with cartesian-within-group / union-across-groups semantics. Loaded by `audit_llm.scenarios.load_scenario`, which validates every model selection against `config/models.yaml`. `quantization`/`abliteration` transform the *model* (a vLLM engine kwarg vs a *different weights repo*); `temperature`/`system_prompt_idx` are sampling-request fields. Abliteration is pinned to `temperature=1.0 / system_prompt_idx=-1` (the loader rejects any other abliterated cell — no released data exists). **CSV invariant:** the scenarios package never emits/regenerates any prompt-request CSV — `datasets/Bits_Datasets/*.csv` are consumed unchanged and their SHA-256 (`compute_file_hash` → `source_csv_hash`) gates the feature cache. Full contract: `config/scenarios/README.md`.

- **Load-bearing model-label strings (`audit_llm.scenarios.variation` / `resolver`)**: a resolved `Instance(base_model, temperature, system_prompt_idx, quantization, abliterated)` exposes `storage_name` (on-disk / `model_idx` id) and `label` (classification-class name) that must stay **byte-identical** to the legacy strings parsed by `xp_tools.model_filtering` (`full_var_model_name_to_original_model_name` / `full_var_model_name_to_var_name`). Conventions: quantized storage = `{base}@@{quant_key}` (`@@` = `QUANTIZATION_SEPARATOR`, the suffix is the quant *key* e.g. `bitsandbytes_int4`, distinct from the vLLM engine kwarg *value* `bitsandbytes`); base/quant label = `{storage_name}_temp-{t}_sp-{sp}` (variation suffix built via `label_formatting.assemble_iterator_name_and_value`, temperature first, `temp-1.0` keeps its trailing `.0`, `sp--1` for sp=-1); abliterated label = `{abliterated_repo}_ablit` with **no** temp/sp suffix (mirrors `"_".join([model_name, "ablit"])` in `data_preparation.py`, where `{abliterated_repo}` is the *different weights* repo, not the base). The two model-transform axes are isolated in `scenarios.resolver`: `resolve_quantization` **reuses** `QUANTIZATION_VLLM_PARAMS` (a vLLM kwarg), `resolve_abliteration` **reuses** the inverse of `ABLITERATED_MODELS_MAP_TO_ORIGINAL` (a different repo) — never re-declare either. NB the round-trip parsers require model/repo ids contain no `_` (current ids use only `/ - . digits`). Importing `variation`/`resolver` pulls in `vLLM_Classes`, so the package `__init__` deliberately exports only the light schema/loader path.

- **Analysis-layer scenario consumption (`audit_llm.scenarios.enumerator`)**: `build_instances(scenario)` is the analysis-layer **source of truth** for which model × variation instances exist and their labels; `build_analysis_variation_structures(scenario, df)` is the adapter that derives the analysis structures (`model_variations_indices` / `quantized_model_variations_indices` / abliterated repos / the abliteration `(temperature, system_prompt_idx)` pin). `experiment_runner._xp_config_init` prefers a `scenario:` xp_config key (path to `config/scenarios/*.yaml`) and falls back to the legacy `model_variations` / `quantized_model_variations` / `abliterated_models` keys when absent — **both coexist**. On the scenario path the abliterated pin flows as `Experiment_config["abliterated_variation"]`; `prepare_dataset_features` reads it instead of the hardcoded `(temperature==1.0)&(system_prompt_idx==-1)` filter (which stays physically present as the legacy fallback). The legacy `compute_model_variations_indices` (`xp_tools.variation_context`) is superseded but kept present for backward compatibility; the adapter **owns its own sample-index grouping** and never calls it. This is a *shared scenario enumerator consumed by the analysis layer* — the inference path (`scripts/Run_Inferences.py`) keeps its own vLLM expansion (documented asymmetry); do not call it a unified inference+analysis source of truth.

## Reproduction data fixtures (light subset, manifest, golden refs)

- **`Productions/FLiPS_ICML_light_subset/`** is the in-repo, zero-download CPU smoke fixture (carved by `scripts/make_light_subset.py` from the sibling `audit-llm` canonical run, kept READ-ONLY via a `cp --reflink=auto` scratch). It is **flattened**: `run_config.{json,pickle}` + `Analysis/` + `Experiments/` sit directly under the subset root — the run-path convention `AuditionsAnalysis(run_path)` requires (`Analysis_Classes.py:48-55` accepts `run_config.json` *or* `.pickle`; `XP_script_global.py` sets `run_path = Productions/<RUN_NAME>`). The same flattening is what `fetch_data.py` must produce for the downloaded headline run.
- **`run_config` must be scrubbed before committing**: the canonical pickle carries HPC `/lustre` paths (`model_path_JZ`, `vllm_model_path`, `hf_model_path`). The carver drops `model_path_JZ` and blanks the `*_model_path` dicts (kept as `{model: ""}` so analysis lookups never `KeyError`); those paths are only used by the GPU inference path, never the cached analysis classify.
- **Feature-cache carve invariants** (so a subset stays correct AND tiny): `intra/<model>.npy` is `(N_iter=len(datasets/Bits_Datasets/FLiPS_ICML.csv)=10000, 57)`; npy row `i` ↔ CSV row `i`, and `get_samples_indices` selects rows via the CSV variation grid (NOT Answers row order). So you may subset the **models** and **token-pairs** axes but must keep npy rows fully intact and the CSV byte-identical (its SHA-256 must match every `manifest.source_csv_hash`). The feature array's model-axis = `manifest["models"]` order; the model list = `sorted(Answers.Model.unique())` + `PRNG_MODELS` (only `numpy_default`), so `numpy_default` must stay in `manifest["models"]` + `intra/` for axis alignment even though `PRNGs: None` keeps it out of the classes.
- **`golden_refs.json`** (subset root) is the out-of-box golden reference: `label_set` (materialised class labels, from `…/checkpoints/new_var_models_idx.json`), `feature_cache_md5` (per shipped npy), and `smoke_results` (per-`(batch_type,bs,clf)` accuracy from the joblib `train_size*_all.pkl` checkpoint — the closed-set path stores a `{batch_type:{bs:{tp:{clf:{accuracy_mean…}}}}}` pickle, NOT an LLMmap-style `results.json`). It is captured by running the **shipped** `XP_configs/Smoke_light/Smoke_light.yaml` (the SAME config `make gate-golden` re-runs).
- **`XP_configs/Smoke_light/Smoke_light.yaml`** quirks learned the hard way: `include_quantized` is a strict list — use YAML `null` (not the string `None`) to exclude; `token_pairs` must be an explicit list (an empty `[]` resolves to the full `TOKEN_PAIRS_SET` and the classifier then `KeyError`s on uncarved pairs); `mix_tp_at_pred` requires token pairs that are members of the `FLiPS` group (`'0-1'` is the lone non-member). For `unique_tp_in_mix` the rule is: `'max'` (the schema default) draws `bs` distinct pairs per mixed batch, so it needs `C(n_tp, bs) >= max_nb_of_uplet` for every **plotted** `bs` — feasible on the light subset because it now ships **5** pairs and the subset configs cap `batch_prediction_sizes` at ≤4 (`C(5,4)=5`). A run with too few pairs at a given `bs` must instead pin `unique_tp_in_mix` to a small int (e.g. `[2]`). (Historically the F07 figure layer eagerly built uplets for the hardcoded `bs∈2..8` regardless of the config and crashed on the unused sizes; `get_tp_names_of_group` now scopes the build to the single requested `bs`, so only the plotted batch sizes need to be feasible.) `force_class_size` is omitted (uses the `"auto"` default — balance to the smallest class, ~500 rows here), and when set as an explicit int it must be strictly `> test_size`, now enforced at config-load by `ClassificationConfig._validate_force_class_size` rather than as a deep sklearn crash.
- **`data/manifest.sha256`** lists real per-file SHA-256 for present members (`gen_manifest.py`), `0000…`×64 only for the off-disk headline `FLiPS_ICML_run`; entries with the zero hash are skipped as external by the verifier.
- **`scripts/fetch_data.py` verify contract**: URL precedence is `--url` > `$FLIPS_DATA_URL` > the `ZENODO_RECORD_URL` constant (a `PLACEHOLDER_DOI`; a real fetch against it is *refused*). `--check-only` classifies every manifest entry as PASS / FAIL / MISSING / EXTERNAL and exits non-zero **only on a hash mismatch** — a not-yet-fetched member is MISSING (benign), a `0000…` hash is EXTERNAL/skipped. The download path verifies each unpacked file against the manifest and rolls the member back on mismatch; members are keyed by the top-level `Productions/<member>` component, and the `headline` record is unpacked *flattened* (the extractor locates the dir holding `run_config.*` and strips intermediate levels like `merged_sub_runs/`).

## Reproduction wiring (merge-wrapper configs, gate-A)

- **`train_size_dict_map` = "merged-wrapper mode" → only F01 is plotted.** A `Batch_Classification_across_token_pairs` config that sets `classification_config.train_size_dict_map` regenerates the F01 superposition figure by *loading* the named per-source checkpoints (one `(path → label)` entry per curve) instead of plotting its own classification. `plot_classifier_curves` therefore early-returns right after F01 whenever `train_size_dict_map` is set (`training_size_analysis.py`) — F02–F07 are per-source plots that read keys (mix_tp uplets; for a `tp_wise`-only LLMmap cache, the `mix_tp_at_pred` batch-type key itself) absent from a loaded checkpoint and would `KeyError`. **Gate on the presence of `train_size_dict_map`, NOT on source count** (the old `len(all_summary_results) > 1` proxy crashed the single-source case). Label substrings are load-bearing: `"LLMmap"` → `load_train_size_dict` reads `results.json` + plots via the LLMmap dispatch branch; `"FLiPS"` → reads `train_size{N}_{item}.pkl`; `"Open-set"` → injects the `Unseen` label.
- **The main classification pipeline ALWAYS runs first** (`classify_batch.py` iterates `_load_or_run_train_size_item` before `plot_trainsize_wise_curves`), even for a merge-wrapper config: with no pre-existing checkpoint it re-classifies against the run's feature caches, then the wrapper *discards* that result and plots from `train_size_dict_map[0]`. Consequence for a zero-download run: the wrapper's own `token_pairs` / `force_class_size` / grid must be **valid on whatever run it points at** (it cannot reach absent data), even though they do not change the cache-driven F01 output.
- **`XP_configs/e3_llmmap_baseline/LLMmap_only_tp.yaml`** is the out-of-box E3 curve — a single-source merge-wrapper whose one `train_size_dict_map` entry points at the in-repo `XP_configs/e3_flips_vs_llmmap/llmmap_if_data/…/train_size_checkpoints` (label `"LLMmap-IF"`). It lives in its **own** config-dir so `run_experiments.py --config-dir e3_llmmap_baseline` globs exactly one YAML and never the data-gated 3-curve `e3_flips_vs_llmmap/FLIPS_mix_tp.yaml` (whose FLIPS arms point at the off-disk `FLiPS_ICML_run` and raise `FileNotFoundError`). Its main pipeline is pointed at the light subset (the 5 carved FLiPS-group pairs, `batch_prediction_sizes` capped at 4 so `unique_tp_in_mix` rides the `'max'` default, `max_nb_of_uplet:2`).
- **Gate-A** (`scripts/gates/gate_a.py`, stdlib-only, always-runs, zero-download): PART 1 runs the LLMmap-only variant and asserts an `F01_*.pdf`; PART 2 runs `Smoke_light` and asserts ≥1 PDF. `run_experiments.py` does **not** propagate the inner XP exit code (it only prints `Command failed…`), so the gate also scans the produced `xp_logs/.../xp_*_log_*.log` for `Traceback`/`FileNotFoundError`. `--llmmap-only` runs PART 1 only (for `make repro-llmmap-e3`).
- **Generated outputs under the committed subset must be cleaned.** A run against `FLiPS_ICML_light_subset` writes `Experiments/Batch_Classification_across_token_pairs/<xp_name>/` **and** a regenerable `Experiments/banned_token_pairs.json`; the root `.gitignore`’s `!Productions/FLiPS_ICML_light_subset/**` un-ignores them, so they show as untracked. Gate-A removes the xp dirs it created (snapshot-guarded; only inside the subset's generated-experiments dir) and the `banned_token_pairs.json` it created — leaving the committed fixture byte-clean and the gate idempotent. (`xp_logs/` is already git-ignored.) `gate_golden.py` reuses `make_light_subset.clean_generated_outputs()` (pre- and post-run) for the same byte-clean idempotency. `make smoke` is the one out-of-box target that deliberately *leaves* its output (so the user sees the PDF); discard it with `git clean`/`git checkout` or `make clean RUN_NAME=…` against a non-subset run.

## Reproduction wiring (top-level Makefile)

- **`make` is the tier-aware entry point** (`Makefile`, default goal `help`). Variables: `PYTHON ?= python`, `RUN_NAME ?= FLiPS_ICML_run` (the data-gated headline run), `SCENARIO ?= main`; config-dir vars (`CLOSEDSET_DIR=e1_closedset_headline`, `OPENSET_DIR=e2_openset_headline`, `COMPARISON_DIR=e3_flips_vs_llmmap`, `ABLATIONS_DIR?=Ablations`, `LLMMAP_DIR=e3_llmmap_baseline`, `SMOKE_DIR=Smoke_light`); `LIGHT_RUN:=FLiPS_ICML_light_subset` is hardwired (out-of-box targets bind to the committed subset, never `RUN_NAME`). No absolute paths; every `repro-*` runs `run_experiments.py … --mode inline`.
- **ENV requirement:** classifying targets need `audit_llm` importable — run inside the poetry `.venv` (or `make … PYTHON=.venv/bin/python`). A bare system `python` makes the XP child die with `ModuleNotFoundError: No module named 'audit_llm'` (the dispatcher adds `src/` to its *own* sys.path only, not the child's).
- **Three tiers, tagged in `make help`:** *out-of-box* (`gate-a`, `gate-golden`, `gate-b-parity`, `gate-b` (part b headline SKIPs), `repro-llmmap-e3` = `gate_a.py --llmmap-only`, `smoke`) run on the committed subset with zero download; *data-gated* (`repro-closedset`/`-openset`/`-comparison`/`-ablations`) need the uploaded headline run; *Mode-B* lives in docs. `fetch`/`fetch-check`/`fetch-headline` wrap `fetch_data.py`.
- **Data-gated SKIP contract** (`$(call repro_gated,<dir>,<suffix>)`): the guard checks `Productions/$(RUN_NAME)/run_config.{json,pickle}` and, when absent, prints a loud "needs the user-uploaded FLiPS_ICML_run … SKIPPING" and exits 0 — it never crashes with `FileNotFoundError`. Guard + run are one backslash-joined shell line so the SKIP short-circuits the run (each Make recipe line is otherwise its own shell). The `repro_gated` define begins with `@`, so it **cannot be nested** inside another shell `if/else` (e.g. `repro-ablations`, which also checks for the config dir) — those targets inline the guard.
- **`gate-golden`** (`scripts/gates/gate_golden.py`) re-runs the shipped `Smoke_light` config (RUN_NAME=`FLiPS_ICML_light_subset`, xp-suffix=`smoke`) and asserts three LAYERED tolerances vs `golden_refs.json`: (1) label set EXACT, (2) feature-cache parity — md5 within the pinned env, `np.allclose` (rtol 1e-5) behind `--reference-features <dir>` cross-env, (3) accuracy band ±2 pp `tp_wise` / ±5 pp `mix_tp`. It imports the *capture* extractors from `make_light_subset.py` (`_collect_label_set`/`_collect_smoke_accuracy`/`_md5`) so the verify path reads checkpoints byte-identically to how `golden_refs.json` was captured — divergence would make the gate meaningless.
- **`gate-b`** (`scripts/gates/gate_b.py`; part (a) standalone = `make gate-b-parity` → `scripts/gates/gate_b_parity.py`) is the two-part gate the legacy-path deletion gates on. **Part (a) — data-free class-parity firewall (always runs):** the enumerator label set (`build_instances(load_scenario("main"))`) MUST exactly equal the legacy path's, computed from config alone (NO `main_dataset`, NO archive) — base models + `@@` quant aliases from the `config/models.yaml` registry, abliterated = the legacy run config's `abliterated_models` ∩ the registry's data-backed `abliterated_variants` (so the config-listed-but-data-absent 6th repo is excluded via the SSOT, landing on 200+32+5=237), variation keys via the **legacy** `compute_model_variations_indices` fed a synthetic config-derived 1-row-per-cell DataFrame. **EXACT set equality** is the firewall (count + strings); order is NOT asserted (sklearn sorts `classes_`, so the set fixes the confusion-matrix axes, consistent with the golden label-set check). **Part (b) — coarse-banded accuracy sanity (read-only, never recomputes):** the in-repo LLMmap `results.json` nested `ds_wise/<bs>/no_token_pairs/llmmap_clf/accuracy_mean` (~0.33, NOT a top-level 96%) is read both raw (`ds_wise`) and via `load_train_size_dict` (in-memory `tp_wise`) and band-checked (±5 pp; bs6≈0.3337, bs4≈0.3231); the **headline** accuracy check is DATA-GATED on `Productions/$(RUN_NAME)/run_config.{json,pickle}` and prints a **loud SKIP-not-pass** when the off-disk `FLiPS_ICML_run` is absent (it never falsely passes on the light slice). The bands are wide on purpose (unseeded `random.sample` in `token_pair_mixing.py` ⇒ ~±3.4 pp run-to-run; exact regression is `gate-golden`'s job). Final stdout token **`GATE-B-RESULT: PASS | SKIP | FAIL`** is what the manual headline verification tier reads. Needs `audit_llm` importable (the `.venv`; a `src/` sys.path fallback also works).
- **`figures`** finds `micro_pr_curve_cache.pkl` / `roc_figscore_cache.pkl` anywhere under `Productions/` and re-plots via `preview_micro_pr_curve_cached.py` / `preview_openset_roc.py`; with no cache present it prints a hint and exits 0. **`clean`** removes generated xp dirs / Figures / `*_cache.pkl` under `Productions/$(RUN_NAME)` + `tmp/*_preview` + `xp_logs/`, and **refuses** when `RUN_NAME==FLiPS_ICML_light_subset` to protect the committed fixture.

## Class balancing (`force_class_size`)

- **`force_class_size` is a per-class cap with two meanings depending on the path** — same YAML key, different op. In single-token-pair classification (`balance()`, `single_classification.py`) it forces each class to *exactly* N (oversampling short classes *with replacement* when an explicit `int < class size`); in multi/open-set (`_reducing_token_pair`, `multi_classification.py`) it truncates each class to *at most* N (slice `[:N]`, never oversamples). Both are applied **post-split, train-fold only** (or, for the multi pre-split pool cap in `_prepare_data`, by *down*-truncation that creates no duplicates), so no training row ever leaks into a test fold.
- **Default is `"auto"`, resolved lazily to the minimum class count** of the already-cleaned labels at the point of use (`_resolve_force_class_size` in multi; the `=="auto"` branch in single's `balance()`). `"auto"` undersamples only ⇒ no duplication, no leakage, and adapts to the data — so classification configs no longer hardcode it. An explicit `int` remains available to cap classes *below* the data size (e.g. for speed) or to fix the value regardless of the data.
- **Fresh-run / fixture configs use `"auto"`; configs that reuse existing data pin `force_class_size: 500`.** Smoke and `LLMmap_only` (zero-download fixtures) omit it — on the subset every `(model,temp,sp)` cell has ~500 rows so `"auto"` resolves to ~500, and `gate-golden` confirms the smoke stays within band. The reproduction configs that re-read the existing `FLiPS_ICML_run` — both headline `_full` runs and the `e3_flips_vs_llmmap` wrapper — **pin `500`** so they match the data/checkpoints they were built against (the open-set score fingerprint `_upstream_meta` hard-gates on `force_class_size`; a change invalidates the pickles ⇒ multi-GB rebuild). Rule of thumb: `"auto"` for fresh runs, an explicit int when reusing prior checkpoints.
- **The `force_class_size > test_size` invariant is enforced at config-load** by `ClassificationConfig._validate_force_class_size` for explicit ints (and at resolution time for `"auto"`), replacing the old cryptic `StratifiedShuffleSplit` `ValueError`. It also requires `test_size` to be a positive integer per-class count when `force_class_size` is set.

## Config schema defaults (`experiment_config_schema.py`)

- **A schema default of `None` is *not* the same as an unset field.** `load_experiment_config`
  returns `config.model_dump(exclude_none=False)`, so every field — including those left out of the
  YAML — appears in the dict with its declared default. For a field whose consumer reads it as
  `config.get("key", FALLBACK)`, a schema default of `None` means the key is *present and `None`*, so
  `.get` returns `None` (not `FALLBACK`) and the in-code fallback is dead. Concretely, omitting
  `splitter_type` / `normalization_methods` used to crash (`SPLITTER_MAP[None]` / `None.get(...)`),
  which is why every shipped config set them. **Rule:** to make such a field genuinely optional, give
  it a concrete schema default that matches the consumer's intended fallback — do not rely on the
  `config.get(key, FALLBACK)` second arg. Current concrete defaults: `classifiers=["XGBoost"]`,
  `splitter_type="StratifiedShuffleSplit"`, `default_normalization="auto"`,
  `normalization_methods={seq_length: none}`.
- **`batch_types` is *derived*, not a static default.** `ClassificationConfig._derive_batch_types`
  (model_validator) fills it from `batch_prediction_sizes` when unset: size `1` → `tp_wise`, any size
  `> 1` → `mix_tp_at_pred`. An explicit `batch_types` is always respected. Existing configs that set
  it stay byte-identical (golden-safe).

## Label string stability

Three label string forms are **load-bearing** — they are parsed byte-for-byte by `xp_tools.model_filtering` (`full_var_model_name_to_original_model_name` / `full_var_model_name_to_var_name`) and by downstream analysis. **Never change these conventions without updating all consumers in the same commit.**

| Form | Example | Rule |
|---|---|---|
| `@@<quant_key>` | `Qwen/Qwen2-7B-Instruct@@fp8` | `@@` = `QUANTIZATION_SEPARATOR` from `audit_llm.file_io`; suffix is the quant *key* (e.g. `bitsandbytes_int4`), NOT the vLLM engine kwarg value (`bitsandbytes`) |
| `{abliterated_repo}_ablit` | `failspy/Meta-Llama-3-8B-Instruct-abliterated-v3_ablit` | `{abliterated_repo}` is the *abliterated* HF id, not the base; **no** `temp-x_sp-y` suffix appended |
| `temp-{t}_sp-{sp}` | `temp-1.0_sp--1` | `sp--1` for `sp=-1`; trailing `.0` in temperature is preserved; variation suffix appended to `storage_name` for base/quant labels |

These conventions are enforced by `tests/test_variation_resolver.py` (byte-identity assertions). `scenarios.resolver` and `scenarios.variation` are the canonical source; do not re-declare the maps or regenerate labels elsewhere.

## Leak-grep allowlist

A repo-wide leak grep (pattern `/home/` | `/lustre` | `idris` | `jean.zay` | `jean_zay` | `\bJZ\b` | `wcx@` | `hf_token` | `api_key` | `password` over `*.py` `*.yaml` `*.yml` `*.sh` `*.bash` `*.slurm` `*.ini` `*.toml` `*.ipynb` `*.log` `*.json`, excluding scaffolding directories excluded from the public build and `config/models.hpc.example.yaml`) is run as part of the pre-release checklist. The following remaining matches are benign and intentional — none is a path, account, or secret leak.

| File:line | Match | Why it is safe |
|---|---|---|
| Productions/Intersection_vocab/Intersection_of_34_models.json:7176 | password | the literal token " password" inside a tokenizer-vocabulary intersection (an import-time dependency); model vocabulary data, not a credential |
| Productions/Intersection_vocab/Intersection_of_34_models.json:14387 | password | same file, the token "password"; tokenizer vocabulary, not a credential |
| scripts/make_light_subset.py (`scrub_run_config` needle tuple) | /lustre, /home/, wcx@ | the literals are the *defensive scrub allow-list* — the carver fails loudly if any of these survives in the run_config it writes; they enable leak-removal, they are not a leak |

`config/models.hpc.example.yaml` is excluded by design: it is the documented place for cluster filesystem paths and carries a "# DO NOT commit with real paths" banner.

## Third-party components

| Component | Repository | SPDX License | Copyright | Intake note |
|---|---|---|---|---|
| LLMmap | [github.com/pasquini-dario/LLMmap](https://github.com/pasquini-dario/LLMmap) | MIT | Copyright (c) 2024 pasquini-dario | The adapter code, cached inference results, and clone recipe are all permissible under the MIT license. |

## Licensing / REUSE compliance

- **`REUSE.toml` is the single source of truth for SPDX metadata** (REUSE spec 3.x). Rather than
  inject a header into every one of the ~380 files (many are binary `.parquet`/`.npy`/`.npz`/`.pkl`
  or comment-less JSON), licensing is declared with `[[annotations]]` glob blocks. `python -m reuse
  lint` must report **compliant** (a required pre-release check); every file needs BOTH an
  `SPDX-FileCopyrightText` and an `SPDX-License-Identifier` — a license-only annotation fails.
- **Buckets:** first-party code / config / CI / docs → **MIT** (PEReN); datasets and derived
  data/results → **CC-BY-4.0** (PEReN); generated lockfiles → **CC0-1.0**; the vendored
  `src/Fingerprinting_methods/LLMmap/**` subtree → **MIT, `2024 pasquini-dario`** (upstream, see the
  Third-party table above — *not* blanket PEReN). The CC-BY-4.0 data tag does **not** override the
  stricter per-model OUTPUT terms (Cohere = CC-BY-NC, Gemma = ToU; see `data/LICENSE` and
  `docs/provenance.md`, Decision 6).
- **Block order matters:** when several globs match a path the **last** matching block wins, so the
  data block sits after the code block (e.g. `XP_configs/**/llmmap_if_data/**` resolves CC-BY-4.0,
  not the broader `XP_configs/**` MIT), and the LLMmap subtree uses `precedence = "override"` to keep
  pure upstream attribution.
- **`reuse lint` parses *every* file as text** — including markdown — so prose that spells out an
  SPDX license-identifier tag with a value is misread as an *invalid SPDX expression*. Scaffolding
  directories excluded from the public build are covered with `precedence = "override"` so reuse
  disregards their inline (prose) tags. Pre-existing per-file headers (some dated `2024`,
  `contact@peren.gouv.fr`) take precedence over the `REUSE.toml` glob for those individual
  files (REUSE "closest" rule) and are left as-is — the `2024`/`2026` coexistence is cosmetic and
  lint-clean.
