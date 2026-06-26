# Mode A — reproduction without a GPU

Mode A is the **no-GPU** reproduction path. It does **not** re-run inference; it consumes
already-generated traces and re-runs the *analysis* (feature extraction, classification, plotting).

Mode A is **three honestly-labelled tiers**, not a single "download-and-get-the-paper-numbers" button.
Read the tier you can actually run, and check the provenance tag on every command before you trust an
output as a paper number.

| Tier | What it reproduces | Prerequisite | Cost |
|---|---|---|---|
| **1. out-of-box** | LLMmap E3 baseline curve + figure styling, on a **12-model FLIPS dev slice** (grid temps `0.4/0.7/1.0`, **not** the paper 8-temperature grid) | nothing — committed in the repo | seconds–minutes, CPU, zero download |
| **2. data-gated headline** | the headline **96% / 90%** closed-/open-set numbers + the FLIPS E3 arms | the user-uploaded off-disk `FLiPS_ICML_run` (HPC export → Zenodo) | a download + minutes–hours, CPU |
| **3. Mode B** | the data itself, from scratch | a GPU box | see [`mode-b.md`](mode-b.md) |

> **Beyond the canned experiments**, the same analysis runs on **your own scenario** — your own
> subset of the generated variations. See [`custom-scenario.md`](custom-scenario.md) (it ships a
> runnable, zero-download example).

> **Provenance tags** used throughout this doc:
> `out-of-box` · `data-gated` · `Mode-B-only` · `synthetic-preview` · `guessed-mapping (verify)`.
> A `synthetic-preview` command plots **fabricated** numbers (layout/styling preview only) — it
> reproduces **nothing**. A `guessed-mapping (verify)` row is a paper-figure↔script correspondence we
> could not confirm from the repo alone; treat it as a lead, not a fact.

Environment note (applies to every tier): the classification targets need `audit_llm` importable. Run
inside the project poetry env, or pass an interpreter that has it, e.g.
`make smoke PYTHON=.venv/bin/python`. A bare system `python` makes the experiment child process die
with `ModuleNotFoundError: No module named 'audit_llm'`.

> **Run from a git clone (not a packaged install).** Path resolution
> (`audit_llm.path_utils.get_repository_level_path`) walks parent directories looking for a `.git`
> directory to find the repo root (and thus `Productions/`, `XP_configs/`, `config/`). A
> non-editable `pip install` into `site-packages` has **no `.git`**, so the walk falls through to
> the filesystem root and paths break. Reproduce from a cloned working tree (or an editable
> `pip install -e .` inside the clone).

---

## Tier 1 — out-of-box (zero download, CPU)

Everything here runs against the committed in-repo light subset
(`Productions/FLiPS_ICML_light_subset/`, a 12-model FLIPS dev slice) and the shipped LLMmap cache
(`XP_configs/e3_flips_vs_llmmap/llmmap_if_data/`). No upload, no download, no GPU.

```bash
make gate-a            # always-runs gate: LLMmap-only F01 E3 replot + light-subset smoke (>=1 PDF each)
make repro-llmmap-e3   # just the LLMmap-only F01 E3 curve (the out-of-box arm of E3)
make smoke             # zero-download CPU smoke classification on the light subset, emits >=1 PDF
make gate-golden       # golden regression: re-run smoke, assert results vs golden_refs.json
```

What you actually get:
- `make gate-a` / `make repro-llmmap-e3` regenerate the **LLMmap baseline** E3 accuracy-vs-queries
  curve (`F01_*.pdf`) from the in-repo `llmmap_if_data/` cache — no `FileNotFoundError`, by design of
  the LLMmap-only F01 variant (`XP_configs/e3_llmmap_baseline/LLMmap_only_tp.yaml`).
- `make smoke` re-classifies the 5 carved FLIPS-group token-pairs of the dev slice and emits
  `F01/F03/F04/F06` PDFs.
- `make gate-golden` re-runs that smoke classification and asserts, against
  `Productions/FLiPS_ICML_light_subset/golden_refs.json`: (1) the enumerator label set **exactly**,
  (2) feature-cache parity (md5 within the pinned env, `np.allclose` cross-env), (3) accuracy within a
  **coarse band** (±5 pp `mix_tp` / ±2 pp `tp_wise`). The band is wide on purpose — `n_splits=2` plus
  the unseeded `random.sample` in `token_pair_mixing.py` give ~±3.4 pp run-to-run.

This tier reproduces the **LLMmap baseline + the figure-generation machinery on a dev slice**. It does
**not** reproduce the paper headline numbers — those need Tier 2.

## Tier 2 — data-gated headline (needs the uploaded run)

The headline **96% (E1 closed-set) / 90% (E2 open-set)** numbers and the FLIPS arms of E3 live in the
off-disk `FLiPS_ICML_run` (the HPC export). Until you fetch it, these targets **SKIP loudly (exit 0)** —
they never crash.

```bash
make fetch-headline    # download the off-disk FLiPS_ICML_run (Zenodo / your HPC export) — see data/README.md
make repro-closedset   # E1 closed-set headline (96%)
make repro-openset     # E2 open-set headline (90%)
make repro-comparison  # 3-curve E3: FLiPS closed/open vs LLMmap (the FLIPS arms)
make repro-ablations   # ablation sweep (configs ship with the headline; SKIPs if XP_configs/Ablations/ absent)
```

Each data-gated target checks for `Productions/$(RUN_NAME)/run_config.{json,pickle}` (the flattened
headline run). If absent it prints a `needs the user-uploaded FLiPS_ICML_run … SKIPPING` message and
exits 0. Override the run name with `make repro-closedset RUN_NAME=my_run`.

How to upload + wire the headline archive (DOI is a placeholder today) is documented in
[`../../data/README.md`](../../data/README.md).

## Tier 3 — Mode B (from scratch, GPU)

To regenerate the traces themselves (and anything `Mode-B-only`, e.g. log-prob diagnostics), follow
the GPU path in [`mode-b.md`](mode-b.md): vLLM inference → parse + merge → then re-enter Mode A.

---

## Table 1 — experiment → command

| Exp | What | Command | Tag |
|---|---|---|---|
| **E0** | DCA showcase bar chart | regenerate via [`mode-b.md`](mode-b.md); figure emitted by the analysis pipeline | `data-gated` |
| **E1** | closed-set classification (headline 96%) | `make repro-closedset` | `data-gated` |
| **E2** | open-set classification (headline 90%) | `make repro-openset` | `data-gated` |
| **E3** | LLMmap arm (accuracy vs N_t) | `make repro-llmmap-e3` | `out-of-box` (LLMmap-only F01 variant) |
| **E3** | FLiPS arms (FLiPS-closed/open vs LLMmap, 3-curve) | `make repro-comparison` | `data-gated`, `guessed-mapping (verify)` (the 3-curve F01 mapping is config-named, not paper-figure-confirmed) |
| **D1** | log-prob diagnostics | — (regenerate via [`mode-b.md`](mode-b.md)) | `Mode-B-only` (logprobs are **not on disk** in any archive tier) |
| **D2** | diagnostic / robustness | `guessed-mapping (verify)` — no confirmed repo target | `guessed-mapping (verify)` |
| **D3** | diagnostic / robustness | `guessed-mapping (verify)` — no confirmed repo target | `guessed-mapping (verify)` |
| **A1–A9** | ablation sweep | `make repro-ablations` (ships with headline) | `data-gated`, `guessed-mapping (verify)` (per-ablation figure mapping unconfirmed) |

## Table 2 — fast figure re-render from a cached score file

Figures are produced by the analysis pipeline during a real run (see *"Where the in-experiment figures
land"* below). Two helper scripts re-render a figure quickly from a small cached score file, so you can
redraw without repeating the multi-GB analysis — they call the **same** rendering code the live run
uses, so they cannot diverge from a real run's output.

| Script | Draws | Command | Data source | Tag |
|---|---|---|---|---|
| `preview_micro_pr_curve_cached.py` | closed-set micro P/R-vs-confidence curves | `… --cache <…>/micro_pr_curve_cache.pkl [--out <ModelWiseTables dir>]` | **REAL** (from `micro_pr_curve_cache.pkl`) | `out-of-box` if cache present (built by a run) |
| `preview_openset_roc.py` | open-set ROC/PR + α-rule figures | `… --cache <…>/roc_figscore_cache.pkl [--out <dir>]` | **REAL** (from `roc_figscore_cache.pkl`) | `out-of-box` if cache present |

`make figures` automates these two: it scans `Productions/` for `micro_pr_curve_cache.pkl`
and `roc_figscore_cache.pkl` (produced by a run when the corresponding cache flag is set, or shipped in
a fetched archive) and replots from each. With no caches present it prints a "run `make smoke` or a
`make repro-*` first" hint and does nothing.

> Paper-figure ↔ preview-script correspondence is **not** asserted here. The cached scripts redraw the
> *same* figures the live run produces (so they cannot diverge from a real run's output). Mapping a
> specific paper figure number (e.g. "Figure 4") onto a specific script is `guessed-mapping (verify)` —
> confirm against the paper before citing.

---

## Where the in-experiment figures land

A real run (`make smoke`, `make repro-*`, or `make gate-a`) writes its PDFs under the run's Experiments
tree, **not** via the `preview_*` scripts:

```
Productions/<run>/Experiments/Batch_Classification_across_token_pairs/<xp_name>/
    <calc_item>/<clf>/<train_size>/
        ModelWiseCurves/<metric>/F01_accuracy_vs_queries_tr<ts>.pdf   # F01 (+ the 3-curve overlay from e3_flips_vs_llmmap)
                                  F03_all_groups_tr<ts>.pdf
                                  F04_histogram_tr<ts>.pdf            # F04 token-pair histogram
                                  F06_best_mix_tr<ts>.pdf
                                  F02_mix_pred_utp*  F05_mix_train_utp*  F07_mix_vs_tpwise*
        ModelWiseTables/<effective_key>/                              # tables + the cached *_cache.pkl
```

- `<xp_name>` = `<config-stem>_<xp-suffix>` (e.g. `LLMmap_only_tp_smoke`); `<calc_item>` is e.g.
  `mix_tp_at_pred` or `tp_wise`; `<clf>` e.g. `llmmap_clf`; `<train_size>` e.g. `40`; `<metric>` e.g.
  `accuracy`.
- **F01 variants** (`e3_llmmap_baseline` and the 3-curve `e3_flips_vs_llmmap`) set `train_size_dict_map`,
  which puts the run in *merged-wrapper* mode: **only `F01_*.pdf` is emitted** (F02–F07 read keys absent
  from a loaded checkpoint). The 3-curve overlay PDF is the `F01_accuracy_vs_queries_*` from
  `e3_flips_vs_llmmap` once headline data is present.
- `make smoke` (no `train_size_dict_map`) emits `F01/F03/F04/F06`.

---

## Reproduction-fidelity caveat (per tier)

- **Tier 1 (out-of-box):** reproduces the **LLMmap baseline curve + the figure-generation styling on a
  12-model dev slice** (grid temps `0.4/0.7/1.0`, not the paper 8-temperature grid). It is a wiring/
  regression check, **not** the paper headline. `gate-golden` allows ±5 pp (`mix_tp`) / ±2 pp
  (`tp_wise`) because `n_splits=2` + the unseeded `random.sample` in `token_pair_mixing.py` give
  ~±3.4 pp run-to-run.
- **Tier 2 (data-gated):** reproduces the **headline 96%/90% numbers from the uploaded
  `FLiPS_ICML_run`** — i.e. it re-derives the paper numbers from the released traces. Fidelity is bound
  to that uploaded run's contents.
- **Tier 3 (Mode B):** reproduces the **methodology**, not bit-identical traces: inference at `temp > 0`
  is non-deterministic, so regenerated traces differ run-to-run even at a fixed seed.

The analysis classifier seed is **42**; each run snapshots its environment to
`Productions/<run>/run_config.json` (the leak-scrubbed copy; `run_config.pickle` alongside) — consult it
to see exactly which grid, models, and versions produced a given set of figures.

---

## Quick check

Copy-paste, fully offline, CPU-only (Tier 1). Run inside the project's poetry env, or pass
`PYTHON=.venv/bin/python` so the classification child process can `import audit_llm` (a bare system
`python` dies with `ModuleNotFoundError: No module named 'audit_llm'`):

```bash
make gate-a       PYTHON=.venv/bin/python   # LLMmap E3 replot + light-subset smoke; expect "gate-A: PASS (2/2 parts passed)"
make smoke        PYTHON=.venv/bin/python   # zero-download CPU smoke classification; writes the F0* PDFs below
make gate-golden  PYTHON=.venv/bin/python   # asserts label set + feature parity + accuracy band vs golden_refs.json; "PASS (3/3 layers)"
make fetch-check  PYTHON=.venv/bin/python   # offline manifest verify of the committed subset (21 pass, 1 external, 0 fail)
```

**Output PDFs** (all under `Productions/FLiPS_ICML_light_subset/Experiments/Batch_Classification_across_token_pairs/`):

- `make gate-a` PART 1 — the **LLMmap-only E3 curve** replotted from the in-repo `llmmap_if_data/` cache:
  `LLMmap_only_tp_gate_a/ClosedSet/all/XGBoost/40/ModelWiseCurves/accuracy/F01_accuracy_vs_queries_tr40.pdf`
- `make smoke` — the **dev-slice smoke figures** (F01 accuracy curve + F03/F04/F06 and the micro-PR/ROC
  tables): `Smoke_light_smoke/ClosedSet/all/XGBoost/40/ModelWiseCurves/accuracy/F01_accuracy_vs_queries_tr40.pdf`
  (plus `F03_all_groups_tr40.pdf`, `F04_histogram_tr40.pdf`, `F06_best_mix_tr40.pdf`, and the
  `ModelWiseTables/{mix_tp_at_pred_utp2,tp_wise}/FLiPS_2_splits_micro_pr_curve_bs_*.pdf` curves).

`gate-a` and `gate-golden` **self-clean** their generated experiment dirs so the committed fixture stays
byte-identical; `make smoke` deliberately **leaves** its `Smoke_light_smoke/` output so you can open the
PDFs — discard it afterwards with `git clean -fd Productions/FLiPS_ICML_light_subset/` (or just re-run
`make gate-golden`, which removes it).

The data-gated headline targets will SKIP-loud until you run `make fetch-headline`. The GPU
from-scratch path is [`mode-b.md`](mode-b.md).
