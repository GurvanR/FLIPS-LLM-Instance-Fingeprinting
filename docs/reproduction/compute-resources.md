# Compute resources — Mode-A headline reproduction

Measured cost of the **data-gated** headline experiments (E1 closed-set, E2 open-set, E3 comparison)
on the off-disk `FLiPS_ICML_run`. All are **CPU-only** re-classification of already-generated traces
(no GPU, no inference). Numbers below are read directly from the run logs (`TIME SPENT`, `RSS=…`
markers) on a mid-size HPC node (smaller than a top-tier national cluster), not a profiler — treat them as solid ballparks.

> **Why this matters.** A single experiment is a multi-hour, ~16–20 GB-RAM job that writes **tens of GB**
> under `Productions/<run>/Experiments/`. This is why the headline tier does not run on a typical 16 GB
> laptop (the analysis process alone peaks at ~13 GB resident) and is documented as HPC-class.

## Summary

| Experiment | Wall-clock | Peak RAM (process) | CPU | Disk under `Experiments/` |
|---|---|---|---|---|
| **E1 closed-set** | **6 h 08 m** (367.9 min) | **~13.3 GB** (≈15–18 GB incl. workers) | 8 feature workers; XGBoost uses all cores (node ≥ 10 logical CPUs) | **44 GB** (38 GB per-experiment `probas_sidecars` + ~6 GB shared cache/misc) |
| **E2 open-set** | _pending — fill after the run_ | _pending_ | _pending_ | **~+40 GB again** (own `probas_sidecars`) |
| **E3 comparison** | _pending (lighter)_ | _pending_ | _pending_ | **small** — merge-wrapper: plots from existing E1/E2 checkpoints + committed LLMmap cache (feature cache is a `CACHE HIT`); verify at runtime |

> **Disk: E1 + E2 dominate; E3 is light.** The big cost (`probas_sidecars`, 38 of E1's 44 GB) is
> **per-experiment**, written by the two full classifications (E1, E2). E3 is a merge-wrapper that
> re-plots the F01 figure from the E1/E2 checkpoints (features `CACHE HIT`), so it adds little. Plan for
> **~100–130 GB free** (E1 ~44 + E2 ~40 + ~6 GB shared cache + E3 light). Only the ~6 GB feature cache is
> reused across all three. See the Disk section below.

**Node:** a mid-size HPC node (smaller than a top-tier national cluster such as Jean Zay). Exact logical-core count is not in the
log; the default feature-worker pool `min(8, cpu−2) = 8` implies **≥ 10 logical CPUs**.

> **LLMmap baseline (E3 comparison arm) is *not* in these numbers.** FLIPS classifies with **XGBoost**;
> the LLMmap baseline is a trained **neural-net** (PyTorch Lightning), so regenerating it from scratch is
> ~**10–20×** the FLIPS XGBoost analysis runtime above. That is why it is **not** recomputed here — the
> repo uses the committed `llmmap_if_data/` cache, and the full pre-trained `LLMmap_ICML_run` is hosted on
> Zenodo (`scripts/fetch_data.py`) rather than retrained. See README §(a).

---

## E1 — closed-set (the 96 % headline)

**Outcome:** reproduced. `mix_tp`, `ts=40`, XGBoost, averaged over 30 token-pairs:
bs=2 → 0.8552, bs=4 → 0.9247, bs=6 → 0.9418, **bs=8 → 0.9629 ± 0.0118** (the ~96 % E1 number).
Per-token-pair (`tp_wise`) arm: bs=1 → 0.6703 … bs=8 → 0.8708. Log ends `XP … done successfully.`

### Time — 367.9 min total

| Phase | Window | Duration | Work |
|---|---|---|---|
| Feature compute / load | 17:41 → 19:27 | **105.9 min** (logged) | 31 token-pairs; 17 (re)computed (564 per-model feature computes ≈ 2–2.7 s each) + load/assemble all cached features |
| Classification — `mix_tp` | 19:27 → 22:38 | ~3 h 11 m | XGBoost, 5 splits, batch sizes 1–8, token-pair mixing — **dominant cost** |
| Classification — `tp_wise` | 22:38 → 23:27 | ~49 m | per-token-pair XGBoost |
| Tables + figures | 23:27 → 01:35 | ~2 h 08 m | P/R curves, FMR/FNMR, histograms, PDFs, fig-caches (mostly single-threaded) |

> **Cache-warm caveat.** This run resumed a prior partial attempt: **14 of 31** token-pairs were a feature
> `CACHE HIT`, only 17 recomputed. A **cold start** (no cache) adds roughly **+30–60 min** to the feature
> phase → ~6.5–7 h total.

### Memory — peak ~13.3 GB
- Analysis process RSS entered the feature phase at **6.1 GB** (the loaded `Answers` + `TokenIDs`
  dataframes) and peaked at **13.3 GB** by the end of feature compute.
- Each of the 8 feature-compute workers: ~0.5–0.6 GB. Parent + workers overlap → realistic system
  peak **~15–18 GB**. **Plan for ~16–20 GB free.**

### CPU
- **Feature phase:** `ProcessPoolExecutor` with **8 workers** — but largely I/O / assembly-bound (only
  ~24 CPU-min of real compute spread over 106 min wall), so cores were only partly busy here.
- **Classification:** XGBoost runs with **no `n_jobs`** → grabs **every logical core** for ~4 h. This is
  where the CPU budget goes. Rough total: **~45–70 CPU-hours**, almost all XGBoost.
- Knobs: `FLIPS_FEATURE_WORKERS` caps the feature pool; `OMP_NUM_THREADS` caps XGBoost.

### Disk — 44 GB under `Experiments/` after E1 (and it scales ~linearly)
Measured split: **38 GB of the 44 GB is `probas_sidecars`**, the rest (~6 GB) is the shared feature
cache plus checkpoints/tables/PDFs/fig-caches.

| Path | What | Scope | Reused by E2/E3? |
|---|---|---|---|
| `…/<xp_name>/probas_sidecars/<batch_type>/<cell_id>.npz` | spilled `full_probas` / `full_y_true` (full per-split prediction-probability matrices + true labels) | **per-experiment** | **no** — each experiment writes its own |
| `feature_computation_data/` (per-`{token_pair}/{model}` `.npy`/`.npz`) | NIST/SmallNist features | **shared, run-level** | **yes** — `CACHE HIT`, computed once |
| `Batch_Classification_across_token_pairs/<xp_name>/` (checkpoints, tables, PDFs, fig-caches) | classification outputs | per-experiment | no |

`probas_sidecars` lives under the per-experiment `fig_save_path`
(`multi_classification.py:_probas_sidecar_dir` → `Experiments/<experiment_fun>/<xp_name>/probas_sidecars/`),
**not** the shared cache. It is the dominant disk consumer **and** it is per-experiment — so **E2 and E3
each write their own ~40 GB**. Total disk grows **roughly linearly**: budget **~120–150 GB free** for all
three. (Only the ~6 GB feature cache is shared.)

**Pruning (optional).** The sidecars are spilled to keep prediction probabilities out of RAM and are
consumed during table/figure generation; `make figures` re-plots from the separate `*_cache.pkl`, not
these `.npz`. So once an experiment has finished and its figures/`gate-b` are done, its `probas_sidecars/`
is a candidate for deletion if disk-constrained — **verify nothing you still need reloads them before
deleting** (e.g. a later re-plot that bypasses the fig-cache).

To record the split on the node:
```bash
du -sh Productions/FLiPS_ICML_run/Experiments                                   # total (≈44 GB after E1)
du -sh Productions/FLiPS_ICML_run/Experiments/feature_computation_data          # shared cache (~6 GB)
du -sh Productions/FLiPS_ICML_run/Experiments/*/*/probas_sidecars               # per-experiment (~38 GB)
```

---

## E2 — open-set (the 90 % headline)

_Pending — run `make repro-openset` (open-set adds threshold/alpha sweeps on top of E1's pipeline, so
expect similar-or-higher time and the same ~16–20 GB RAM; disk = increment only, feature cache reused).
Fill the Summary row and this section from the run log (`TIME SPENT`, `RSS=…`) once it completes._

---

## How to read the numbers off a log
```bash
grep 'TIME SPENT'        xp_logs/.../<run>.log     # total wall-clock (minutes)
grep -E 'START|DONE in'  xp_logs/.../<run>.log     # feature-phase boundaries + RSS
grep -oE 'RSS=[0-9]+MB'  xp_logs/.../<run>.log | grep -oE '[0-9]+' | sort -n | tail -1   # peak RAM
grep 'accuracy (avg over' xp_logs/.../<run>.log    # per-bs accuracy
```
