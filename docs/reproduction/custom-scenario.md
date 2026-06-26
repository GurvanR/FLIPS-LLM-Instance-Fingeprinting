# Run your own scenario

The reproduction targets (E1/E2/E3) are not the only thing this repo can run. The same pipeline
runs **your own scenario** — your own set of model × variation instances — end to end, from
inference through classification to figures. This page is the **map**; each step links to the
reference page that already documents it.

## The two layers (read this first)

"Variations" live in **two deliberately-separate layers**:

1. **Inference generates** the data. *Which* cells exist — every `(model, temperature,
   system_prompt_idx, …)` combination — is decided when you generate, by the dataset CSV and the
   inference run. See [`../codebase/dataset.md`](../codebase/dataset.md) and
   [`../codebase/inference.md`](../codebase/inference.md).
2. **A scenario selects** which of those generated cells become classification **classes**. A
   scenario is a small YAML (`config/scenarios/<name>.yaml`) validated against the model registry
   [`config/models.yaml`](../../config/models.yaml). See
   [`../codebase/scenarios.md`](../codebase/scenarios.md) and the field contract in
   [`../../config/scenarios/README.md`](../../config/scenarios/README.md).

The single rule that ties them together:

> **A scenario can only select cells that inference actually generated.** Select a cell with no
> data → an empty/degenerate class. So pick your path by asking: *do the variations I want already
> exist on disk?*

- **Yes** → you only need a scenario. Go to **Path A**.
- **No** (you want new temperatures / system prompts / models) → generate them first. Go to
  **Path B**, then come back to Path A.

---

## Path A — select your own subset (no GPU)

Use this when the variations you want already exist in a run on disk — the committed light subset,
a [fetched headline run](mode-a.md), or a run you generated via Path B.

1. **Write a scenario** `config/scenarios/<name>.yaml` listing your base models and a variation
   grid. Field rules (axes, *cartesian-within-a-group / union-across-groups*, validation, the CSV
   invariant): [`../../config/scenarios/README.md`](../../config/scenarios/README.md). Every base
   model must exist in [`config/models.yaml`](../../config/models.yaml).
2. **Point an experiment config at it** with the `scenario:` key instead of an inline
   `model_variations` grid — see the field reference in
   [`../codebase/configuration-advanced.md`](../codebase/configuration-advanced.md#variation-selection-scenario-vs-model_variations).
3. **Run the pipeline**, then re-plot:
   ```bash
   python XP_configs/run_experiments.py --config-dir <your_dir> \
       --run-name <run> --xp-suffix <name> --mode inline
   make figures        # redraw from the caches the run wrote
   ```

### A runnable example ships with the repo

[`config/scenarios/example_custom.yaml`](../../config/scenarios/example_custom.yaml) +
[`XP_configs/example_custom/example_custom.yaml`](../../XP_configs/example_custom/example_custom.yaml)
are a minimal, copy-and-edit template. The scenario selects a 4-class subset (2 base models × 2
temperatures) from the **committed light subset**, so it runs on CPU with zero download:

```bash
make example-scenario        # or the raw run_experiments.py command above
make figures
```

It writes its figures and a `micro_pr_curve_cache.pkl` under
`Productions/FLiPS_ICML_light_subset/Experiments/…/example_custom_example/`. Copy the two files,
change the model list / variation grid, and you are running your own scenario.

---

## Path B — generate your own variations first (GPU)

Use this when you need cells that *don't* exist yet: new temperatures, new system prompts, new
models, or quantization not in the released data. This is the existing inference path — **follow it
there, don't duplicate it here**:

1. **Build a dataset** (the generation grid: temperatures, system prompts, prompt indices) —
   [`../codebase/dataset.md`](../codebase/dataset.md) §"Creating a new dataset". Sampling
   parameters live in the **CSV rows**, not in `Inference_configs.yaml`.
2. **Make sure your models are registered** in [`config/models.yaml`](../../config/models.yaml)
   (base / quantized / abliterated).
3. **Run inference** — [`../codebase/inference.md`](../codebase/inference.md) and the GPU
   walkthrough [`mode-b.md`](mode-b.md) (install, per-model sizing, parse + merge + flatten).

Once your run is on disk, **return to Path A**: write a scenario that selects the cells you just
generated, and run the pipeline.

---

## Coordination contract & pitfalls

The scenario must be a **subset** of what was generated. Common mismatches:

| Pitfall | Symptom |
|---|---|
| Scenario selects a `(model, temp, sp)` cell with no generated data | empty / degenerate class |
| Token-pair set at analysis ≠ the set inference used | train/test skew or missing pairs |
| A base model not in `config/models.yaml` | `load_scenario` raises `ValueError` (names the offender) |
| Editing a source CSV under `datasets/Bits_Datasets/` | invalidates the released feature cache — its SHA-256 gates the cache (see [`../../config/scenarios/README.md`](../../config/scenarios/README.md) §CSV invariant) |
| Crossing abliteration off its pinned cell | rejected — abliteration is pinned to `temperature=1.0, system_prompt_idx=-1` (no released data elsewhere) |

## Validate before you spend compute

- The scenario validates the instant it loads (model names, quantization pairs, abliteration pin):
  ```bash
  python -c "from audit_llm.scenarios.loader import load_scenario; print(load_scenario('<name>'))"
  ```
- `make gate-b-parity` is a data-free firewall asserting the scenario and legacy paths agree on the
  label set.
- Run the shipped `make example-scenario` once to confirm your toolchain end to end.

## See also

- [`mode-a.md`](mode-a.md) — the no-GPU analysis tiers and where figures land.
- [`mode-b.md`](mode-b.md) — the from-scratch GPU inference path.
- [`../codebase/scenarios.md`](../codebase/scenarios.md) — the enumerator and the shipped scenarios.
- [`../../config/scenarios/main.yaml`](../../config/scenarios/main.yaml) — the 237-instance paper
  default, a fully-populated scenario to copy from.
