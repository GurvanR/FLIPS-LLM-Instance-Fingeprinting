#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: CC-BY-4.0
"""Selection spec + carver for the zero-download light smoke subset.

This script builds ``Productions/FLiPS_ICML_light_subset/`` — a ~95 MB,
self-contained slice of the 4.2 GB canonical light run that a CPU-only smoke
classification can run against with **zero network and zero user upload**. It is
the in-repo fixture behind ``make smoke`` and ``make gate-golden``.

SAFETY (governs this whole phase)
---------------------------------
The canonical run at ``CANONICAL_SRC`` is **READ-ONLY**. We never write, move,
or delete anything there. We ``cp --reflink=auto`` only the *needed* source
paths into a temporary CoW scratch dir and carve from the scratch; the canonical
``run_config.pickle`` size+mtime is asserted unchanged before and after.

WHAT GETS CARVED (flattened — NO ``merged_sub_runs/`` level, so
``AuditionsAnalysis(run_path)`` finds ``run_config`` directly under ``run_path``):
- ``run_config.json`` + ``run_config.pickle`` — **scrubbed** of the HPC cluster
  filesystem paths (``model_path_JZ`` dropped; ``*_model_path`` dicts blanked). Analysis
  only needs ``scrapping_rule`` / ``min_seq_length`` / ``MAX_TOKENS`` /
  ``Dataset_relative_path`` / ``TOKEN_PAIRS_SET`` / the ``*_models`` dicts.
- ``Analysis/Graph_Numeric/Answers.parquet`` — filtered to ``TOKEN_PAIRS`` ×
  ``BASE_MODELS`` (all temperature/system_prompt rows, order preserved).
- ``Experiments/feature_computation_data/<tp>/{manifest.json, intra/<model>.npy}``
  for each tp — ``manifest["models"]`` trimmed to the chosen base models (sorted)
  plus ``numpy_default`` (the lone ``PRNG_MODELS`` entry, kept for model-axis
  alignment); ``feature_index`` untouched; ``inter/`` dropped (never loaded).
- ``Experiments/model_index.json`` — chosen base (sorted) + ``numpy_default``.
- ``golden_refs.json`` — see ``--capture-golden``.

ROW ALIGNMENT (why subsetting is safe): ``intra/<model>.npy`` is ``(N_iter, 57)``
with ``N_iter = len(datasets/Bits_Datasets/FLiPS_ICML.csv) = 10000``; row ``i`` ↔
CSV row ``i``. ``get_samples_indices`` selects rows via the CSV variation grid,
not Answers row positions, so npy rows must stay fully intact — we subset only
the **models** and **token-pairs** axes, never the sample rows. The CSV is kept
byte-identical (its SHA-256 matches every ``manifest.source_csv_hash``).

USAGE
-----
python scripts/make_light_subset.py --dry-run        # print the plan, no writes
python scripts/make_light_subset.py                  # carve the subset
python scripts/make_light_subset.py --capture-golden # carve + run smoke + golden_refs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Selection spec (edit here to change the carve)                              #
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
# Canonical light run lives in the sibling `audit-llm` checkout (this build machine).
# Override with FLIPS_LIGHT_RUN_SRC if your layout differs. No absolute path is baked
# in so the repo carries no machine-specific leak (the only place this is run).
CANONICAL_SRC = Path(
    os.environ.get(
        "FLIPS_LIGHT_RUN_SRC",
        str(REPO_ROOT.parent / "audit-llm" / "Productions" / "FLiPS_ICML_light_run" / "merged_sub_runs"),
    )
)
SUBSET_ROOT = REPO_ROOT / "Productions" / "FLiPS_ICML_light_subset"
RUN_NAME = "FLiPS_ICML_light_subset"

# 5 token pairs and 3 diverse base models. Five pairs (not 3) so the mix_tp pipeline
# can run unique_tp_in_mix='max' (the schema default) for batch sizes up to 4: 'max'
# draws bs distinct pairs per mixed batch and the uplet builder needs
# C(n_pairs, bs) >= max_nb_of_uplet (=2); C(5,4)=5 satisfies it, C(4,4)=1 would not.
# This lets every light-subset config omit unique_tp_in_mix (see e3_llmmap_baseline,
# Smoke_light, example_custom), provided their batch_prediction_sizes stay <= 4.
# Classes = BASE_MODELS × the Smoke_light variation grid; numpy_default is carried
# in the feature cache for model-axis alignment but excluded from classification
# (Smoke_light sets PRNGs: None).
# All five are members of the "FLiPS" token-pair group (get_token_pairs_of_group),
# which mix_tp_at_pred requires; '0-1' is the only available pair NOT in that group.
TOKEN_PAIRS = ["ali-yg", "awk-ise", "crit-SA", "City-load", "df-bits"]
BASE_MODELS = [
    "CohereForAI/c4ai-command-r-plus",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "google/gemma-2-9b-it",
]
PRNG_MODEL = "numpy_default"  # the only entry of audit_llm ...model_names.PRNG_MODELS

# Strictly enforced: keeps the carve bounded and zero-download. ~18 MB/token-pair
# (3 base models + numpy_default), so 5 pairs land near ~95 MB; 110 MB leaves headroom.
MAX_SUBSET_BYTES = 110 * 1024 * 1024  # ~110 MB

SMOKE_CONFIG_DIR = "Smoke_light"
SMOKE_CONFIG_NAME = "Smoke_light"
GOLDEN_REFS_PATH = SUBSET_ROOT / "golden_refs.json"

# Header note for run_config keys that carry HPC filesystem paths (scrubbed out).
_LEAK_KEYS_DROPPED = ["model_path_JZ"]
_PATH_DICT_KEYS = ["vllm_model_path", "hf_model_path", "openrouter_model_path"]


def _sanitize(name: str) -> str:
    """'/' -> '__' (mirror of audit_llm.file_io.sanitize_model_name)."""
    return name.replace("/", "__")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _human(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} MB"


class SizeGuard:
    """Track bytes written into the subset and abort past MAX_SUBSET_BYTES."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.total = 0

    def add(self, nbytes: int, what: str) -> None:
        if self.total + nbytes > self.limit:
            raise SystemExit(
                f"SIZE GUARD: writing {what} (+{_human(nbytes)}) would exceed the "
                f"{_human(self.limit)} budget (running total {_human(self.total)}). "
                "Reduce TOKEN_PAIRS or BASE_MODELS."
            )
        self.total += nbytes


# --------------------------------------------------------------------------- #
# run_config scrub                                                            #
# --------------------------------------------------------------------------- #
def scrub_run_config(raw: dict) -> dict:
    """Strip HPC cluster filesystem paths; keep only analysis-relevant content intact."""
    cfg = dict(raw)
    for k in _LEAK_KEYS_DROPPED:
        cfg.pop(k, None)
    # Blank the model-path dicts but keep their keys so analysis lookups never
    # KeyError; these paths are only used by the (GPU) inference path.
    for k in _PATH_DICT_KEYS:
        if isinstance(cfg.get(k), dict):
            cfg[k] = {mk: "" for mk in cfg[k]}
        elif k in cfg:
            cfg[k] = ""
    cfg["_light_subset_note"] = (
        "Scrubbed slice of FLiPS_ICML_light_run/merged_sub_runs for the in-repo "
        "smoke fixture. HPC filesystem paths removed; only analysis-relevant fields kept."
    )
    # Defensive: blow up if any leak path survives anywhere in the structure.
    blob = json.dumps(cfg, default=str)
    for needle in ("/lustre", "/home/", "@a100"):
        if needle in blob:
            raise SystemExit(f"run_config scrub failed: '{needle}' still present.")
    return cfg


# --------------------------------------------------------------------------- #
# Carve                                                                       #
# --------------------------------------------------------------------------- #
def _cow_copy(src: Path, dst: Path) -> None:
    """cp --reflink=auto src dst (CoW if the FS supports it, else a plain copy)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cp", "--reflink=auto", "-r", str(src), str(dst)],
        check=True,
    )


def _canonical_sentinel() -> tuple[int, int]:
    st = (CANONICAL_SRC / "run_config.pickle").stat()
    return (st.st_size, st.st_mtime_ns)


def plan_lines() -> list[str]:
    sorted_base = sorted(BASE_MODELS)
    manifest_models = [_sanitize(m) for m in sorted_base] + [PRNG_MODEL]
    npy_per_tp = len(manifest_models)
    # estimate intra size from canonical
    est = 0
    for tp in TOKEN_PAIRS:
        intra = CANONICAL_SRC / "Experiments" / "feature_computation_data" / tp / "intra"
        for sm in manifest_models:
            f = intra / f"{sm}.npy"
            if f.exists():
                est += f.stat().st_size
    lines = [
        "LIGHT SUBSET SELECTION PLAN",
        f"  canonical src : {CANONICAL_SRC}  (READ-ONLY)",
        f"  subset dest   : {SUBSET_ROOT}  (FLATTENED — no merged_sub_runs/ level)",
        f"  run name      : {RUN_NAME}",
        f"  token pairs   : {TOKEN_PAIRS}",
        f"  base models   : {sorted_base}",
        f"  + PRNG model  : {PRNG_MODEL} (axis alignment; excluded from classify)",
        f"  intra npy/tp  : {npy_per_tp} files",
        f"  est intra size: {_human(est)} ({len(TOKEN_PAIRS)} tp)",
        "",
        "FLATTENED LAYOUT to be written:",
        "  run_config.json                                  (scrubbed, leak-free)",
        "  run_config.pickle                                (scrubbed)",
        "  Analysis/Graph_Numeric/Answers.parquet           (filtered tp×models)",
        "  Experiments/model_index.json",
    ]
    for tp in TOKEN_PAIRS:
        lines.append(f"  Experiments/feature_computation_data/{tp}/manifest.json")
        lines.append(f"  Experiments/feature_computation_data/{tp}/intra/<{npy_per_tp} models>.npy")
    lines += [
        "  golden_refs.json                                 (--capture-golden)",
        "",
        "GOLDEN_REFS plan (--capture-golden):",
        f"  run smoke via XP_configs/run_experiments.py --config-dir {SMOKE_CONFIG_DIR}",
        f"    --run-name {RUN_NAME} --xp-suffix smoke --mode inline",
        "  capture: label_set (sorted class labels), feature_cache_md5 (per intra npy),",
        "           smoke_accuracy {batch_type: accuracy_mean}; then delete generated outputs.",
        f"  size guard: abort if subset would exceed {_human(MAX_SUBSET_BYTES)}.",
    ]
    return lines


def carve(guard: SizeGuard) -> dict[str, str]:
    """Carve the subset from a CoW scratch of the canonical run. Returns md5 map."""
    sentinel_before = _canonical_sentinel()
    sorted_base = sorted(BASE_MODELS)
    manifest_models = [_sanitize(m) for m in sorted_base] + [PRNG_MODEL]

    if SUBSET_ROOT.exists():
        shutil.rmtree(SUBSET_ROOT)
    SUBSET_ROOT.mkdir(parents=True)

    feature_md5: dict[str, str] = {}

    with tempfile.TemporaryDirectory(prefix="light_subset_scratch_") as tmp:
        scratch = Path(tmp) / "scratch"
        scratch.mkdir()

        # 1. CoW-copy only the needed source paths into scratch.
        _cow_copy(CANONICAL_SRC / "run_config.pickle", scratch / "run_config.pickle")
        _cow_copy(
            CANONICAL_SRC / "Analysis" / "Graph_Numeric" / "Answers.parquet",
            scratch / "Answers.parquet",
        )
        for tp in TOKEN_PAIRS:
            src_tp = CANONICAL_SRC / "Experiments" / "feature_computation_data" / tp
            _cow_copy(src_tp / "manifest.json", scratch / tp / "manifest.json")
            for sm in manifest_models:
                _cow_copy(src_tp / "intra" / f"{sm}.npy", scratch / tp / "intra" / f"{sm}.npy")

        # 2. Scrubbed run_config.{json,pickle}
        with open(scratch / "run_config.pickle", "rb") as f:
            raw_cfg = pickle.load(f)
        cfg = scrub_run_config(raw_cfg)
        cfg_json = json.dumps(cfg, indent=2, default=str).encode()
        guard.add(len(cfg_json), "run_config.json")
        (SUBSET_ROOT / "run_config.json").write_bytes(cfg_json)
        cfg_pkl = pickle.dumps(cfg)
        guard.add(len(cfg_pkl), "run_config.pickle")
        (SUBSET_ROOT / "run_config.pickle").write_bytes(cfg_pkl)

        # 3. Filtered Answers.parquet (lazy scan to keep RAM bounded).
        import polars as pl

        answers_dst = SUBSET_ROOT / "Analysis" / "Graph_Numeric"
        answers_dst.mkdir(parents=True)
        out_parquet = answers_dst / "Answers.parquet"
        (
            pl.scan_parquet(scratch / "Answers.parquet")
            .filter(
                pl.col("Token_pair").is_in(TOKEN_PAIRS) & pl.col("Model").is_in(BASE_MODELS)
            )
            .sink_parquet(out_parquet)
        )
        guard.add(out_parquet.stat().st_size, "Answers.parquet")

        # 4. model_index.json (base sorted + PRNG, 0-based).
        exp_dst = SUBSET_ROOT / "Experiments"
        exp_dst.mkdir(parents=True)
        model_index = {m: i for i, m in enumerate(sorted_base + [PRNG_MODEL])}
        mi_json = json.dumps(model_index, indent=4).encode()
        guard.add(len(mi_json), "model_index.json")
        (exp_dst / "model_index.json").write_bytes(mi_json)

        # 5. Per-token-pair manifest + intra npy.
        for tp in TOKEN_PAIRS:
            src_manifest = json.loads((scratch / tp / "manifest.json").read_text())
            new_manifest = dict(src_manifest)
            new_manifest["models"] = manifest_models  # trimmed, order-aligned
            tp_dst = exp_dst / "feature_computation_data" / tp
            (tp_dst / "intra").mkdir(parents=True)
            man_bytes = json.dumps(new_manifest, indent=2).encode()
            guard.add(len(man_bytes), f"{tp}/manifest.json")
            (tp_dst / "manifest.json").write_bytes(man_bytes)
            for sm in manifest_models:
                src_npy = scratch / tp / "intra" / f"{sm}.npy"
                dst_npy = tp_dst / "intra" / f"{sm}.npy"
                guard.add(src_npy.stat().st_size, f"{tp}/intra/{sm}.npy")
                shutil.copy2(src_npy, dst_npy)
                feature_md5[f"{tp}/{sm}.npy"] = _md5(dst_npy)

    sentinel_after = _canonical_sentinel()
    if sentinel_before != sentinel_after:
        raise SystemExit("FATAL: canonical run_config.pickle changed — read-only invariant violated.")

    return feature_md5


# --------------------------------------------------------------------------- #
# Golden refs                                                                 #
# --------------------------------------------------------------------------- #
def _run_smoke() -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "XP_configs" / "run_experiments.py"),
        "--config-dir", SMOKE_CONFIG_DIR,
        "--run-name", RUN_NAME,
        "--xp-suffix", "smoke",
        "--mode", "inline",
    ]
    print("Running smoke:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, stdin=subprocess.DEVNULL)


def _xp_output_dir() -> Path:
    return (
        SUBSET_ROOT
        / "Experiments"
        / "Batch_Classification_across_token_pairs"
        / f"{SMOKE_CONFIG_NAME}_smoke"
    )


def _collect_smoke_accuracy() -> dict[str, Any]:
    """Pull accuracy from the closed-set joblib checkpoints.

    The closed-set classifier stores results as ``train_size*_all.pkl`` checkpoints
    shaped ``{batch_type: {bs: {tp_or_uplet: {clf: {accuracy_mean, ...}}}}}`` (NOT
    the LLMmap ``results.json`` shape). We aggregate per ``(batch_type, bs, clf)``:
    the mean ``accuracy_mean`` across token-pairs/uplets, plus the per-tp values.
    """
    import joblib

    out: dict[str, Any] = {}
    base = _xp_output_dir()
    for ckpt in sorted(base.rglob("train_size*_all.pkl")):
        try:
            data = joblib.load(ckpt)
        except Exception:
            continue
        for batch_type, by_bs in data.items():
            if not isinstance(by_bs, dict):
                continue
            for bs, by_tp in by_bs.items():
                if not isinstance(by_tp, dict):
                    continue
                for tp, by_clf in by_tp.items():
                    for clf, metrics in by_clf.items():
                        if not isinstance(metrics, dict) or "accuracy_mean" not in metrics:
                            continue
                        key = f"{batch_type}|bs={bs}|{clf}"
                        slot = out.setdefault(key, {"per_tp": {}})
                        slot["per_tp"][str(tp)] = round(float(metrics["accuracy_mean"]), 6)
    # Aggregate: mean accuracy_mean across token-pairs per (batch_type, bs, clf).
    for key, slot in out.items():
        vals = list(slot["per_tp"].values())
        slot["accuracy_mean_over_tps"] = round(sum(vals) / len(vals), 6) if vals else None
    return out


def _collect_label_set() -> list[str]:
    """Read the materialised classification class labels from new_var_models_idx.json."""
    labels: set[str] = set()
    base = _xp_output_dir()
    for nm in sorted(base.rglob("new_var_models_idx.json")):
        try:
            data = json.loads(nm.read_text())
        except Exception:
            continue
        if isinstance(data, dict):
            labels.update(str(v) for v in data.values())  # values are the class labels
    return sorted(labels)


def capture_golden(feature_md5: dict[str, str]) -> None:
    _run_smoke()
    golden = {
        "_doc": (
            "Out-of-box golden regression reference for the light subset. "
            "Re-captured by `make gate-golden` running the SAME XP_configs/Smoke_light config. "
            "Layered tolerances: label_set EXACT; feature_cache md5 within-env / np.allclose cross-env; "
            "accuracy band >= +/-5pp mix_tp, +/-2pp tp_wise (wide: unseeded token_pair_mixing variance)."
        ),
        "smoke_config": f"XP_configs/{SMOKE_CONFIG_DIR}/{SMOKE_CONFIG_NAME}.yaml",
        "run_name": RUN_NAME,
        "token_pairs": TOKEN_PAIRS,
        "base_models": sorted(BASE_MODELS),
        "feature_cache_md5": feature_md5,
        "label_set": _collect_label_set(),
        "smoke_results": _collect_smoke_accuracy(),
    }
    GOLDEN_REFS_PATH.write_text(json.dumps(golden, indent=2))
    print(f"wrote {GOLDEN_REFS_PATH}")


def clean_generated_outputs() -> None:
    """Remove smoke-generated artefacts so only the input subset + golden_refs commit."""
    gen = _xp_output_dir().parent  # Batch_Classification_across_token_pairs/
    if gen.exists():
        shutil.rmtree(gen)
    for sub in ("Analysis/Graph_Numeric/Figures", "Analysis/clf_comparison_debug"):
        p = SUBSET_ROOT / sub
        if p.exists():
            shutil.rmtree(p)
    # filter_token_pairs() auto-creates this when absent; drop the generated copy so
    # the committed subset stays input-only (gate-golden's re-run regenerates it).
    banned = SUBSET_ROOT / "Experiments" / "banned_token_pairs.json"
    if banned.exists():
        banned.unlink()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="Print the plan, write nothing.")
    ap.add_argument(
        "--capture-golden",
        action="store_true",
        help="After carving, run the Smoke_light classification and write golden_refs.json.",
    )
    args = ap.parse_args(argv)

    if not CANONICAL_SRC.is_dir():
        print(f"error: canonical source not found: {CANONICAL_SRC}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\n".join(plan_lines()))
        return 0

    guard = SizeGuard(MAX_SUBSET_BYTES)
    feature_md5 = carve(guard)
    print(f"carved {SUBSET_ROOT} — running total {_human(guard.total)} (budget {_human(MAX_SUBSET_BYTES)})")

    if args.capture_golden:
        capture_golden(feature_md5)
        clean_generated_outputs()
        # Re-print final on-disk size after cleanup.
        total = sum(f.stat().st_size for f in SUBSET_ROOT.rglob("*") if f.is_file())
        print(f"final subset size after golden capture + cleanup: {_human(total)}")
    else:
        # Write a golden_refs stub with md5s so the layout is complete even without a run.
        GOLDEN_REFS_PATH.write_text(
            json.dumps(
                {
                    "_doc": "Run `make_light_subset.py --capture-golden` to fill smoke_results/label_set.",
                    "feature_cache_md5": feature_md5,
                    "smoke_results": {},
                    "label_set": [],
                },
                indent=2,
            )
        )
        print(f"wrote {GOLDEN_REFS_PATH} (stub — re-run with --capture-golden for accuracy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
