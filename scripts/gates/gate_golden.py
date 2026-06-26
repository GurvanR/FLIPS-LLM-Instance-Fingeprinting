#!/usr/bin/env python3
"""Gate-golden — the out-of-box golden regression (CPU-only, zero-download).

Re-runs the SAME ``XP_configs/Smoke_light`` classification originally used to capture
``Productions/FLiPS_ICML_light_subset/golden_refs.json``, then asserts three LAYERED
tolerances against that reference. This is the build-device runtime cross-check that the
the sanitize + scenario refactor preserved behaviour — a numeric regression no
``--dry-run`` can catch.

    LAYER 1 — enumerator label set EXACT
        The materialised classification class labels (from ``new_var_models_idx.json``)
        must equal ``golden_refs["label_set"]`` exactly (set + sorted order).

    LAYER 2 — feature-cache parity
        Default (within the pinned env): the md5 of every shipped intra npy under
        ``Experiments/feature_computation_data/<tp>/intra/<model>.npy`` must equal the
        captured ``golden_refs["feature_cache_md5"]`` value. The committed cache is *read*
        (cache hit), not recomputed, so md5 is byte-stable even across a fresh clone.
        Cross-env ``np.allclose`` (tight rtol) is wired behind ``--reference-features
        <dir>`` for the scenario where features are *regenerated* on a different
        numpy/BLAS build (a bare md5 breaks there — golden-repro finding); it compares
        each committed npy to a second-env copy. Not exercised on the build device.

    LAYER 3 — accuracy within band
        Per ``<batch_type>|bs=<n>|<clf>`` the re-run ``accuracy_mean_over_tps`` must be
        within ±2 pp of golden for ``tp_wise`` keys and ±5 pp for ``mix_tp`` keys — the
        wide band the unseeded ``token_pair_mixing.py`` mixing requires (~±3.4 pp
        run-to-run, same env/seed).

Extraction parity: the label-set / accuracy / md5 extractors are imported from
``scripts/make_light_subset.py`` (the capture script) so the verify path reads the
checkpoints byte-identically to how the reference was captured — if they diverged the
gate would be meaningless. That import also fixes the run identity to
RUN_NAME=``FLiPS_ICML_light_subset`` / xp-suffix=``smoke`` (``_xp_output_dir()``), which
is exactly what the golden gate is defined over.

Run robustness mirrors gate-A: ``run_experiments.py`` does not propagate the inner XP
exit code (it only prints ``Command failed…``), so we also scan the produced
``xp_logs/.../xp_*_log_*.log`` for ``Traceback``/``FileNotFoundError`` and assert the
checkpoint exists. Generated outputs under the committed subset are removed afterwards
(unless ``--keep-outputs``) so the fixture stays byte-clean and the gate is idempotent.

Wired into the Makefile as ``make gate-golden``.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_EXPERIMENTS = REPO_ROOT / "XP_configs" / "run_experiments.py"
SUBSET_ROOT = REPO_ROOT / "Productions" / "FLiPS_ICML_light_subset"
GOLDEN_REFS_PATH = SUBSET_ROOT / "golden_refs.json"

SMOKE_CONFIG_DIR = "Smoke_light"
XP_SUFFIX = "smoke"  # MUST match make_light_subset._xp_output_dir() (Smoke_light_smoke)
RUN_NAME = "FLiPS_ICML_light_subset"

LOG_ERROR_MARKERS = ("Traceback (most recent call last)", "FileNotFoundError")

# Layer-3 accuracy bands (absolute, fraction units). tp_wise is tight; the mix_tp
# batch-types carry the unseeded token-pair-mixing variance and need a wide band.
BAND_TP_WISE = 0.02   # ±2 pp
BAND_MIX_TP = 0.05    # ±5 pp


def _load_capture_module():
    """Import scripts/make_light_subset.py so the verify extractors are byte-identical
    to the capture extractors (the core correctness property of a golden gate).

    Loaded by file path (scripts/ is not an importable package). Module-level code is
    side-effect-free: it only builds Path constants; polars/joblib/numpy are imported
    lazily inside the functions we call.
    """
    src = REPO_ROOT / "scripts" / "make_light_subset.py"
    spec = importlib.util.spec_from_file_location("make_light_subset", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _latest_xp_log() -> Path | None:
    log_dir = REPO_ROOT / "xp_logs" / RUN_NAME / XP_SUFFIX
    logs = sorted(log_dir.glob("xp_*_log_*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def _run_smoke() -> tuple[bool, str]:
    """Run the Smoke_light config inline; return (ok, message). ok=False on any failure
    the dispatcher prints, a dirty xp log, or a missing checkpoint."""
    cmd = [
        sys.executable, str(RUN_EXPERIMENTS),
        "--config-dir", SMOKE_CONFIG_DIR,
        "--run-name", RUN_NAME,
        "--xp-suffix", XP_SUFFIX,
        "--mode", "inline",
    ]
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT),
        stdin=subprocess.DEVNULL,  # so click.confirm can't hang headless
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    out = proc.stdout or ""
    if proc.returncode != 0 or "Command failed with exit code" in out:
        return False, "dispatcher reported a failed experiment\n" + out.strip()[-1500:]
    log = _latest_xp_log()
    if log is None:
        return False, "no xp log produced (the experiment never started)"
    text = log.read_text(errors="replace")
    for marker in LOG_ERROR_MARKERS:
        if marker in text:
            return False, f"{marker} found in {log.relative_to(REPO_ROOT)}"
    return True, str(log.relative_to(REPO_ROOT))


def _check_label_set(mls, golden: dict) -> tuple[bool, str]:
    actual = mls._collect_label_set()
    expected = sorted(golden.get("label_set", []))
    if actual == expected:
        return True, f"{len(actual)} labels match exactly"
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    detail = []
    if missing:
        detail.append(f"missing {missing}")
    if extra:
        detail.append(f"unexpected {extra}")
    if not detail:  # same set, different order
        detail.append("set equal but order differs")
    return False, "; ".join(detail)


def _check_feature_md5(mls, golden: dict, reference_features: Path | None) -> tuple[bool, str]:
    expected = golden.get("feature_cache_md5", {})
    if not expected:
        return False, "golden_refs has no feature_cache_md5"
    fcd = SUBSET_ROOT / "Experiments" / "feature_computation_data"
    mismatches: list[str] = []
    for rel, exp_md5 in sorted(expected.items()):  # rel = "<tp>/<sanitized_model>.npy"
        tp, _, fname = rel.partition("/")
        npy = fcd / tp / "intra" / fname
        if not npy.exists():
            mismatches.append(f"{rel}: missing committed npy")
            continue
        got = mls._md5(npy)
        if got == exp_md5:
            continue
        if reference_features is not None:
            # Cross-env: compare values, not bytes (numpy/BLAS-tolerant).
            import numpy as np
            ref_npy = reference_features / tp / "intra" / fname
            if not ref_npy.exists():
                mismatches.append(f"{rel}: md5 differs and no reference npy at {ref_npy}")
            elif np.allclose(np.load(npy), np.load(ref_npy), rtol=1e-5, atol=1e-8):
                continue  # values match within tolerance — cross-env parity OK
            else:
                mismatches.append(f"{rel}: np.allclose failed vs reference")
        else:
            mismatches.append(f"{rel}: md5 {got} != golden {exp_md5}")
    if mismatches:
        return False, "; ".join(mismatches[:6]) + (" …" if len(mismatches) > 6 else "")
    mode = "np.allclose (cross-env)" if reference_features else "md5 (within-env)"
    return True, f"{len(expected)} feature caches parity OK [{mode}]"


def _band_for(key: str) -> float:
    if "tp_wise" in key:
        return BAND_TP_WISE
    return BAND_MIX_TP  # mix_tp_at_pred / mix_tp_at_train etc.


def _check_accuracy(mls, golden: dict) -> tuple[bool, str]:
    actual = mls._collect_smoke_accuracy()
    expected = golden.get("smoke_results", {})
    if not expected:
        return False, "golden_refs has no smoke_results"
    failures: list[str] = []
    checked = 0
    for key, slot in sorted(expected.items()):
        exp_acc = slot.get("accuracy_mean_over_tps")
        if exp_acc is None:
            continue
        if key not in actual or actual[key].get("accuracy_mean_over_tps") is None:
            failures.append(f"{key}: not reproduced by the re-run")
            continue
        got_acc = actual[key]["accuracy_mean_over_tps"]
        band = _band_for(key)
        checked += 1
        if abs(got_acc - exp_acc) > band:
            failures.append(
                f"{key}: got {got_acc:.4f} vs golden {exp_acc:.4f} (Δ={abs(got_acc-exp_acc):.4f} > band {band})"
            )
    if failures:
        return False, "; ".join(failures)
    return True, f"{checked} accuracy keys within band (±2pp tp_wise / ±5pp mix_tp)"


def main() -> int:
    import json

    ap = argparse.ArgumentParser(description="Gate-golden: out-of-box golden regression.")
    ap.add_argument("--keep-outputs", action="store_true",
                    help="Do not delete the generated smoke outputs (for debugging).")
    ap.add_argument("--reference-features", type=Path, default=None,
                    help="Cross-env: a second-env feature_computation_data dir; Layer 2 then "
                         "uses np.allclose instead of md5 when md5 differs.")
    args = ap.parse_args()

    if not GOLDEN_REFS_PATH.exists():
        print(f"[FAIL] gate-golden: golden_refs not found at {GOLDEN_REFS_PATH.relative_to(REPO_ROOT)}")
        return 1
    golden = json.loads(GOLDEN_REFS_PATH.read_text())

    mls = _load_capture_module()

    print(f"=== gate-golden : re-run {SMOKE_CONFIG_DIR} on {RUN_NAME} (CPU, zero-download) ===")
    # Pre-clean so we exercise the from-scratch classify (no stale checkpoint masks a regression).
    if not args.keep_outputs:
        mls.clean_generated_outputs()

    run_ok, run_msg = _run_smoke()
    if not run_ok:
        print(f"  [FAIL] smoke run: {run_msg}")
        if not args.keep_outputs:
            mls.clean_generated_outputs()
        return 1
    print(f"  smoke run log clean: {run_msg}")

    layers = [
        ("LAYER 1 label-set EXACT", _check_label_set(mls, golden)),
        ("LAYER 2 feature-cache parity", _check_feature_md5(mls, golden, args.reference_features)),
        ("LAYER 3 accuracy band", _check_accuracy(mls, golden)),
    ]

    if not args.keep_outputs:
        mls.clean_generated_outputs()

    ok = all(passed for _, (passed, _) in layers)
    print()
    for name, (passed, detail) in layers:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    print("=" * 60)
    print(f"gate-golden: {'PASS' if ok else 'FAIL'} ({sum(p for _, (p, _) in layers)}/{len(layers)} layers)")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
