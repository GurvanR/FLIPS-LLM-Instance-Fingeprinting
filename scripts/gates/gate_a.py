#!/usr/bin/env python3
"""Gate-A — the always-runs, zero-download reproduction gate (stdlib only).

Both checks run against the committed in-repo light subset
(``Productions/FLiPS_ICML_light_subset/``) — no network, CPU-only, must pass on a fresh clone:

  PART 1 — LLMmap E3 replot
    Run the LLMmap-only F01 variant (``XP_configs/e3_llmmap_baseline/LLMmap_only_tp.yaml``) through
    ``run_experiments.py`` and confirm it emits the F01 PDF regenerated from the in-repo
    ``XP_configs/e3_flips_vs_llmmap/llmmap_if_data/`` cache with NO ``FileNotFoundError``/``Traceback``.
    The variant's own config-dir holds exactly one YAML, so the dispatcher's ``*.yaml`` glob is
    unambiguous and never picks up the crashing 3-curve ``e3_flips_vs_llmmap/FLIPS_mix_tp.yaml``.

  PART 2 — light-subset smoke
    Run ``XP_configs/Smoke_light/`` and confirm it produces >= 1 PDF.

Why we scan logs instead of trusting the exit code: ``run_experiments.py`` runs each experiment in a
subprocess and only *prints* on failure (run_experiments.py:125-128) — it returns 0 even when the
inner XP crashed. So we (a) detect ``Command failed`` in the dispatcher's stdout, (b) scan the
produced ``xp_logs/.../xp_*_log_*.log`` for ``Traceback``/``FileNotFoundError``, and (c) assert the
expected PDF(s) exist on disk.

Git hygiene: generated outputs land under the COMMITTED subset
(``Productions/<run>/Experiments/Batch_Classification_across_token_pairs/<xp_name>/``). The root
``.gitignore`` un-ignores everything under ``FLiPS_ICML_light_subset/**``, so we remove the xp dirs
we created (unless ``--keep-outputs``) — the gate is idempotent and leaves the committed fixture
byte-clean. We only ever delete dirs we created under the in-repo subset; the carved fixture
(``run_config``, ``Analysis/``, ``Experiments/feature_computation_data/``) is never touched.

Wired into the Makefile as ``make gate-a`` and reused for ``make repro-llmmap-e3`` (part 1 only).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_EXPERIMENTS = REPO_ROOT / "XP_configs" / "run_experiments.py"

DEFAULT_RUN_NAME = "FLiPS_ICML_light_subset"
DEFAULT_XP_SUFFIX = "gate_a"
VARIANT_CONFIG_DIR = "e3_llmmap_baseline"   # holds ONLY LLMmap_only_tp.yaml (unambiguous glob)
SMOKE_CONFIG_DIR = "Smoke_light"
EXPERIMENT_FUN = "Batch_Classification_across_token_pairs"

# Substrings that mark a failed inner XP run in the captured log / dispatcher stdout.
LOG_ERROR_MARKERS = ("Traceback (most recent call last)", "FileNotFoundError")


def _config_stems(config_dir: str) -> list[str]:
    """YAML stems in XP_configs/<config_dir>/ (mirrors run_experiments._find_yaml_configs)."""
    root = REPO_ROOT / "XP_configs" / config_dir
    exclude = {"XP_config_libs", "Old_configs"}
    return sorted(
        p.stem
        for p in root.glob("*.yaml")
        if not any(part in exclude for part in p.parts)
    )


def _xp_output_dirs(run_name: str, config_dir: str, xp_suffix: str) -> list[Path]:
    """The per-experiment output dirs produced by a run (xp_name = <stem>_<suffix>)."""
    base = (
        REPO_ROOT / "Productions" / run_name / "Experiments" / EXPERIMENT_FUN
    )
    return [base / f"{stem}_{xp_suffix}" for stem in _config_stems(config_dir)]


def _latest_xp_log(run_name: str, xp_suffix: str) -> Path | None:
    log_dir = REPO_ROOT / "xp_logs" / run_name / xp_suffix
    logs = sorted(log_dir.glob("xp_*_log_*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def _safe_rmtree_under_subset(path: Path, run_name: str) -> None:
    """Remove *path* only if it is strictly inside Productions/<run_name>/Experiments/<fun>/.

    Guard rails so a bad run-name can never delete the committed fixture or anything outside the
    in-repo subset's generated-experiments dir.
    """
    allowed_parent = (
        REPO_ROOT / "Productions" / run_name / "Experiments" / EXPERIMENT_FUN
    ).resolve()
    target = path.resolve()
    if target.parent == allowed_parent and target.is_dir():
        shutil.rmtree(target)


def _banned_token_pairs_path(run_name: str) -> Path:
    """The regenerable cache run_xp writes directly under Experiments/ (experiment_runner.py:114)."""
    return REPO_ROOT / "Productions" / run_name / "Experiments" / "banned_token_pairs.json"


def _run_experiment(config_dir: str, run_name: str, xp_suffix: str) -> tuple[bool, str]:
    """Invoke run_experiments.py inline. Returns (dispatcher_ok, combined_stdout)."""
    cmd = [
        sys.executable,
        str(RUN_EXPERIMENTS),
        "--config-dir", config_dir,
        "--run-name", run_name,
        "--xp-suffix", xp_suffix,
        "--mode", "inline",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        stdin=subprocess.DEVNULL,  # so click.confirm at run_experiments.py:296 can't hang headless
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = proc.stdout or ""
    dispatcher_ok = proc.returncode == 0 and "Command failed with exit code" not in out
    return dispatcher_ok, out


def _check_log_clean(run_name: str, xp_suffix: str) -> tuple[bool, str]:
    log = _latest_xp_log(run_name, xp_suffix)
    if log is None:
        return False, "no xp log produced (the experiment never started)"
    text = log.read_text(errors="replace")
    for marker in LOG_ERROR_MARKERS:
        if marker in text:
            return False, f"{marker} found in {log.relative_to(REPO_ROOT)}"
    return True, str(log.relative_to(REPO_ROOT))


def _run_part(
    *, name: str, config_dir: str, run_name: str, xp_suffix: str,
    pdf_glob: str, require_f01: bool,
) -> bool:
    """Run one gate part and verify it (clean log + expected PDF). Returns pass/fail."""
    print(f"\n=== gate-A {name} : --config-dir {config_dir} --run-name {run_name} ===")

    # Pre-clean so we exercise the from-scratch path (no stale checkpoint can mask a failure).
    for d in _xp_output_dirs(run_name, config_dir, xp_suffix):
        _safe_rmtree_under_subset(d, run_name)

    dispatcher_ok, out = _run_experiment(config_dir, run_name, xp_suffix)
    if not dispatcher_ok:
        print(out.strip()[-1500:])
        print(f"  [FAIL] {name}: dispatcher reported a failed experiment")
        return False

    log_ok, log_msg = _check_log_clean(run_name, xp_suffix)
    if not log_ok:
        print(f"  [FAIL] {name}: {log_msg}")
        return False

    pdfs = sorted(
        p for d in _xp_output_dirs(run_name, config_dir, xp_suffix) for p in d.rglob(pdf_glob)
    )
    if not pdfs:
        print(f"  [FAIL] {name}: no PDF matching {pdf_glob!r} under the xp output dir")
        return False
    if require_f01 and not any(p.name.startswith("F01_") for p in pdfs):
        print(f"  [FAIL] {name}: expected an F01_*.pdf, found {[p.name for p in pdfs]}")
        return False

    print(f"  log clean: {log_msg}")
    for p in pdfs[:6]:
        print(f"  PDF: {p.relative_to(REPO_ROOT)}")
    print(f"  [PASS] {name}: {len(pdfs)} PDF(s)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate-A: always-runs, zero-download repro gate.")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME,
                        help="Productions/<run> with run_config + carved subset (default: the in-repo light subset).")
    parser.add_argument("--xp-suffix", default=DEFAULT_XP_SUFFIX, help="xp_suffix for the gate runs.")
    parser.add_argument("--keep-outputs", action="store_true",
                        help="Do not delete the generated xp dirs after the gate (for debugging).")
    parser.add_argument("--llmmap-only", action="store_true",
                        help="Run only PART 1 (the LLMmap E3 replot) — used by `make repro-llmmap-e3`.")
    args = parser.parse_args()

    # Snapshot the regenerable banned-token-pairs cache so we only delete it if the gate created it
    # (never a pre-existing artifact, e.g. when pointed at a real run).
    banned = _banned_token_pairs_path(args.run_name)
    banned_existed_before = banned.exists()

    parts_ok: list[bool] = []
    parts_ok.append(_run_part(
        name="PART 1 (LLMmap E3 replot)", config_dir=VARIANT_CONFIG_DIR,
        run_name=args.run_name, xp_suffix=args.xp_suffix,
        pdf_glob="F01_*.pdf", require_f01=True,
    ))
    if not args.llmmap_only:
        parts_ok.append(_run_part(
            name="PART 2 (light-subset smoke)", config_dir=SMOKE_CONFIG_DIR,
            run_name=args.run_name, xp_suffix=args.xp_suffix,
            pdf_glob="*.pdf", require_f01=False,
        ))

    if not args.keep_outputs:
        for cfg in (VARIANT_CONFIG_DIR, SMOKE_CONFIG_DIR):
            for d in _xp_output_dirs(args.run_name, cfg, args.xp_suffix):
                _safe_rmtree_under_subset(d, args.run_name)
        if banned.exists() and not banned_existed_before:
            banned.unlink()

    ok = all(parts_ok)
    print("\n" + ("=" * 60))
    print(f"gate-A: {'PASS' if ok else 'FAIL'} ({sum(parts_ok)}/{len(parts_ok)} parts passed)")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
