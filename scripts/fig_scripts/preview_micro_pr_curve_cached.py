"""Regenerate the closed-set micro P/R-vs-confidence figures fast from a cached file.

Reads a ``micro_pr_curve_cache.pkl`` produced by ``compute_micro_pr_curve`` when the
``micro_pr_curve_cache`` config flag is set, and redraws the per-bs
``{group}_{n_splits}_splits_micro_pr_curve_bs_{bs}.pdf`` figures. This skips the
multi-GB checkpoint load, the ``load_full_probas`` reads, and the threshold×sample
triple loop entirely, so plot-styling edits in
``audit_llm.plotting.micro_pr_curves`` regenerate in well under a second.

It draws the *real* curves from cached score arrays, using the same
``save_micro_pr_curve_figure`` the live run calls — so the preview and the in-place
figures cannot diverge.

Usage::

    python scripts/fig_scripts/preview_micro_pr_curve_cached.py --cache <path/to/micro_pr_curve_cache.pkl>
    python scripts/fig_scripts/preview_micro_pr_curve_cached.py --cache <path> --out <dir>

The cache is portable: build it once on the figure server, copy the (few-KB) file
locally, and iterate on the plot code here. Pass ``--out`` the real ModelWiseTables
dir to regenerate the figures in place.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from audit_llm.plotting.micro_pr_curves import save_micro_pr_curve_figure


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        required=True,
        help="Path to micro_pr_curve_cache.pkl (written when micro_pr_curve_cache is set).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root / "tmp" / "micro_pr_curve_preview",
        help="Directory to write the figures into (default: tmp/micro_pr_curve_preview/). "
        "Pass the real ModelWiseTables dir to regenerate the figures in place.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.cache.exists():
        raise SystemExit(
            f"Cache not found: {args.cache}\n"
            "Build it by running the XP once with `micro_pr_curve_cache: true` in the "
            "classification config; it writes micro_pr_curve_cache.pkl into each "
            "ModelWiseTables/<effective_key> dir."
        )

    payload = joblib.load(args.cache)
    pr_curve_data = payload["pr_curve_data"]
    n_splits = payload["n_splits"]
    effective_key = payload.get("effective_key")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Loaded cache {args.cache} (n_splits={n_splits}, effective_key={effective_key}, "
        f"batch_sizes={sorted(pr_curve_data)}); writing to {out_dir}"
    )

    for bs in sorted(pr_curve_data):
        pdf_out = save_micro_pr_curve_figure(pr_curve_data[bs], out_dir, n_splits)
        print(f"wrote {pdf_out}")


if __name__ == "__main__":
    main()
