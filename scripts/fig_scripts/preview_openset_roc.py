"""Regenerate the open-set ROC/PR figures fast from a small cached score file.

Reads a ``roc_figscore_cache.pkl`` produced by ``OpenSetClassification`` when the
``openset_fig_cache`` config flag is set, and redraws the four figures
(``roc_curve_unseen_vs_known``, ``pr_vs_confidence_unseen_and_global``,
``roc_curve_alpha``, ``roc_curve_alpha_overlay``), plus the alpha-sweep P/R figure and
the metrics table. This skips the multi-GB ``mix_tp_at_pred_utp*.pkl`` load and the
re-prediction pass entirely, so plot-styling edits in
``audit_llm.plotting.threshold_plots`` regenerate in well under a second.

Usage::

    python scripts/fig_scripts/preview_openset_roc.py --cache <path/to/roc_figscore_cache.pkl>
    python scripts/fig_scripts/preview_openset_roc.py --cache <path> --out <dir>

The cache is portable: build it once on the figure server, copy the (tens-of-MB) file
locally, and iterate on the plot code here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from audit_llm.plotting.threshold_plots import (
    plot_alpha_roc_curves,
    plot_openset_roc_curves,
    plot_roc_curves_overlay,
    plot_unseen_and_global_pr_vs_alpha,
    plot_unseen_and_global_pr_vs_confidence,
    save_openset_metrics_table,
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        required=True,
        help="Path to roc_figscore_cache.pkl (written when openset_fig_cache is set).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root / "tmp" / "openset_roc_preview",
        help="Directory to write the figures into (default: tmp/openset_roc_preview/). "
        "Pass the real batch_type fig dir to regenerate the figures in place.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.cache.exists():
        raise SystemExit(
            f"Cache not found: {args.cache}\n"
            "Build it by running the XP once with `openset_fig_cache: true` in the "
            "classification config; it writes roc_figscore_cache.pkl into each "
            "batch_type fig dir."
        )

    payload = joblib.load(args.cache)
    roc_data = payload["roc_data"]
    pr_curves_by_bs = payload["pr_curves_by_bs"]
    batch_prediction_sizes = payload["batch_prediction_sizes"]
    unseen_prevalence = payload.get("unseen_prevalence")
    alpha = payload.get("alpha")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Loaded cache {args.cache} (alpha={alpha}, "
        f"batch_prediction_sizes={batch_prediction_sizes}); writing to {out_dir}"
    )

    plot_openset_roc_curves(
        roc_data,
        fig_save_path=out_dir,
        batch_prediction_sizes=batch_prediction_sizes,
    )
    plot_unseen_and_global_pr_vs_confidence(
        pr_curves_by_bs,
        fig_save_path=out_dir,
        batch_prediction_sizes=batch_prediction_sizes,
        unseen_prevalence=unseen_prevalence,
    )
    plot_unseen_and_global_pr_vs_alpha(
        pr_curves_by_bs,
        fig_save_path=out_dir,
        batch_prediction_sizes=batch_prediction_sizes,
    )
    plot_alpha_roc_curves(
        pr_curves_by_bs,
        fig_save_path=out_dir,
        batch_prediction_sizes=batch_prediction_sizes,
    )
    plot_roc_curves_overlay(
        pr_curves_by_bs,
        roc_data,
        fig_save_path=out_dir,
        batch_prediction_sizes=batch_prediction_sizes,
    )
    save_openset_metrics_table(
        roc_data,
        fig_save_path=out_dir,
        batch_prediction_sizes=batch_prediction_sizes,
    )

    for f in sorted(out_dir.glob("*.pdf")):
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
