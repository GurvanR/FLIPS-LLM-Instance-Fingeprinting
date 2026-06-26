"""Classifier comparison plots from a saved Batch_Classification XP pickle.

Loads ``train_size_dict[train_size]["all"]`` from a checkpoint pickle written
by ``batch_classification_across_token_pairs`` and produces two 2-panel
bar plots comparing classifiers on micro-averaged accuracy:

- Figure A: bs=1 / tp_wise  vs  bs=8 / mix_tp_at_pred (utp=8)
- Figure B: bs=1 / tp_wise  vs  bs=8 / tp_wise

Each bar = one classifier. Height = mean over splits of (mean over
token-pairs or uplets of per-split overall accuracy). Error bar = std
over splits of the same per-split-averaged accuracy.

The per-split overall accuracy is derived from
``summary[bt][bs][tp_or_uplet][clf]["confusion_matrix_all"]`` (a list of
per-split confusion matrices) via ``trace(cm) / cm.sum()``. The raw
per-split accuracy scalars are not stored in the pickle (only mean/std
of metrics are written by ``summarize_metrics``), so CM-derivation is
the only available path.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Order used in figures when --clf-order is not supplied.
DEFAULT_CLF_ORDER: Tuple[str, ...] = (
    "XGBoost",
    "Random Forest",
    "Gradient Boosting",
    "MLP_strong",
    "Logistic Regression",
    "LDA",
    "SVM",
    "KNN",
)


def _fmt_sig(v: float, n_sig: int = 3) -> str:
    """Format ``v`` with exactly ``n_sig`` significant digits, keeping trailing
    zeros (unlike ``f'{x:.{n_sig}g}'`` which strips them)."""
    if not np.isfinite(v) or v == 0:
        return "0." + "0" * max(0, n_sig - 1)
    from math import floor, log10
    decimals = max(0, n_sig - 1 - int(floor(log10(abs(v)))))
    return f"{v:.{decimals}f}"


def _fmt_pct_3sig(v: float) -> str:
    """Format ``v`` (a fraction in [0, 1]) as a percentage with 3 sig figs."""
    return _fmt_sig(v * 100.0, n_sig=3) + "%"


def _per_split_accuracy_from_cms(cms: Sequence[np.ndarray]) -> np.ndarray:
    """Overall accuracy per confusion matrix: trace(cm) / cm.sum()."""
    accs = []
    for cm in cms:
        total = cm.sum()
        if total == 0:
            accs.append(np.nan)
        else:
            accs.append(float(np.trace(cm)) / float(total))
    return np.asarray(accs, dtype=float)


def gather_clf_scores(
    summary: Dict,
    *,
    batch_type_key: str,
    bs: int,
    clfs: Sequence[str],
) -> Dict[str, Dict[str, np.ndarray | float]]:
    """Aggregate per-clf scores from the pickle's nested summary dict.

    Returns ``{clf: {"per_split_means": vec_len_n_splits, "mean": float, "std": float}}``.

    Aggregation:
      1. For each clf, iterate over all (tp or uplet) entries under
         ``summary[batch_type_key][bs]``.
      2. For each entry fetch ``confusion_matrix_all`` (list of n_splits CMs)
         and convert each CM to its overall accuracy.
         → matrix shape (n_tp_or_uplets, n_splits)
      3. Mean across the tp/uplet axis → vector of length n_splits.
      4. Mean and std of that vector.
    """
    if batch_type_key not in summary:
        raise KeyError(
            f"batch_type_key {batch_type_key!r} not in summary; "
            f"available keys: {sorted(summary.keys())}"
        )
    bt_summary = summary[batch_type_key]
    if bs not in bt_summary:
        raise KeyError(
            f"bs={bs} not under {batch_type_key!r}; "
            f"available bs: {sorted(bt_summary.keys())}"
        )
    bs_summary = bt_summary[bs]
    if not bs_summary:
        raise ValueError(
            f"No tp/uplet entries under summary[{batch_type_key!r}][{bs}]"
        )

    out: Dict[str, Dict[str, np.ndarray | float]] = {}
    for clf in clfs:
        per_split_acc_rows: List[np.ndarray] = []  # one row per tp/uplet
        for tp_or_uplet, clf_map in bs_summary.items():
            if clf not in clf_map:
                logger.warning(
                    "clf %r missing under [%s][%d][%s]; skipping that tp/uplet",
                    clf, batch_type_key, bs, tp_or_uplet,
                )
                continue
            cm_list = clf_map[clf].get("confusion_matrix_all")
            if cm_list is None:
                raise KeyError(
                    f"'confusion_matrix_all' missing at "
                    f"summary[{batch_type_key!r}][{bs}][{tp_or_uplet!r}][{clf!r}]. "
                    f"Was the XP run with confusion-matrix recording enabled?"
                )
            per_split_acc_rows.append(_per_split_accuracy_from_cms(cm_list))

        if not per_split_acc_rows:
            raise ValueError(
                f"No data found for clf={clf!r} under [{batch_type_key!r}][{bs}]"
            )

        # Shape: (n_tp_or_uplets, n_splits). Ragged guard:
        lengths = {len(r) for r in per_split_acc_rows}
        if len(lengths) != 1:
            raise ValueError(
                f"Inconsistent n_splits across tp/uplets for clf={clf!r}: "
                f"got lengths {lengths}"
            )
        mat = np.vstack(per_split_acc_rows)
        per_split_means = np.nanmean(mat, axis=0)  # mean over tp/uplet for each split

        # Derive number of classes from the last CM for chance-baseline plotting.
        last_cm = cm_list[-1]
        n_classes = int(last_cm.shape[0]) if last_cm.ndim == 2 else None

        out[clf] = {
            "per_split_means": per_split_means,
            "mean": float(np.nanmean(per_split_means)),
            "std": float(np.nanstd(per_split_means)),
            "n_splits": int(per_split_means.shape[0]),
            "n_tp_or_uplets": int(mat.shape[0]),
            "n_classes": n_classes,
        }
    return out


def _resolve_clf_order(
    scores_left: Dict,
    scores_right: Dict,
    clf_order: Optional[Sequence[str]],
) -> List[str]:
    """Resolve x-axis classifier ordering.

    ``clf_order`` accepts:
      - an explicit sequence of clf names → kept as-is (filtered to present clfs);
      - ``"auto-by-right"`` → sort by ``scores_right[clf]["mean"]`` descending,
        so the right panel is monotonically decreasing left→right; the same
        order is applied to the left panel for direct visual comparison;
      - ``None`` → fall back to ``DEFAULT_CLF_ORDER``.
    Any classifier present but not listed in an explicit order goes to the tail.
    """
    present = set(scores_left.keys()) & set(scores_right.keys())

    if clf_order == "auto-by-right":
        return sorted(
            present, key=lambda c: scores_right[c]["mean"], reverse=True,
        )

    if clf_order is None:
        clf_order = DEFAULT_CLF_ORDER
    ordered = [c for c in clf_order if c in present]
    leftover = sorted(present - set(ordered))
    return ordered + leftover


def plot_grouped_comparison(
    scores_bs1: Dict[str, Dict],
    scores_bs8: Dict[str, Dict],
    *,
    bar_labels: Tuple[str, str],
    out_pdf: Path,
    clf_order: Optional[Sequence[str] | str] = None,
    clf_display_names: Optional[Dict[str, str]] = None,
    chance_baseline: Optional[float] = None,
    figsize: Tuple[float, float] = (8.0, 4.0),
) -> None:
    """Draw a single grouped bar plot (bs=1 vs bs=8 per clf) and save as PDF.

    Two bars per classifier: blue = bs=1 (``scores_bs1``), orange = bs=8
    (``scores_bs8``). ``clf_display_names`` maps internal clf names
    (e.g. ``"MLP_strong"``) to x-tick labels (e.g. ``"MLP"``); missing
    entries default to the clf's own name.
    """
    clfs = _resolve_clf_order(scores_bs1, scores_bs8, clf_order)
    if not clfs:
        raise ValueError("No common classifiers between bs=1 and bs=8 scores.")

    display_names = clf_display_names or {}
    tick_labels = [display_names.get(c, c) for c in clfs]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(clfs))
    width = 0.4

    bs8_offset = -width / 2
    for offset, scores, label, color in [
        (bs8_offset, scores_bs8, bar_labels[1], "tab:orange"),
        (+width / 2, scores_bs1, bar_labels[0], "tab:blue"),
    ]:
        means = np.array([scores[c]["mean"] for c in clfs])
        stds = np.array([scores[c]["std"] for c in clfs])
        std_up = np.minimum(stds, 1.0 - means)
        std_dn = np.minimum(stds, means)
        ax.bar(
            x + offset, means, width,
            yerr=[std_dn, std_up],
            capsize=3, edgecolor="black", linewidth=0.5, alpha=0.85,
            color=color, label=label,
        )

    # Percentage labels on top of bs=8 bars only (3 significant digits).
    for i, c in enumerate(clfs):
        v = scores_bs8[c]["mean"]
        s = scores_bs8[c]["std"]
        # Place above the error bar's upper whisker (clipped to <=1).
        y = min(v + min(s, 1.0 - v) + 0.02, 1.02)
        ax.text(
            i + bs8_offset, y, _fmt_pct_3sig(v),
            ha="center", va="bottom", fontsize=8,
        )

    if chance_baseline is not None and chance_baseline > 0:
        ax.axhline(
            chance_baseline,
            color="red", linestyle="--", linewidth=2.0, alpha=0.9,
            label=f"Random chance (1/{round(1 / chance_baseline)} = {_fmt_sig(chance_baseline, n_sig=2)})",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Classification Accuracy")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=False)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)
    logger.info("Saved %s", out_pdf)


def plot_classifier_comparison(
    checkpoint_pkl_path: Path,
    out_dir: Path,
    *,
    clfs: Sequence[str],
    train_size: int = 40,
    utp: int = 8,
    clf_order: Optional[Sequence[str] | str] = None,
    clf_display_names: Optional[Dict[str, str]] = None,
    calc_item_name: str = "all",
) -> Dict[str, Path]:
    """Load pickle and produce both Figure A and Figure B.

    Returns a dict mapping a short tag ("figA", "figB") to the saved PDF
    path.
    """
    checkpoint_pkl_path = Path(checkpoint_pkl_path)
    out_dir = Path(out_dir)
    loaded = joblib.load(checkpoint_pkl_path)

    # The XP saves `train_size_dict[train_size][calc_item_name]` as the pickle's
    # *value* (see classify_batch.py:217 + line 83). So the top-level keys are
    # batch_types like 'tp_wise', 'mix_tp_at_pred', 'mix_tp_at_pred_utp8' — not
    # train_size. We tolerate either shape so the same script works against
    # callers that pass the wrapped form.
    top_keys = set(loaded.keys()) if isinstance(loaded, dict) else set()
    expected_bt_keys = {"tp_wise", "mix_tp_at_pred", "mix_tp_at_pred_utp8", "mix_tp_at_train"}
    if top_keys & expected_bt_keys:
        summary = loaded
    elif train_size in loaded and calc_item_name in loaded[train_size]:
        summary = loaded[train_size][calc_item_name]
    else:
        raise KeyError(
            f"Could not locate the batch-type summary in the pickle. "
            f"Top-level keys: {sorted(top_keys)}. Expected either a flat summary "
            f"containing one of {sorted(expected_bt_keys)} or "
            f"loaded[{train_size}][{calc_item_name!r}]."
        )

    mix_key = f"mix_tp_at_pred_utp{utp}"
    if mix_key not in summary:
        # Backward-compat: bare "mix_tp_at_pred" key holds identical data when utp_idx == 0.
        if "mix_tp_at_pred" in summary:
            logger.warning(
                "Falling back to bare 'mix_tp_at_pred' key (no %s found); "
                "this is safe when unique_tp_in_mix has a single value.",
                mix_key,
            )
            mix_key = "mix_tp_at_pred"
        else:
            raise KeyError(
                f"Neither {mix_key!r} nor 'mix_tp_at_pred' in summary; "
                f"keys present: {sorted(summary.keys())}"
            )

    # Figure A: tp_wise/bs=1 (blue) vs mix_tp_at_pred/bs=8 (orange)
    figA_bs1 = gather_clf_scores(summary, batch_type_key="tp_wise", bs=1, clfs=clfs)
    figA_bs8 = gather_clf_scores(summary, batch_type_key=mix_key, bs=8, clfs=clfs)
    # Take n_classes from any clf's entry (they're all the same per pickle).
    n_cls = next(iter(figA_bs1.values())).get("n_classes")
    chance = (1.0 / n_cls) if n_cls else None

    figA_path = out_dir / "clf_comparison_tpwise_bs1_vs_mix_bs8.pdf"
    plot_grouped_comparison(
        figA_bs1, figA_bs8,
        bar_labels=(r"$N_t=1$", r"$N_t=8$"),
        out_pdf=figA_path,
        clf_order=clf_order,
        clf_display_names=clf_display_names,
        chance_baseline=chance,
    )

    # Figure B: tp_wise/bs=1 (blue) vs tp_wise/bs=8 (orange)
    figB_bs1 = figA_bs1  # identical gather
    figB_bs8 = gather_clf_scores(summary, batch_type_key="tp_wise", bs=8, clfs=clfs)
    figB_path = out_dir / "clf_comparison_tpwise_bs1_vs_bs8.pdf"
    plot_grouped_comparison(
        figB_bs1, figB_bs8,
        bar_labels=(r"$N_t=1$", r"$N_t=8$"),
        out_pdf=figB_path,
        clf_order=clf_order,
        clf_display_names=clf_display_names,
        chance_baseline=chance,
    )

    return {"figA": figA_path, "figB": figB_path}


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint-pkl", type=Path, required=True,
                   help="Path to .../train_size_checkpoints/<calc_item>/<train_size>/train_size<ts>_<calc_item>.pkl")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Directory where the two PDFs are written.")
    p.add_argument("--train-size", type=int, default=40)
    p.add_argument("--utp", type=int, default=8,
                   help="unique_tp_in_mix value used by the XP (default 8).")
    p.add_argument("--calc-item-name", default="all",
                   help="Calculation item name. With calculations: {token_pairs: token_pairs} this is 'all'.")
    p.add_argument("--clfs", nargs="+", required=True,
                   help="Classifier names (must match keys in CLASSIFIERS_TEMPLATES_MAP).")
    p.add_argument("--clf-order", nargs="*", default=None,
                   help=("Optional explicit x-axis ordering of classifiers, "
                         "OR the single token 'auto-by-right' to sort by descending "
                         "right-bar (bs=8) accuracy."))
    p.add_argument("--rename", action="append", default=[], metavar="CLF=DISPLAY",
                   help=("Override the x-tick label for a classifier. Repeatable, "
                         "e.g. --rename MLP_strong=MLP."))
    return p


def _parse_rename_pairs(pairs: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in pairs:
        if "=" not in raw:
            raise SystemExit(f"--rename value {raw!r} must be CLF=DISPLAY")
        k, v = raw.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_argparser().parse_args()

    # Accept the special sentinel 'auto-by-right' as a single-element --clf-order.
    clf_order: Optional[Sequence[str] | str] = args.clf_order
    if clf_order and len(clf_order) == 1 and clf_order[0] == "auto-by-right":
        clf_order = "auto-by-right"

    paths = plot_classifier_comparison(
        args.checkpoint_pkl,
        args.out_dir,
        clfs=args.clfs,
        train_size=args.train_size,
        utp=args.utp,
        clf_order=clf_order,
        clf_display_names=_parse_rename_pairs(args.rename),
        calc_item_name=args.calc_item_name,
    )
    for tag, path in paths.items():
        print(f"{tag}: {path}")


if __name__ == "__main__":
    main()
