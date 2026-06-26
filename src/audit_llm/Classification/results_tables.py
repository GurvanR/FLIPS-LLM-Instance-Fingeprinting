"""
Statistical computations and table generation for classification results.

Provides functions for calculating mean/std statistics, weighted averages for open-set
scenarios, NxM table statistics, and orchestrating table/heatmap creation.
"""

import json
from pathlib import Path
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import logging
logger = logging.getLogger(__name__)

from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from audit_llm.Classification.batch_results import load_full_probas
from audit_llm.Classification.training_size_analysis import _parse_utp_suffix, _utp_sort_key
from audit_llm.file_io import QUANTIZATION_SEPARATOR
from audit_llm.plot_configs import (
    GRID_CONFIG,
    LEGEND_CONFIG,
    XLABEL_CONFIG,
    XTICKS_CONFIG,
    YLABEL_CONFIG,
)
from audit_llm.plotting.heatmaps import make_accuracy_heatmap
from audit_llm.plotting.micro_pr_curves import (
    plot_micro_pr_curve_on_ax,
    save_micro_pr_curve_figure,
)
from audit_llm.plotting.tables import (
    make_accuracy_table_latex,
    make_accuracy_table_markdown,
    make_accuracy_table_nxm_latex,
)
from audit_llm.xp_tools.model_filtering import (
    full_var_model_name_to_var_name,
    group_models_idx_by_var_or_orig,
)
from audit_llm.xp_tools import get_token_pairs_of_group, get_tp_names_of_group


def get_effective_batch_type_keys(summary_dict: dict, batch_types: list) -> list:
    """Return iteration keys, expanding mixing batch_types into their utp-specific keys.

    For batch_types with utp-specific entries (e.g. 'mix_tp_at_train_utp2'),
    those are returned instead of the bare batch_type key (backward compat duplicate).
    """
    all_keys = set(summary_dict.keys())
    effective = []
    for bt in batch_types:
        utp_keys = sorted(k for k in all_keys if k.startswith(f"{bt}_utp"))
        if utp_keys:
            effective.extend(utp_keys)
        elif bt in all_keys:
            effective.append(bt)
    return effective


def _select_best_utp_keys(
    effective_keys: list,
    summary_dict: dict,
    clf: str,
    datasets: list,
    metric: str = "accuracy",
) -> list:
    """From a list of effective keys, keep only the best utp key per mix mode.

    Non-utp keys (e.g. ``tp_wise``) are always kept.  For utp-specific keys
    (e.g. ``mix_tp_at_pred_utp2``, ``mix_tp_at_pred_utp4``), only the one with
    the highest mean metric across FLiPS datasets at batch_size=1 fallback to
    the max available batch size is kept.
    """
    # Separate non-utp keys from utp keys grouped by prefix
    kept = []
    utp_groups: Dict[str, list] = {}  # prefix -> list of (key, utp_val)
    for key in effective_keys:
        for prefix in ("mix_tp_at_pred_utp", "mix_tp_at_train_utp"):
            if key.startswith(prefix):
                utp_groups.setdefault(prefix, []).append(key)
                break
        else:
            kept.append(key)

    # For each mix prefix, keep the key with the highest utp value
    # ('max' is the loosest constraint → treated as the highest by _utp_sort_key)
    for prefix, keys in utp_groups.items():
        best_key = max(keys, key=lambda k: _utp_sort_key(_parse_utp_suffix(k, prefix)))
        kept.append(best_key)
        logger.info("Highest utp key for %s: %s", prefix.rstrip("_utp"), best_key)

    return kept


def _get_tp_names_for_key(summary_dict, effective_key, bs, tp_group_name, datasets):
    """Get dataset names for an effective key and batch size.

    For tp_wise: uses get_tp_names_of_group (standard group logic).
    For mix modes: extracts tp_names directly from the summary dict keys,
    filtered to keep only uplets whose constituent token pairs belong to the group.
    Returns (tp_names, summary_key) where summary_key is the key to use in the summary dict.
    """
    if effective_key == "tp_wise" or bs == 1:
        tp_names = get_tp_names_of_group(tp_group_name, mode="tp_wise", bs=bs, token_pairs=datasets)
        return tp_names, "tp_wise"

    # For mix modes, get tp_names from the actual summary keys
    if effective_key not in summary_dict or bs not in summary_dict[effective_key]:
        return [], effective_key

    all_tp_names = list(summary_dict[effective_key][bs].keys())

    # Filter by tp_group membership: keep uplets where at least one constituent tp is in the group
    group_tps = set(get_token_pairs_of_group(tp_group_name, token_pairs=datasets))
    if not group_tps:
        return [], effective_key

    filtered = [uplet for uplet in all_tp_names if any(tp in uplet for tp in group_tps)]
    return filtered, effective_key


def _save_auroc_agg_markdown(bs_group_aurocs_agg, save_tables_path, group_name, n_splits):
    """Write a markdown table of macro/micro AUROC and macro/micro Precision & Recall per batch size.

    Precision & Recall are derived from hard argmax predictions (threshold = 0, all predictions kept).
    macro = unweighted average over classes; micro = globally pooled TP/FP/FN.
    """
    lines = [
        "| Batch Size | Macro AUROC | Micro AUROC | Macro Precision | Micro Precision | Macro Recall | Micro Recall |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    def fmt(v):
        if v is None:
            return "N/A"
        m, ci = v["means"], v["CI"]
        return f"**{m:.4f}** (±{ci:.4f})" if not np.isnan(m) else "NaN"

    for bs in sorted(bs_group_aurocs_agg.keys()):
        label = "Single Query" if bs == 1 else f"{bs}-queries"
        d = bs_group_aurocs_agg[bs]
        lines.append(
            f"| {label} "
            f"| {fmt(d.get('macro'))} "
            f"| {fmt(d.get('micro'))} "
            f"| {fmt(d.get('macro_precision'))} "
            f"| {fmt(d.get('micro_precision'))} "
            f"| {fmt(d.get('macro_recall'))} "
            f"| {fmt(d.get('micro_recall'))} |"
        )
    out = Path(save_tables_path) / f"{group_name}_{n_splits}_splits_auroc_agg.md"
    out.write_text("\n".join(lines), encoding="utf-8")


def _alpha_str_for_path(alpha: float) -> str:
    """Mirror of openset_classification._alpha_str for path reconstruction.

    0.05 → "0_05"; 0.1 → "0_1".
    """
    s = f"{alpha:.4f}".rstrip("0").rstrip(".") or "0"
    return s.replace(".", "_")


def _parse_unseen_pr_from_md(md_path: Path, clf: str) -> dict | None:
    """Parse precision/recall for ``clf`` from an openset_metrics_bs_*.md file.

    Format written by save_openset_metrics_table:
        | Classifier | Precision (Unseen) | Recall (Unseen) | F1 (Unseen) | AUROC |
        |:---:|:---:|:---:|:---:|:---:|
        | XGBoost | 0.823 ± 0.045 | 0.756 ± 0.034 | ... | ... |
    Returns ``{"precision": float, "recall": float}`` or None.
    """
    try:
        text = md_path.read_text()
    except OSError as exc:
        logger.warning("Could not read %s: %s", md_path, exc)
        return None
    for line in text.splitlines():
        if not line.startswith("|") or ":---" in line or "Classifier" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3 or cells[0] != clf:
            continue
        try:
            precision = float(cells[1].split("±")[0].strip())
            recall = float(cells[2].split("±")[0].strip())
        except (ValueError, IndexError):
            return None
        return {"precision": precision, "recall": recall}
    return None


def _load_unseen_pr_from_disk(
    fig_save_path: Path,
    calc_item,
    train_size,
    effective_key: str,
    clf: str,
    batch_sizes: list[int],
    alpha: float,
) -> dict | None:
    """Load precision/recall sidecar JSONs written by save_openset_metrics_table.

    Returns ``{bs: {"precision": ..., "recall": ...}}`` for the given clf, or
    ``None`` if nothing was found. Falls back to parsing the .md table when
    the JSON sidecar is missing (pre-existing experiments).
    """
    alpha_dir = f"alpha_{_alpha_str_for_path(alpha)}"
    base = (
        Path(fig_save_path) / "train_size_checkpoints" / str(calc_item)
        / str(train_size) / effective_key / alpha_dir
    )
    if not base.exists():
        return None
    out: dict = {}
    for bs in batch_sizes:
        json_path = base / f"openset_metrics_bs_{bs}.json"
        entry: dict | None = None
        if json_path.exists():
            try:
                payload = json.loads(json_path.read_text())
                clf_entry = payload.get(clf)
                if clf_entry is not None:
                    entry = {
                        "precision": clf_entry["mean_precision"],
                        "recall": clf_entry["mean_recall"],
                    }
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Could not parse %s: %s", json_path, exc)
        if entry is None:
            md_path = base / f"openset_metrics_bs_{bs}.md"
            if md_path.exists():
                entry = _parse_unseen_pr_from_md(md_path, clf)
        if entry is not None:
            out[bs] = entry
    return out or None


def make_model_wise_tables(
    train_size_dict: Dict[float, Dict[float, Dict[any, any]]],
    fig_save_path: Path,
    classification_config: Dict,
    datasets: list[str],
    models_idx: Dict,
    batch_sizes: list[int] = [1, 2, 3, 4, 5, 8],
    tp_group_names: list[str] = ["FLiPS", "0-1"],
    skip_nxm: bool = False,
    openset_roc_data: dict | None = None,
) -> None:
    """
    Plot metric curves across test sizes for given calculation_items, classifiers, batch sizes, and batch type.

    Args:
        train_size_dict: mapping train_size -> { calculation_item -> pipe_summary_dict }
        metrics: list of metric names to plot (e.g. ['accuracy', 'f1'])
        batch_sizes: list of batch sizes to include in curves
        batch_type: one of 'mix_tp_at_pred', 'tp_wise', or 'across_and_tp_wise'
        fig_save_path: directory path to save figures
        show: whether to display figures interactively

    For each calculation_item, for each classifier, and for each metric, produces a plot
    where each batch_size in batch_sizes is a separate curve (mean ± std error bars).
    For 'across_and_tp_wise' batch_type, also creates tp_wise plots for each batch_size.
    Figures are saved via save_fig_and_show().

    Info on summary objects:
        summary[key_name][batch_size][tp][clf][f'{metric}_mean'] = np.mean(vals)
        summary[key_name][batch_size][tp][clf][f'{metric}_std'] = np.std(vals)

        where key_name in ['tp_wise', 'mix_tp_at_pred']
        where vals are values of the metrics.

    """

    batch_types = classification_config.get("batch_types") or ["tp_wise"]
    n_splits: int = classification_config["n_splits"]
    if classification_config.get("openset", False):
        models_idx = models_idx | {len(models_idx): "Unseen"}

    train_sizes = sorted(train_size_dict.keys())
    calculation_items = sorted({t for ts in train_size_dict.values() for t in ts.keys()})
    for calc_item in calculation_items:
        for clf in classification_config["classifiers"]:
            for train_size in train_sizes:
                batch_type_dict = train_size_dict[train_size][calc_item]
                effective_keys = get_effective_batch_type_keys(batch_type_dict, batch_types)
                effective_keys = _select_best_utp_keys(effective_keys, batch_type_dict, clf, datasets)

                for effective_key in effective_keys:
                    save_tables_path = Path(fig_save_path) / str(calc_item) / str(clf) / str(train_size) / "ModelWiseTables" / effective_key
                    save_tables_path.mkdir(exist_ok=True, parents=True)

                    # Build per-bs precision/recall for this clf from openset roc_data.
                    # In-memory roc_data is only populated on fresh runs; on checkpoint
                    # reloads it's None, so fall back to the JSON sidecar that
                    # save_openset_metrics_table writes next to the .md table.
                    unseen_pr_by_bs: dict | None = None
                    if openset_roc_data:
                        unseen_pr_by_bs = {
                            bs: {
                                "precision": openset_roc_data[bs][clf]["mean_precision"],
                                "recall": openset_roc_data[bs][clf]["mean_recall"],
                            }
                            for bs in openset_roc_data
                            if clf in openset_roc_data[bs]
                        } or None
                    elif classification_config.get("openset", False):
                        unseen_pr_by_bs = _load_unseen_pr_from_disk(
                            fig_save_path=fig_save_path,
                            calc_item=calc_item,
                            train_size=train_size,
                            effective_key=effective_key,
                            clf=clf,
                            batch_sizes=batch_sizes,
                            alpha=classification_config.get("alpha_quantile_threshold", 0.05),
                        )

                    for tp_group_name in tp_group_names:
                        bs_group_accuracies, bs_group_accuracies_extra = compute_group_accuracies(
                            batch_type_dict, datasets, effective_key, batch_sizes, tp_group_name, clf
                        )

                        is_openset = classification_config.get("openset", False)

                        # Compute the micro-PR curve FIRST so its per-bs data can be
                        # embedded into the heatmap's blank quantization region below.
                        pr_curve_data_by_bs: dict | None = None
                        if (
                            classification_config.get("compute_micro_pr_curve", False)
                            and classification_config.get("store_prediction_probas", False)
                            and not is_openset
                        ):
                            pr_curve_data_by_bs = compute_micro_pr_curve(
                                batch_type_dict, datasets, effective_key, batch_sizes,
                                tp_group_name, clf, save_tables_path, n_splits,
                                cache=classification_config.get("micro_pr_curve_cache", False),
                            )

                        if bs_group_accuracies:
                            create_and_save_tables(
                                bs_group_accuracies,
                                effective_key,
                                models_idx,
                                tp_group_name,
                                save_tables_path,
                                n_splits,
                                bs_accuracies_extra=bs_group_accuracies_extra,
                                skip_nxm=skip_nxm,
                                unseen_pr_by_bs=unseen_pr_by_bs,
                                pr_curve_data_by_bs=pr_curve_data_by_bs,
                            )
                        else:
                            logger.warning(
                                "No accuracies computed for %s/%s at calc_item %s, clf %s, train_size %s",
                                tp_group_name, effective_key, calc_item, clf, train_size,
                            )

                        if classification_config.get("store_prediction_probas", False):
                            bs_group_aurocs, bs_group_aurocs_agg = compute_group_aurocs(
                                batch_type_dict, datasets, effective_key, batch_sizes, tp_group_name, clf
                            )
                            if bs_group_aurocs:
                                create_and_save_tables(
                                    bs_group_aurocs,
                                    effective_key,
                                    models_idx,
                                    tp_group_name + "_AUROC",
                                    save_tables_path,
                                    n_splits,
                                    skip_nxm=skip_nxm,
                                )
                            if bs_group_aurocs_agg:
                                _save_auroc_agg_markdown(
                                    bs_group_aurocs_agg, save_tables_path, tp_group_name, n_splits
                                )


def compute_group_accuracies(summary_dict, datasets, effective_key, batch_sizes, tp_group_name, clf):
    """
    Compute aggregated accuracies across multiple datasets in a group.

    Args:
        summary_dict: Dict with batch_type keys at top level.
            Structure: summary_dict[batch_type_key][bs][tp][clf][metric]
        datasets: list of all token pair names
        effective_key: the batch_type key to use (e.g. 'tp_wise', 'mix_tp_at_train_utp2')
        batch_sizes: list of batch sizes
        tp_group_name: 'FLiPS' or '0-1'
        clf: classifier name

    Returns (bs_group_accuracies, {})
    """
    bs_group_accuracies = {}

    for bs in batch_sizes:
        group_names, mode = _get_tp_names_for_key(summary_dict, effective_key, bs, tp_group_name, datasets)

        if not group_names:
            continue

        logger.debug("group_names = %s in compute_group_accuracies (mode=%s)", group_names, mode)

        if mode not in summary_dict:
            logger.warning("Mode '%s' not found in summary_dict keys %s, skipping bs=%s", mode, list(summary_dict.keys()), bs)
            continue

        batch_size_confusion_matrix_map = summary_dict[mode]

        if bs not in batch_size_confusion_matrix_map:
            logger.warning("Batch size %s not in summary for mode '%s', skipping", bs, mode)
            continue

        validate_confusion_matrices(batch_size_confusion_matrix_map, [bs], group_names, [clf])

        all_group_accuracies = []
        if len(group_names) > 1:
            for tp in group_names:
                if tp in batch_size_confusion_matrix_map[bs] and clf in batch_size_confusion_matrix_map[bs][tp]:
                    mean_cms = batch_size_confusion_matrix_map[bs][tp][clf]["confusion_matrix_mean"]
                    per_class_acc = np.diag(mean_cms) / mean_cms.sum(axis=1)
                    all_group_accuracies.append(per_class_acc)
        else:
            assert len(group_names) == 1, "group_names should have at least one dataset."
            tp = group_names[0]
            if tp in batch_size_confusion_matrix_map[bs] and clf in batch_size_confusion_matrix_map[bs][tp]:
                all_cms = batch_size_confusion_matrix_map[bs][tp][clf]["confusion_matrix_all"]
                for cm in all_cms:
                    per_class_acc = np.diag(cm) / cm.sum(axis=1)
                    all_group_accuracies.append(per_class_acc)

        if all_group_accuracies:
            all_group_accuracies = np.vstack(all_group_accuracies)
            logger.debug("all_group_accuracies.shape = %s", all_group_accuracies.shape)
            bs_group_accuracies[bs] = {
                "means": np.nanmean(all_group_accuracies, axis=0),
                "CI": 1.96 * np.nanstd(all_group_accuracies, axis=0) / np.sqrt(len(all_group_accuracies)),
            }

    return bs_group_accuracies, {}


def compute_group_aurocs(summary_dict, datasets, effective_key, batch_sizes, tp_group_name, clf):
    """Per-class one-vs-rest AUROC, macro/micro AUROC, and macro/micro Precision & Recall.

    Requires full_probas and full_y_true in probs_save_map
    (set store_prediction_probas: true in classification_config).

    Precision & Recall in the aggregated dict are derived from hard argmax predictions
    (i.e. threshold = 0, every sample gets a prediction). macro = unweighted mean over
    classes; micro = globally pooled TP/FP/FN counts.

    Not called for open-set experiments (full_y_true may contain -1).
    """
    bs_group_aurocs = {}
    bs_group_aurocs_agg = {}

    for bs in batch_sizes:
        group_names, mode = _get_tp_names_for_key(
            summary_dict, effective_key, bs, tp_group_name, datasets
        )
        if not group_names or mode not in summary_dict:
            continue

        bs_data = summary_dict[mode].get(bs, {})
        all_aurocs = []
        all_macro_aurocs = []
        all_micro_aurocs = []
        all_macro_precisions = []
        all_micro_precisions = []
        all_macro_recalls = []
        all_micro_recalls = []

        for tp in group_names:
            if tp not in bs_data or clf not in bs_data[tp]:
                continue
            psm = bs_data[tp][clf].get("probs_save_map", {})
            full_probas, full_y_true = load_full_probas(psm)

            if full_probas is None or full_y_true is None:
                continue

            n_classes = full_probas.shape[1]

            # --- Per-class one-vs-rest AUROC ---
            per_class_auroc = np.full(n_classes, np.nan)
            for k in range(n_classes):
                y_bin = (full_y_true == k).astype(int)
                if 0 < y_bin.sum() < len(y_bin):
                    per_class_auroc[k] = roc_auc_score(y_bin, full_probas[:, k])
            all_aurocs.append(per_class_auroc)

            # --- Macro AUROC ---
            try:
                macro_auroc = roc_auc_score(full_y_true, full_probas, multi_class='ovr', average='macro')
            except ValueError:
                macro_auroc = np.nan
            all_macro_aurocs.append(macro_auroc)

            # --- Micro AUROC ---
            try:
                y_true_bin = label_binarize(full_y_true, classes=list(range(n_classes)))
                micro_auroc = roc_auc_score(y_true_bin, full_probas, average='micro')
            except ValueError:
                micro_auroc = np.nan
            all_micro_aurocs.append(micro_auroc)

            # --- Precision & Recall from hard argmax predictions (threshold = 0) ---
            y_pred_hard = np.argmax(full_probas, axis=1)
            try:
                all_macro_precisions.append(
                    precision_score(full_y_true, y_pred_hard, average='macro', zero_division=0)
                )
                all_micro_precisions.append(
                    precision_score(full_y_true, y_pred_hard, average='micro', zero_division=0)
                )
                all_macro_recalls.append(
                    recall_score(full_y_true, y_pred_hard, average='macro', zero_division=0)
                )
                all_micro_recalls.append(
                    recall_score(full_y_true, y_pred_hard, average='micro', zero_division=0)
                )
            except Exception as exc:
                logger.warning("Could not compute precision/recall for %s bs=%s: %s", tp, bs, exc)

        if all_aurocs:
            arr = np.vstack(all_aurocs)
            bs_group_aurocs[bs] = {
                "means": np.nanmean(arr, axis=0),
                "CI": 1.96 * np.nanstd(arr, axis=0) / np.sqrt(len(arr)),
            }

        def _agg(vals):
            """Return {means, CI} dict or None if vals is empty."""
            if not vals:
                return None
            a = np.array(vals)
            n = len(a)
            return {"means": np.nanmean(a), "CI": 1.96 * np.nanstd(a) / np.sqrt(n)}

        if all_macro_aurocs or all_micro_aurocs:
            bs_group_aurocs_agg[bs] = {
                "macro":           _agg(all_macro_aurocs),
                "micro":           _agg(all_micro_aurocs),
                "macro_precision": _agg(all_macro_precisions),
                "micro_precision": _agg(all_micro_precisions),
                "macro_recall":    _agg(all_macro_recalls),
                "micro_recall":    _agg(all_micro_recalls),
            }

    return bs_group_aurocs, bs_group_aurocs_agg


def compute_micro_pr_curve(
    summary_dict, datasets, effective_key, batch_sizes, tp_group_name, clf, save_tables_path, n_splits,
    cache=False,
):
    """Micro-averaged precision & recall vs confidence threshold.

    Follows the algorithm in Documentation/micro_precision_recall_vs_threshold.md:
    - For each threshold t in [0, 1]:
      - y_pred[i] = argmax(probas[i]);  conf[i] = max(probas[i])
      - conf[i] >= t and y_pred correct → TP
      - conf[i] >= t and y_pred wrong   → FP + FN
      - conf[i] <  t                    → FN (rejected = missed)
      - precision(t) = TP / (TP + FP),  recall(t) = TP / (TP + FN)

    Saves:
    - ``{group}_{n_splits}_splits_micro_pr_curve.md``  — markdown table
    - ``{group}_{n_splits}_splits_micro_pr_curve.pdf`` — line plot

    Only called for closed-set experiments (openset: false).
    Requires store_prediction_probas: true.

    Returns a dict ``{bs: {thresholds, mean_prec, std_prec, mean_rec, std_rec,
    tp_group_name, bs}}`` for each bs that produced a curve PDF, or ``None``
    if no data was available. The returned arrays are reused to embed the
    curve inside the NxM heatmap's blank quantization region.
    """
    import matplotlib
    matplotlib.use("Agg")  # safe for headless HPC runs
    import matplotlib.pyplot as plt

    thresholds = np.linspace(0, 1, 101)
    # Accumulate (precision, recall) arrays across all tp/bs, then average
    pr_per_bs: dict = {}  # bs -> list of (precisions_array, recalls_array)

    for bs in batch_sizes:
        group_names, mode = _get_tp_names_for_key(
            summary_dict, effective_key, bs, tp_group_name, datasets
        )
        if not group_names or mode not in summary_dict:
            continue

        bs_data = summary_dict[mode].get(bs, {})

        for tp in group_names:
            if tp not in bs_data or clf not in bs_data[tp]:
                continue
            psm = bs_data[tp][clf].get("probs_save_map", {})
            full_probas, full_y_true = load_full_probas(psm)

            if full_probas is None or full_y_true is None:
                continue

            y_pred = np.argmax(full_probas, axis=1)   # (N,)
            conf = np.max(full_probas, axis=1)         # (N,)
            n_samples = len(full_y_true)

            precisions = []
            recalls = []
            for t in thresholds:
                TP = FP = FN = 0
                for i in range(n_samples):
                    if conf[i] >= t:
                        if y_pred[i] == full_y_true[i]:
                            TP += 1
                        else:
                            FP += 1
                            FN += 1
                    else:
                        FN += 1
                p = TP / (TP + FP) if (TP + FP) > 0 else 1.0
                r = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                precisions.append(p)
                recalls.append(r)

            pr_per_bs.setdefault(bs, []).append(
                (np.array(precisions), np.array(recalls))
            )

    if not pr_per_bs:
        logger.warning(
            "compute_micro_pr_curve: no data found for %s / %s", tp_group_name, effective_key
        )
        return None

    # ---- Build averaged curves ----
    # For the MD table and plot we average over datasets within each bs,
    # then show all batch sizes together.
    save_tables_path = Path(save_tables_path)

    # --- Markdown table ---
    md_lines = ["# Micro-averaged Precision & Recall vs Confidence Threshold\n"]
    all_bs_sorted = sorted(pr_per_bs.keys())

    # Header — one set of P/R columns per batch size
    bs_labels = ["Single Query" if bs == 1 else f"{bs}-queries" for bs in all_bs_sorted]
    header_cells = ["| Threshold |"]
    sep_cells = ["| :---: |"]
    for lbl in bs_labels:
        header_cells.append(f" Precision ({lbl}) | Recall ({lbl}) |")
        sep_cells.append(" :---: | :---: |")
    md_lines.append("".join(header_cells))
    md_lines.append("".join(sep_cells))

    mean_curves: dict = {}  # bs -> (mean_prec, mean_rec)
    for bs in all_bs_sorted:
        pairs = pr_per_bs[bs]
        mean_prec = np.mean([p for p, _ in pairs], axis=0)
        mean_rec  = np.mean([r for _, r in pairs], axis=0)
        mean_curves[bs] = (mean_prec, mean_rec)

    for j, t in enumerate(thresholds):
        row = f"| {t:.2f} |"
        for bs in all_bs_sorted:
            mp, mr = mean_curves[bs]
            row += f" {mp[j]:.4f} | {mr[j]:.4f} |"
        md_lines.append(row)

    md_out = save_tables_path / f"{tp_group_name}_{n_splits}_splits_micro_pr_curve.md"
    md_out.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Saved micro P/R curve table → %s", md_out)

    # --- PDF plots — one figure per query regime, aligned with the openset
    # plot_unseen_and_global_pr_vs_confidence style: dashed P / solid R in blue,
    # variance band across (tp, iteration), saturation marker lines at
    # P=0.99, 0.999, 0.9999 in light-to-dark blue with rotated annotations.
    # No grey lines (closed set has no Unseen prevalence / α threshold to mark).
    bs_single = 1
    bs_max = max(all_bs_sorted)
    subplot_bs = sorted({bs for bs in (bs_single, bs_max) if bs in mean_curves})

    pr_curve_data: dict[int, dict] = {}

    for bs in subplot_bs:
        pairs = pr_per_bs[bs]
        prec_stack = np.stack([p for p, _ in pairs], axis=0)
        rec_stack = np.stack([r for _, r in pairs], axis=0)
        mean_prec = prec_stack.mean(axis=0)
        std_prec = prec_stack.std(axis=0)
        mean_rec = rec_stack.mean(axis=0)
        std_rec = rec_stack.std(axis=0)

        pr_curve_data[bs] = {
            "thresholds": thresholds,
            "mean_prec": mean_prec,
            "std_prec": std_prec,
            "mean_rec": mean_rec,
            "std_rec": std_rec,
            "tp_group_name": tp_group_name,
            "bs": bs,
        }

        pdf_out = save_micro_pr_curve_figure(pr_curve_data[bs], save_tables_path, n_splits)
        logger.info("Saved micro P/R curve plot (bs=%s) → %s", bs, pdf_out)

    # --- FMR / FNMR plot ---
    fig2, axes2 = plt.subplots(1, len(subplot_bs), figsize=(5 * len(subplot_bs), 4), sharey=True)
    if len(subplot_bs) == 1:
        axes2 = [axes2]

    for ax2, bs in zip(axes2, subplot_bs):
        lbl = "Single Query" if bs == 1 else f"{bs}-queries"
        mp, mr = mean_curves[bs]
        ax2.plot(thresholds, 1 - mp, label="FMR (1−Precision)", color="steelblue")
        ax2.plot(thresholds, 1 - mr, label="FNMR (1−Recall)", color="darkorange", linestyle="--")
        ax2.set_xlabel("Confidence threshold")
        ax2.set_ylabel("Error rate")
        ax2.set_title(lbl)
        ax2.legend(fontsize=8)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)

    fig2.suptitle(f"FMR / FNMR vs Threshold — {tp_group_name} ({effective_key})", fontsize=11)
    fig2.tight_layout()

    pdf_out2 = save_tables_path / f"{tp_group_name}_{n_splits}_splits_micro_fmr_fnmr_curve.pdf"
    fig2.savefig(pdf_out2, bbox_inches="tight")
    plt.close(fig2)
    logger.info("Saved FMR/FNMR curve plot → %s", pdf_out2)

    if cache:
        cache_path = save_tables_path / "micro_pr_curve_cache.pkl"
        joblib.dump(
            {
                "pr_curve_data": pr_curve_data,
                "tp_group_name": tp_group_name,
                "n_splits": n_splits,
                "effective_key": effective_key,
            },
            cache_path,
            compress=3,
        )
        logger.info("Saved micro P/R curve fig cache → %s", cache_path)

    return pr_curve_data


def get_simple_stats(data_dict, models_idx, reverted_extra_idx=None):
    """
    Calculates Mean/Std for standard (non-NxM) tables.
    Returns: {batch_size: {'mean': float, 'std': float}}
    """
    if not data_dict:
        return None

    stats = {}
    all_indices = list(models_idx.keys())

    for bs, values in data_dict.items():
        means = values["means"]
        cis = values["CI"]

        # If this is 'extra' data, the indices might need mapping (e.g. finding 'Unseen' idx)
        # But for simple averaging across "all models provided", we usually just take all available indices.
        # However, to be strict about the weighting logic, we pass the indices 0..N-1

        # Note: If reverted_extra_idx is provided, we are processing extra data.
        # We assume data_dict aligns with models_idx logic or has its own structure.
        # To simplify, we calculate stats over *all* values present in the arrays.
        current_indices = range(len(means))

        m, s = compute_weighted_average(current_indices, means, cis, models_idx)
        stats[bs] = {"mean": m, "std": s}

    return stats


def compute_weighted_average(indices, all_means, all_cis, global_models_idx):
    """
    Computes weighted mean/std for a subset of model indices.

    Logic:
    - If the global 'models_idx' indicates Open Set mode (last class is "Unseen"),
      that "Unseen" class gets a weight of (N_seen / 2).
    - All other classes get a weight of 1.
    - If "Unseen" is not in the passed 'indices', standard average is returned.
    """
    if not indices:
        return np.nan, np.nan

    # 1. Extract values for the subset
    subset_means = np.array([all_means[i] for i in indices])
    subset_cis = np.array([all_cis[i] for i in indices])

    # 2. Determine Weights
    # Check if global configuration implies Open Set
    labels = list(global_models_idx.values())
    is_open_set = labels[-1] == "Unseen"

    weights = np.ones(len(indices))

    if is_open_set:
        # N_seen is total classes - 1 (the unseen one)
        n_seen = len(labels) - 1
        open_proportion = 0.1
        weight_unseen = (open_proportion * n_seen) / (
            1 - open_proportion
        )  # so that unseen counts as much as half the seen classes combined.

        # Apply weight only if the specific index corresponds to "Unseen"
        for i, original_idx in enumerate(indices):
            # We check the label of the original index
            if global_models_idx[original_idx] == "Unseen":
                logger.debug("weighting unseen")
                weights[i] = weight_unseen

    # 3. Weighted Average (Masking NaNs)
    def w_avg(arr, w):
        mask = ~np.isnan(arr)
        if not np.any(mask):
            return np.nan
        return np.average(arr[mask], weights=w[mask])

    return w_avg(subset_means, weights), w_avg(subset_cis, weights)


def get_nxm_stats(bs, current_bs_accuracies, models_idx):
    """
    Calculates Matrix, Row Avgs, Col Avgs, and Grand Avg for NxM tables.
    """
    if bs not in current_bs_accuracies:
        return None

    means = current_bs_accuracies[bs]["means"]
    cis = current_bs_accuracies[bs]["CI"]
    # Grouping (Assumes external helpers exist)
    orig_groups = group_models_idx_by_var_or_orig(models_idx, group_by="orig")
    var_groups = group_models_idx_by_var_or_orig(models_idx, group_by="var")

    # Sort keys
    orig_labels = sorted(orig_groups.keys())
    var_names = sorted(var_groups.keys())
    # if 'Unseen' in var_names:
    if "ablit" in var_names:
        var_names.remove("ablit")
        var_names.append("ablit")

    stats = {
        "orig_labels": orig_labels,
        "var_names": var_names,
        "matrix": {},  # (orig, var) -> {mean, std}
        "row_avgs": {},  # orig -> {mean, std}
        "col_avgs": {},  # var -> {mean, std}
        "grand_avg": {},  # {mean, std}
    }

    # A. Fill Matrix & Calculate Row Averages
    all_valid_indices = []

    for orig in orig_labels:
        row_indices = []
        for var in var_names:
            # Find index for this specific (orig, var) combo
            idx = next((i for i, name in orig_groups[orig] if full_var_model_name_to_var_name(name) == var), None)

            if idx is not None:
                m, s = means[idx], cis[idx]
                stats["matrix"][(orig, var)] = {"mean": m, "std": s}
                if not np.isnan(m):
                    row_indices.append(idx)
                    all_valid_indices.append(idx)
            else:
                stats["matrix"][(orig, var)] = {"mean": np.nan, "std": np.nan}

        # Row Average
        logger.debug("row average %d", len(row_indices))
        rm, rs = compute_weighted_average(row_indices, means, cis, models_idx)
        stats["row_avgs"][orig] = {"mean": rm, "std": rs}

    # B. Calculate Column Averages
    for var in var_names:
        col_indices = []
        for i, name in var_groups[var]:
            # Only include if valid
            if not np.isnan(means[i]):
                col_indices.append(i)
        logger.debug("col average %d", len(col_indices))
        cm, cs = compute_weighted_average(col_indices, means, cis, models_idx)
        stats["col_avgs"][var] = {"mean": cm, "std": cs}

    # C. Grand Total Average
    logger.debug("grand average %d", len(all_valid_indices))
    gm, gs = compute_weighted_average(all_valid_indices, means, cis, models_idx)
    stats["grand_avg"] = {"mean": gm, "std": gs}

    # D. Filter out variation columns where ALL cells are NaN
    # (e.g., quantized model columns for system prompt variations with no data)
    non_empty_vars = [
        var for var in stats["var_names"]
        if any(not np.isnan(stats["matrix"].get((orig, var), {"mean": np.nan})["mean"])
               for orig in stats["orig_labels"])
    ]
    if len(non_empty_vars) < len(stats["var_names"]):
        for var in set(stats["var_names"]) - set(non_empty_vars):
            stats["col_avgs"].pop(var, None)
        stats["var_names"] = non_empty_vars

    # Reorder for a clean "quantized data block" in the bottom-right of the heatmap.
    # Quantized variants live as variation COLUMNS (per docs/codebase/quantized-models.md):
    # a var is quantized iff at least one of its underlying full
    # model names contains the QUANTIZATION_SEPARATOR. Move quant columns to the right,
    # and move rows that have data in those columns to the bottom. The resulting blank
    # rectangle sits at the top-right.
    quant_var_set = {
        v for v in stats["var_names"]
        if any(QUANTIZATION_SEPARATOR in name for _, name in var_groups[v])
    }
    quant_vars = [v for v in stats["var_names"] if v in quant_var_set]
    base_vars = [v for v in stats["var_names"] if v not in quant_var_set]
    stats["var_names"] = base_vars + quant_vars

    quant_capable_origs = {
        o for o in stats["orig_labels"]
        if any(
            not np.isnan(stats["matrix"].get((o, v), {"mean": np.nan})["mean"])
            for v in quant_vars
        )
    }
    top_origs = [o for o in stats["orig_labels"] if o not in quant_capable_origs]
    bottom_origs = [o for o in stats["orig_labels"] if o in quant_capable_origs]
    stats["orig_labels"] = top_origs + bottom_origs

    # Expose membership sets so the plotting code can recompute split indices after
    # Unseen / shortlist filtering shifts row & column counts.
    stats["quant_vars"] = quant_vars
    stats["quant_capable_origs"] = sorted(quant_capable_origs)

    return stats


def create_and_save_tables(
    bs_accuracies,
    batch_type,
    models_idx,
    group_name,
    save_tables_path,
    n_splits,
    bs_accuracies_extra=None,
    reverted_models_idx_extra=None,
    create_heatmaps=True,
    skip_nxm=False,
    unseen_pr_by_bs: dict | None = None,
    pr_curve_data_by_bs: dict | None = None,
):
    """
    Orchestrates the creation of all tables (Markdown, Latex, NxM) and heatmaps.
    Calculates stats first, then passes them to formatting functions.

    Only called when train_size_dict has 1 or 2 entries maximum.

    Parameters:
    -----------
    create_heatmaps : bool
        Whether to generate and save heatmap visualizations (default: True)
    """
    # 1. Truncate batch sizes
    target_bs = [8]
    bs_acc_trunc = {k: v for k, v in bs_accuracies.items() if k in target_bs}
    bs_acc_extra_trunc = (
        {k: v for k, v in bs_accuracies_extra.items() if k in target_bs} if bs_accuracies_extra else None
    )

    # 2. Pre-calculate Simple Stats (for the main summary tables)
    averages_main = get_simple_stats(bs_acc_trunc, models_idx)
    averages_extra = (
        get_simple_stats(bs_acc_extra_trunc, models_idx, reverted_models_idx_extra) if bs_acc_extra_trunc else None
    )

    # 3. Setup Captions/Labels
    if batch_type == "across_and_tp_wise":
        caption = f"classification accuracy for each LLM across {group_name} SS n-uplets, mean and standard deviation over SS-uplets and over each of their {n_splits} train/test splits runs."
        label = f"tab: {batch_type} per-model classification performances {group_name}"
    else:
        caption = f"classification accuracy for each LLM for {group_name} SS, mean and standard deviation over each of {n_splits} train/test splits runs."
        label = f"tab:FullResultsTable"

    # 4. Generate & Save NxM Tables and Heatmaps (Loop over batch sizes)
    #    Skipped when skip_nxm=True (e.g. model_groups mode where model names
    #    have no variation suffixes and get_nxm_stats would be degenerate).
    if not skip_nxm:
        for bs in bs_acc_trunc.keys():
            # A. Calculate Stats for this specific BS
            nxm_stats = get_nxm_stats(bs, bs_acc_trunc, models_idx)

            # B. Markdown NxM
            # Abandonned

            # C. LaTeX NxM
            latex_caption = f"Per-class recalls for {group_name} ({bs}-queries)"
            latex_label = f"tab:{group_name}_nxm_bs{bs}"
            latex_table_nxm = make_accuracy_table_nxm_latex(
                stats=nxm_stats, bs=bs, caption=latex_caption, label=latex_label
            )
            tex_nxm_out = Path(save_tables_path) / f"{group_name}_{n_splits}_splits_nxm_bs{bs}.tex"
            tex_nxm_out.write_text(latex_table_nxm, encoding="utf-8")

            # D. Generate Heatmap
            if create_heatmaps and nxm_stats:
                unseen_pr = unseen_pr_by_bs.get(bs) if unseen_pr_by_bs else None
                embedded_pr_curve = (
                    pr_curve_data_by_bs.get(bs) if pr_curve_data_by_bs else None
                )
                for shortlist in [True, False]:
                    heatmap_title = f"Per-class Recall: {group_name} ({bs}-queries)"
                    fig, ax = make_accuracy_heatmap(
                        stats=nxm_stats,
                        figsize=(12, 10),
                        cmap="RdYlGn",
                        vmin=0,
                        vmax=1,
                        title=heatmap_title,
                        truncate_orig_names=True,
                        shortlist=shortlist,
                        unseen_pr=unseen_pr,
                        embedded_pr_curve=embedded_pr_curve,
                    )

                    if fig is not None:
                        shortlist_mention = "_shortlist" if shortlist else ""
                        pdf_out = (
                            Path(save_tables_path)
                            / f"{group_name}_{n_splits}_splits_nxm_bs{bs}_heatmap{shortlist_mention}.pdf"
                        )
                        fig.savefig(pdf_out, bbox_inches="tight")

                        plt.close(fig)  # Close to free memory

    # 5. Generate & Save Summary LaTeX Table
    latex_table = make_accuracy_table_latex(
        bs_accuracies=bs_acc_trunc,
        models_idx=models_idx,
        averages=averages_main,
        caption=caption,
        label=label,
        bs_accuracies_extra=bs_acc_extra_trunc,
        reverted_models_idx_extra=reverted_models_idx_extra,
        averages_extra=averages_extra,
    )
    latex_out = Path(save_tables_path) / f"{group_name}_{n_splits}_splits.txt"
    latex_out.write_text(latex_table, encoding="utf-8")

    # 6. Generate & Save Summary Markdown Table
    markdown_table = make_accuracy_table_markdown(
        bs_accuracies=bs_acc_trunc,
        models_idx=models_idx,
        averages=averages_main,
        bs_accuracies_extra=bs_acc_extra_trunc,
        reverted_models_idx_extra=reverted_models_idx_extra,
        averages_extra=averages_extra,
    )
    md_out = Path(save_tables_path) / f"{group_name}_{n_splits}_splits.md"
    md_out.write_text(markdown_table, encoding="utf-8")


def validate_confusion_matrices(batch_size_confusion_matrix_map, batch_sizes, datasets, classifiers):
    """Validate that all confusion matrices have the same dimensions."""
    nb_of_class = 0
    for bs in batch_sizes:
        for tp in datasets:
            for clf in classifiers:
                if tp in batch_size_confusion_matrix_map[bs] and clf in batch_size_confusion_matrix_map[bs][tp]:
                    mean_cms = batch_size_confusion_matrix_map[bs][tp][clf]["confusion_matrix_mean"]
                    if nb_of_class == 0:
                        nb_of_class = mean_cms.shape[0]
                    else:
                        assert (
                            mean_cms.shape[0] == nb_of_class
                        ), f"Mismatch: expected {nb_of_class}, got {mean_cms.shape[0]}"
    return nb_of_class


def plot_confusion_matrices_on_tr_size_dict(
    train_size_dict: Dict[float, Dict[float, Dict[any, any]]],
    fig_save_path: Path,
    xp_config: Dict,
    classification_config: Dict,
    datasets: list[str],
    models_idx: Dict,
    batch_sizes: list[int] = [1, 3],
    fixed_variation_name: str = None,  # NEW: For grouping by specific variation
    tp_group_names: list[str] = ["FLiPS", "0-1"],
) -> None:
    """
    Plot confusion_matrix_mean as a confusion matrix, averaged over splits.
    Now supports grouped visualization when model_variations are present.

    Args:
        train_size_dict: mapping train_size -> { calculation_item -> pipe_summary_dict }
        fig_save_path: directory path to save figures
        classification_config: config dict with batch_type, n_splits, classifiers, etc.
        datasets: list of dataset names
        models_idx: mapping of model index -> model name
        batch_sizes: list of batch sizes to include in curves
        xp_config: experiment config (NEW - for model_variations)
        fixed_variation_name: which variation to group by (NEW - for multi-variation case)

    For each calculation_item, for each classifier, and for each metric, produces a plot
    where each batch_size in batch_sizes is a separate curve (mean ± std error bars).
    For 'across_and_tp_wise' batch_type, also creates tp_wise plots for each batch_size.
    Figures are saved via save_fig_and_show().

    Info on summary objects:
        summary[key_name][batch_size][tp][clf][f'{metric}_mean'] = np.mean(vals)
        summary[key_name][batch_size][tp][clf][f'{metric}_std'] = np.std(vals)

        where key_name in ['tp_wise', 'mix_tp_at_pred']
        where vals are values of the metrics.
        where metric can be 'confusion_matrix'

    """
    from audit_llm.Classification.confusion_matrix_utils import create_grouped_cm, plot_grouped_cm
    from audit_llm.Classification.training_size_analysis import compute_means_stds
    from audit_llm.plot_configs import XLABEL_CONFIG, YLABEL_CONFIG

    batch_prediction_sizes = classification_config["batch_prediction_sizes"]
    batch_sizes = [bs for bs in batch_sizes if bs in batch_prediction_sizes]

    metric = "confusion_matrix"
    batch_types = classification_config.get("batch_types") or ["tp_wise"]
    n_splits: int = classification_config["n_splits"]

    # Get label config if available (assuming these are defined globally)
    try:
        xlabel_config = XLABEL_CONFIG
        ylabel_config = YLABEL_CONFIG
    except NameError:
        xlabel_config = {}
        ylabel_config = {}

    if openset := classification_config.get("openset", False):
        models_idx = models_idx | {len(models_idx): "Unseen"}

    train_sizes = sorted(train_size_dict.keys())
    calculation_items = sorted({t for ts in train_size_dict.values() for t in ts.keys()})

    for calc_item in calculation_items:
        for clf in classification_config["classifiers"]:
            for train_size in train_sizes:
                summary_dict = train_size_dict[train_size][calc_item]
                effective_keys = get_effective_batch_type_keys(summary_dict, batch_types)

                for effective_key in effective_keys:
                    save_confusion_matrices_path = (
                        Path(fig_save_path) / str(calc_item) / str(clf) / str(train_size) / "ConfusionMatrices" / effective_key
                    )
                    save_confusion_matrices_path.mkdir(exist_ok=True, parents=True)

                    for tp_group_name in tp_group_names:
                        for bs in batch_sizes:
                            tp_names, summary_key = _get_tp_names_for_key(
                                summary_dict, effective_key, bs, tp_group_name, datasets
                            )
                            if not tp_names:
                                continue

                            means, stds = compute_means_stds(
                                train_size_dict, [train_size], calc_item, clf, metric, tp_names, summary_key, bs
                            )

                            # === Plot confusion matrix if available ===
                            if means and len(means[0].shape) == 2:
                                mean_cm = means[0]
                                logger.debug("vmax = %s", float(np.nanmax(mean_cm)))

                                # Regular confusion matrix (no grouping)
                                fname = f"{tp_group_name}_{calc_item}_{bs}_cm.pdf"
                                plot_grouped_cm(
                                    cm=mean_cm,
                                    save_path=save_confusion_matrices_path,
                                    filename=fname,
                                    title=None,
                                    models_idx=models_idx,
                                    group_labels=None,  # No grouping for regular plot
                                    xlabel_config=xlabel_config,
                                    ylabel_config=ylabel_config,
                                )

                                # === Generate grouped confusion matrix if model_variations exist ===
                                if xp_config.get("model_variations", None) is not None:
                                    for k, model_variation_dict in enumerate(xp_config["model_variations"]):
                                        try:
                                            grouped_cm, group_labels = create_grouped_cm(
                                                mean_cm, models_idx, model_variation_dict, fixed_variation_name=fixed_variation_name
                                            )

                                            if group_labels:
                                                grouped_title = (
                                                    f"Grouped CM - {tp_group_name} "
                                                    f"(var_item={calc_item}, clf={clf}, train_size={train_size}, bs={bs})"
                                                )
                                                grouped_filename = f"{tp_group_name}_{calc_item}_{bs}_cm_grouped_{k}.pdf"

                                                plot_grouped_cm(
                                                    cm=grouped_cm,
                                                    save_path=save_confusion_matrices_path,
                                                    filename=grouped_filename,
                                                    title=grouped_title,
                                                    models_idx=None,  # Not needed for grouped plot
                                                    group_labels=group_labels,
                                                    xlabel_config=xlabel_config,
                                                    ylabel_config=ylabel_config,
                                                )
                                        except Exception as e:
                                            logger.warning("Could not create grouped CM: %s", e)
                            else:
                                logger.warning(
                                    "No confusion matrix data for %s/%s at calc_item %s, clf %s, train_size %s, bs %s",
                                    tp_group_name, effective_key, calc_item, clf, train_size, bs,
                                )
