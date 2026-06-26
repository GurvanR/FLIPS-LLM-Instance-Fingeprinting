"""Open-set classification via composition with MultiTokenPairClassification.

Algorithm overview
------------------
Open-set classification identifies whether a sample belongs to one of the
*known* classes (models seen during training) or to an *unknown* class
(a model not seen during training).

**Outer loop** — For each dataset, classes are repeatedly split into
*known* and *unknown* groups.  A classifier is trained only on known-class
samples, and evaluated on both known *and* unknown samples.

**Inner loop** — Within the known classes of each outer split, a second
(inner) loop further partitions the known classes to simulate the presence
of unknowns.  This produces a distribution of maximum predicted
probabilities for both correctly-classified ("known") and misclassified
("unknown") samples, from which a rejection *threshold* is derived.

**Threshold calibration** — The ``alpha`` quantile of the inner-loop
probability distributions determines the threshold below which predictions
are rejected (labelled as "unknown").

**Final evaluation** — The outer classifier's predictions on the held-out
known and truly-unknown samples are thresholded: predictions with confidence
below the threshold are mapped to the *unknown* label (``-1``).

The ``alpha_trade_off_show`` option sweeps a range of alpha values and
plots the accuracy trade-off between known-class and unknown-class
recognition.
"""

from __future__ import annotations
import joblib
import json
import logging
logger = logging.getLogger(__name__)

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

import numpy as np

from audit_llm.Classification.classification_constants import SPLITTER_MAP
from audit_llm.Classification.Preprocessing_data import fit_transform_normalize
from audit_llm.xp_tools import (
    extract_thresholds,
    get_tp_uplet_name,
    plot_thresholds_distribution,
    relabel_y_labels,
    origin_label,
    plot_alpha_tradeoff,
    plot_alpha_roc_curves,
    plot_openset_roc_curves,
    plot_roc_curves_overlay,
    plot_unseen_and_global_pr_vs_alpha,
    plot_unseen_and_global_pr_vs_confidence,
    save_openset_metrics_table,
)

if TYPE_CHECKING:
    from audit_llm.Classification.multi_classification import MultiTokenPairClassification


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------


def _alpha_str(alpha: float) -> str:
    """Canonical filesystem-safe representation of an alpha value.

    0.05 → "0_05"; 0.1 → "0_1"; 0.10 → "0_1" (canonicalised).
    """
    s = f"{alpha:.4f}".rstrip("0").rstrip(".") or "0"
    return s.replace(".", "_")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OuterIterationResult:
    """Result of one outer-loop iteration for a single dataset."""

    known_classes: np.ndarray
    unknown_classes: np.ndarray
    max_probas: Dict[str, list]  # {"Known": [...], "Unknown": [...]}
    final_outer_probas: Dict  # {"Known": probas, "Unknown": probas}
    final_known_val: np.ndarray  # relabeled y values for the outer known validation set
    label_map: Dict[int, int]  # {relabeled_idx: original_label}


@dataclass
class OpenSetTokenPairResult:
    """Container for all outer-loop iterations of one dataset."""

    iterations: List[OuterIterationResult] = field(default_factory=list)


@dataclass
class OuterIterationResultMixTP:
    """Result of one outer-loop iteration for mix_tp_at_pred open-set.

    Stores per-(TP, bs) raw probas so that final predictions can re-mix
    per-uplet from checkpoints.
    """

    known_classes: np.ndarray
    unknown_classes: np.ndarray
    # {uplet_name: {"Known": [probs_dict, ...], "Unknown": [probs_dict, ...]}}
    max_probas_per_uplet: Dict[str, Dict[str, list]]
    # {bs: {tp_name: {"Known": {clf: probas}, "Unknown": {clf: probas}}}}
    final_outer_probas_per_tp_bs: Dict
    # {bs: {tp_name: y_val_array}}
    final_known_val_per_tp_bs: Dict
    label_map: Dict[int, int]


@dataclass
class OpenSetMixTPResult:
    """Container for all outer-loop iterations of mix_tp_at_pred open-set."""

    iterations: List[OuterIterationResultMixTP] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main class (composition)
# ---------------------------------------------------------------------------


class OpenSetClassification:
    """Open-set classification wrapper around ``MultiTokenPairClassification``.

    Uses composition: all shared infrastructure (splitting, training,
    normalization, batch prediction) is accessed via ``self.multi``.
    """

    def __init__(self, multi: MultiTokenPairClassification) -> None:
        self.multi = multi
        self.roc_data: dict = {}
        self._replot_mode: bool = False

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def replot_from_cache(self) -> None:
        """Regenerate openset figures from persisted classification cache.

        Skips data collection; runs only the plot tail. Both
        `_do_final_predictions` and `_compute_and_plot_roc_curves` load
        their inputs from `OpenSetXP_checkpoints/*.pkl` via
        `_load_mix_tp_result()` / `_load_openset_dict()`. If those pickles
        are missing (legacy XP), loaders return None and downstream calls
        silently no-op with a warning.
        """
        multi = self.multi
        multi._initialize_results_and_conf_structure()
        self._replot_mode = True
        try:
            self._do_final_predictions()
            if multi.roc_curve_show:
                self._compute_and_plot_roc_curves()
        except Exception:
            logger.exception(
                "replot_from_cache failed for batch_type=%s train_size=%s",
                multi.batch_type, getattr(multi, "train_size", "?"),
            )
        finally:
            self._replot_mode = False

        multi.batch_size_results_map = multi.results
        multi.batch_size_confusion_matrices_map = multi.confusion_matrices
        multi.probs_save_map = {}

    def run(self) -> None:
        """Execute the full open-set classification pipeline.

        After completion, populates ``self.multi.batch_size_results_map``,
        ``self.multi.batch_size_confusion_matrices_map``, and
        ``self.multi.probs_save_map``.
        """
        multi = self.multi

        multi._initialize_results_and_conf_structure()
        self._verify_or_write_meta()

        if multi.batch_type == "mix_tp_at_pred":
            self._run_mix_tp_at_pred()
        elif multi.batch_type != "tp_wise":
            raise NotImplementedError(
                f"Batch type '{multi.batch_type}' is not implemented for openset. "
                "Supported: 'tp_wise', 'mix_tp_at_pred'."
            )
        else:
            for tp_name, X in multi.X_s.items():
                logger.info(f"Processing dataset {tp_name} for OpenSet experiments...")
    
                if self._get_openset_ds_path(tp_name).exists():
                    logger.info(
                        f"Dataset {tp_name} already processed for OpenSet experiments, skipping."
                    )
                    continue
    
                dataset_result = OpenSetTokenPairResult()
    
                X_p, y_p = multi._prepare_data(X, batch_type=multi.batch_type, tp_name=tp_name)
    
                # ---- Outer loop ----
                for outer_iter in range(multi.openset_m_splits):
                    known_classes, unknown_classes = multi._class_split(y_p)
    
                    # ---- Outer Known ----
                    mask_known = np.isin(y_p, known_classes)
                    X_outer_known, y_outer_known = X_p[mask_known], y_p[mask_known]
                    y_outer_known_relabeled, label_map = relabel_y_labels(y_outer_known)
    
                    # Splitter created per outer iteration (n_known_classes varies)
                    splitter = SPLITTER_MAP[multi.splitter_type](
                        n_splits=multi.n_splits,
                        random_state=multi.random_seed,
                        test_size=multi.test_size * len(known_classes),
                    )
                    splits = list(splitter.split(X_outer_known, y_outer_known_relabeled))
                    tr_idx, te_idx = splits[outer_iter % multi.n_splits]
    
                    X_outer_known_train = X_outer_known[tr_idx]
                    y_outer_known_train = y_outer_known_relabeled[tr_idx]
                    X_outer_known_val = X_outer_known[te_idx]
                    y_outer_known_val = y_outer_known_relabeled[te_idx]
    
                    # Balance training fold only (after split to avoid leakage)
                    _blank = multi._create_single_token_pair_classif(tp_name)
                    X_outer_known_train, y_outer_known_train = _blank.balance(
                        X_outer_known_train, y_outer_known_train
                    )
    
                    X_outer_known_train, y_outer_known_train = multi._reducing_token_pair(
                        X_outer_known_train, y_outer_known_train, truncation=multi.train_size
                    )
    
                    # ---- Outer Unknown ----
                    mask_unknown = np.isin(y_p, unknown_classes)
                    X_outer_unknown, y_outer_unknown = X_p[mask_unknown], y_p[mask_unknown]
                    X_outer_unknown, y_outer_unknown = multi._reducing_token_pair(
                        X_outer_unknown,
                        y_outer_unknown,
                        truncation=multi.test_size,
                    )
    
                    # ---- Threshold calibration via inner loop ----
                    max_probas: Dict[str, list] = {"Known": [], "Unknown": []}
                    inner_splitter = self._create_inner_splitter()
    
                    for inner_iter in range(multi.openset_m_splits // 2):
                        inner_known_classes, inner_unknown_classes = multi._class_split(
                            y_outer_known_train
                        )
    
                        # Inner Known
                        mask_inner_known = np.isin(y_outer_known_train, inner_known_classes)
                        X_inner_known = X_outer_known_train[mask_inner_known]
                        y_inner_known = y_outer_known_train[mask_inner_known]
                        y_inner_known_relabeled, _ = relabel_y_labels(y_inner_known)
    
                        inner_splits = list(
                            inner_splitter.split(X_inner_known, y_inner_known_relabeled)
                        )
                        inner_tr_idx, inner_te_idx = inner_splits[
                            inner_iter % multi.n_splits
                        ]
    
                        X_inner_known_train = X_inner_known[inner_tr_idx]
                        y_inner_known_train = y_inner_known_relabeled[inner_tr_idx]
                        X_inner_known_val = X_inner_known[inner_te_idx]
                        y_inner_known_val = y_inner_known_relabeled[inner_te_idx]
    
                        # Inner Unknown
                        mask_inner_unknown = np.isin(
                            y_outer_known_train, inner_unknown_classes
                        )
                        X_inner_unknown = X_outer_known_train[mask_inner_unknown]
    
                        # Normalize: fit on inner-known-train, transform all inner sets
                        inner_normalizer = multi._create_normalizer()
                        inner_normalizer.fit(X_inner_known_train)
                        X_inner_known_train_norm = inner_normalizer.transform(
                            X_inner_known_train
                        )
                        X_inner_known_val_norm = inner_normalizer.transform(
                            X_inner_known_val
                        )
                        X_inner_unknown_norm = inner_normalizer.transform(X_inner_unknown)
    
                        # Train + predict
                        y_pred_proba_dict = multi._train_and_predict_proba(
                            X_inner_known_train_norm,
                            y_inner_known_train,
                            X_val={
                                "Known": X_inner_known_val_norm,
                                "Unknown": X_inner_unknown_norm,
                            },
                            tp_name=tp_name,
                        )
    
                        # Collect max probas for Known predictions
                        _, _, max_probas_known = multi._batch_predict_token_pairs_wise(
                            y_pred_proba_dict["Known"],
                            y_inner_known_val,
                            token_pairs=[tp_name],
                        )
                        max_probas["Known"].append(max_probas_known)
    
                        # Collect max probas for Unknown predictions
                        # y_inner_known_val is passed for shape compatibility —
                        # all predictions will be "wrong" anyway since the true labels
                        # come from inner_unknown_classes
                        _, _, max_probas_unknown = multi._batch_predict_token_pairs_wise(
                            y_pred_proba_dict["Unknown"],
                            y_inner_known_val,
                            token_pairs=[tp_name],
                            mode="Unknown",
                        )
                        max_probas["Unknown"].append(max_probas_unknown)
    
                    # ---- Final Outer Train-Test ----
                    outer_normalizer = multi._create_normalizer()
                    outer_normalizer.fit(X_outer_known_train)
                    X_outer_known_train_norm = outer_normalizer.transform(
                        X_outer_known_train
                    )
                    X_outer_known_val_norm = outer_normalizer.transform(X_outer_known_val)
                    X_outer_unknown_norm = outer_normalizer.transform(X_outer_unknown)
    
                    final_outer_probas = multi._train_and_predict_proba(
                        X_outer_known_train_norm,
                        y_outer_known_train,
                        X_val={
                            "Known": X_outer_known_val_norm,
                            "Unknown": X_outer_unknown_norm,
                        },
                        tp_name=tp_name,
                    )
    
                    dataset_result.iterations.append(
                        OuterIterationResult(
                            known_classes=known_classes,
                            unknown_classes=unknown_classes,
                            max_probas=max_probas,
                            final_outer_probas=final_outer_probas,
                            final_known_val=y_outer_known_val,
                            label_map=label_map,
                        )
                    )
    
                self._save_openset_split(dataset_result, tp_name)

        self._do_final_predictions()

        if multi.roc_curve_show:
            self._compute_and_plot_roc_curves()

        # Populate multi's result attributes for downstream consumption
        multi.batch_size_results_map = multi.results
        multi.batch_size_confusion_matrices_map = multi.confusion_matrices
        multi.probs_save_map = {}

    # ------------------------------------------------------------------
    # mix_tp_at_pred open-set pipeline
    # ------------------------------------------------------------------

    def _run_mix_tp_at_pred(self) -> None:
        """Open-set classification with mix_tp_at_pred batch type.

        Instead of processing each TP independently (as in tp_wise), all TPs
        are trained within each outer iteration and predictions are mixed
        across TP uplets.  Threshold calibration (inner loop) and final
        evaluation (outer loop) both operate on mixed predictions.
        """
        multi = self.multi

        # --- Prepare data for all TPs ---
        X_prepared: Dict[str, tuple] = {}
        for tp_name, X in multi.X_s.items():
            X_p, y_p = multi._prepare_data(X, batch_type=multi.batch_type, tp_name=tp_name)
            X_prepared[tp_name] = (X_p, y_p)

        # Use first TP's y as reference (classes should be identical across TPs)
        ref_tp = next(iter(X_prepared))
        y_reference = X_prepared[ref_tp][1]

        checkpoint_key = f"mix_tp_at_pred_utp{multi.unique_tp_in_mix}"
        if self._get_openset_ds_path(checkpoint_key).exists():
            logger.info(
                "mix_tp_at_pred open-set already processed, skipping to final predictions."
            )
        else:
            mix_result = OpenSetMixTPResult()

            for outer_iter in range(multi.openset_m_splits):
                logger.info(
                    "mix_tp_at_pred open-set: outer iteration %d/%d",
                    outer_iter + 1, multi.openset_m_splits,
                )
                known_classes, unknown_classes = multi._class_split(y_reference)

                # --- Per-TP Known/Unknown data ---
                tp_known: Dict[str, tuple] = {}
                tp_unknown: Dict[str, tuple] = {}
                label_map = None

                for tp_name, (X_p, y_p) in X_prepared.items():
                    mask_known = np.isin(y_p, known_classes)
                    X_known, y_known = X_p[mask_known], y_p[mask_known]
                    y_known_relabeled, lm = relabel_y_labels(y_known, label_map=label_map)
                    if label_map is None:
                        label_map = lm
                    tp_known[tp_name] = (X_known, y_known_relabeled)

                    mask_unknown = np.isin(y_p, unknown_classes)
                    X_unk, y_unk = X_p[mask_unknown], y_p[mask_unknown]
                    X_unk, y_unk = multi._reducing_token_pair(X_unk, y_unk, truncation=multi.test_size)
                    tp_unknown[tp_name] = (X_unk, y_unk)

                # Per-TP outer splits (sample counts can differ across TPs)
                tp_outer_splits: Dict[str, tuple] = {}
                for tp_name in multi.X_s:
                    X_known_tp, y_known_tp = tp_known[tp_name]
                    splitter = SPLITTER_MAP[multi.splitter_type](
                        n_splits=multi.n_splits,
                        random_state=multi.random_seed,
                        test_size=multi.test_size * len(known_classes),
                    )
                    splits = list(splitter.split(X_known_tp, y_known_tp))
                    tp_outer_splits[tp_name] = splits[outer_iter % multi.n_splits]

                # Use reference TP for class-level splits only
                ref_tr_idx = tp_outer_splits[ref_tp][0]
                _, y_known_ref = tp_known[ref_tp]
                y_outer_train_ref = y_known_ref[ref_tr_idx]

                # --- Inner loop: threshold calibration ---
                max_probas_per_uplet: Dict[str, Dict[str, list]] = {}
                inner_splitter = self._create_inner_splitter()

                for inner_iter in range(multi.openset_m_splits // 2):
                    inner_known_cls, inner_unknown_cls = multi._class_split(y_outer_train_ref)

                    inner_probas_known: Dict[int, Dict] = {
                        bs: {} for bs in multi.batch_prediction_sizes
                    }
                    inner_probas_unknown: Dict[int, Dict] = {
                        bs: {} for bs in multi.batch_prediction_sizes
                    }
                    inner_y_vals: Dict[int, Dict] = {
                        bs: {} for bs in multi.batch_prediction_sizes
                    }
                    sparse_skip = False

                    for tp_name in multi.X_s:
                        X_known_tp, y_known_tp = tp_known[tp_name]
                        tp_tr_idx, _ = tp_outer_splits[tp_name]
                        X_tr_tp = X_known_tp[tp_tr_idx]
                        y_tr_tp = y_known_tp[tp_tr_idx]

                        # Inner Known
                        mask_ik = np.isin(y_tr_tp, inner_known_cls)
                        X_ik, y_ik = X_tr_tp[mask_ik], y_tr_tp[mask_ik]
                        y_ik_relabeled, _ = relabel_y_labels(y_ik)

                        inner_splits = list(inner_splitter.split(X_ik, y_ik_relabeled))
                        i_tr, i_te = inner_splits[inner_iter % multi.n_splits]

                        X_ik_train = X_ik[i_tr]
                        y_ik_train = y_ik_relabeled[i_tr]
                        X_ik_val = X_ik[i_te]
                        y_ik_val = y_ik_relabeled[i_te]

                        # Inner Unknown
                        mask_iu = np.isin(y_tr_tp, inner_unknown_cls)
                        X_iu = X_tr_tp[mask_iu]

                        # Balance
                        _blank = multi._create_single_token_pair_classif(tp_name)
                        X_ik_train_bal, y_ik_train_bal = _blank.balance(
                            X_ik_train, y_ik_train
                        )

                        for bs in multi.batch_prediction_sizes:
                            if not multi.tp_uplet_dict.get(bs):
                                continue

                            truncation = multi.train_size // multi._resolve_unique_tp(bs)
                            X_trunc, y_trunc = multi._reducing_token_pair(
                                X_ik_train_bal, y_ik_train_bal, truncation=truncation
                            )

                            # Sparse data check
                            classes_arr, counts = np.unique(y_trunc, return_counts=True)
                            if len(classes_arr) < 2 or counts.min() < 2:
                                logger.warning(
                                    "Inner iter %d, TP '%s', bs=%d: sparse data "
                                    "(%s samples in %d classes) after truncation=%d. "
                                    "Skipping this inner iteration.",
                                    inner_iter, tp_name, bs,
                                    counts.tolist(), len(classes_arr), truncation,
                                )
                                sparse_skip = True
                                break

                            # Normalize: fit on truncated train, transform val & unknown
                            normalizer = multi._create_normalizer()
                            normalizer.fit(X_trunc)
                            X_trunc_norm = normalizer.transform(X_trunc)
                            X_ival_norm = normalizer.transform(X_ik_val)
                            X_iunk_norm = normalizer.transform(X_iu)

                            probas = multi._train_and_predict_proba(
                                X_trunc_norm, y_trunc,
                                X_val={"Known": X_ival_norm, "Unknown": X_iunk_norm},
                                tp_name=tp_name,
                            )
                            inner_probas_known[bs][tp_name] = probas["Known"]
                            inner_probas_unknown[bs][tp_name] = probas["Unknown"]
                            inner_y_vals[bs][tp_name] = y_ik_val

                        if sparse_skip:
                            break

                    if sparse_skip:
                        continue

                    # Mix predictions across uplets and collect max_probas
                    _, _, probs_k = multi._batch_predict_mixed_wise(
                        inner_probas_known, inner_y_vals,
                        tp_uplet_dict=multi.tp_uplet_dict,
                    )
                    _, _, probs_u = multi._batch_predict_mixed_wise(
                        inner_probas_unknown, inner_y_vals,
                        tp_uplet_dict=multi.tp_uplet_dict,
                        mode="Unknown",
                    )

                    for uplet_name in probs_k:
                        if uplet_name not in max_probas_per_uplet:
                            max_probas_per_uplet[uplet_name] = {
                                "Known": [], "Unknown": [],
                            }
                        max_probas_per_uplet[uplet_name]["Known"].append(
                            probs_k[uplet_name]
                        )
                        max_probas_per_uplet[uplet_name]["Unknown"].append(
                            probs_u[uplet_name]
                        )

                # --- Outer evaluation ---
                final_probas: Dict[int, Dict] = {
                    bs: {} for bs in multi.batch_prediction_sizes
                }
                final_y_vals: Dict[int, Dict] = {
                    bs: {} for bs in multi.batch_prediction_sizes
                }

                for tp_name in multi.X_s:
                    X_known_tp, y_known_tp = tp_known[tp_name]
                    tp_tr_idx, tp_te_idx = tp_outer_splits[tp_name]
                    X_tr_tp, X_val_tp = X_known_tp[tp_tr_idx], X_known_tp[tp_te_idx]
                    y_tr_tp, y_val_tp = y_known_tp[tp_tr_idx], y_known_tp[tp_te_idx]

                    _blank = multi._create_single_token_pair_classif(tp_name)
                    X_tr_bal, y_tr_bal = _blank.balance(X_tr_tp, y_tr_tp)

                    X_unk_tp, _ = tp_unknown[tp_name]

                    for bs in multi.batch_prediction_sizes:
                        if not multi.tp_uplet_dict.get(bs):
                            continue

                        truncation = multi.train_size // multi._resolve_unique_tp(bs)
                        X_tr_trunc, y_tr_trunc = multi._reducing_token_pair(
                            X_tr_bal, y_tr_bal, truncation=truncation
                        )

                        normalizer = multi._create_normalizer()
                        normalizer.fit(X_tr_trunc)
                        X_tr_norm = normalizer.transform(X_tr_trunc)
                        X_val_norm = normalizer.transform(X_val_tp)
                        X_unk_norm = normalizer.transform(X_unk_tp)

                        probas = multi._train_and_predict_proba(
                            X_tr_norm, y_tr_trunc,
                            X_val={"Known": X_val_norm, "Unknown": X_unk_norm},
                            tp_name=tp_name,
                        )
                        final_probas[bs][tp_name] = probas
                        final_y_vals[bs][tp_name] = y_val_tp

                mix_result.iterations.append(
                    OuterIterationResultMixTP(
                        known_classes=known_classes,
                        unknown_classes=unknown_classes,
                        max_probas_per_uplet=max_probas_per_uplet,
                        final_outer_probas_per_tp_bs=final_probas,
                        final_known_val_per_tp_bs=final_y_vals,
                        label_map=label_map,
                    )
                )

            self._save_openset_split(mix_result, checkpoint_key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_inner_splitter(self):
        """Create a splitter for the inner calibration loop.

        Adjusts test_size so that enough samples remain for batch prediction.
        """
        multi = self.multi
        if int(0.2 * multi.train_size) > max(multi.batch_prediction_sizes) + 1:
            inner_test_size = 0.2
        else:
            inner_test_size = (
                max(multi.batch_prediction_sizes) + 1
            ) / multi.train_size

        assert int(multi.train_size * (1 - inner_test_size)) >= max(
            multi.batch_prediction_sizes
        ), (
            f" {inner_test_size= } Not enough samples to do batch prediction with "
            f"size {max(multi.batch_prediction_sizes)}. Reduce batch size or increase "
            f"train_size or test_size."
        )

        return SPLITTER_MAP[multi.splitter_type](
            n_splits=multi.n_splits,
            random_state=multi.random_seed,
            test_size=inner_test_size,
        )

    def _setup_alpha_iterations(self):
        """Shared alpha-sweep setup: scanning checkpoints and determining iterations.

        Returns (alphas_iterations, completed_alphas, checkpoint_dir).
        """
        multi = self.multi
        checkpoint_dir = self._batch_type_fig_path() / "alpha_checkpoints"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        if multi.alpha_trade_off_show:
            alphas_iterations = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
            multi.alpha_results = {}
            completed_alphas: list = []

            logger.info(f"Scanning for checkpoints in {checkpoint_dir}...")
            for alpha in alphas_iterations:
                alpha_str = str(alpha).replace(".", "_")
                alpha_file = checkpoint_dir / f"checkpoint_alpha_{alpha_str}.pkl"
                if alpha_file.exists():
                    try:
                        alpha_data = joblib.load(alpha_file)
                        multi.alpha_results[alpha] = alpha_data
                        completed_alphas.append(alpha)
                    except (pickle.UnpicklingError, EOFError):
                        logger.warning(
                            f"Warning: Checkpoint for alpha {alpha} is corrupted. "
                            f"Will recalculate."
                        )

            if completed_alphas:
                logger.info(
                    f"Resumed from checkpoints. Already completed: {completed_alphas}"
                )
        else:
            alphas_iterations = [multi.alpha_quantile_threshold]
            completed_alphas = []

        return alphas_iterations, completed_alphas, checkpoint_dir

    def _finalize_alpha_step(self, alpha, openset_keys, checkpoint_dir, thresholds_to_plot):
        """Shared alpha-step finalization: save checkpoint, plot thresholds."""
        multi = self.multi

        if multi.alpha_trade_off_show:
            multi.batch_size_results_map = multi.results
            multi.batch_size_confusion_matrices_map = multi.confusion_matrices
            multi.probs_save_map = {}
            summary = multi.get_raw_summary_results()

            current_alpha_metrics = self._compute_alpha_metrics(openset_keys, summary)

            multi.alpha_results[alpha] = current_alpha_metrics

            alpha_str = str(alpha).replace(".", "_")
            alpha_file = checkpoint_dir / f"checkpoint_alpha_{alpha_str}.pkl"
            joblib.dump(current_alpha_metrics, alpha_file, compress=3)
            logger.info(f"Successfully saved checkpoint for alpha={alpha}")

            multi._initialize_results_and_conf_structure()

        dist_path = self._alpha_fig_path(alpha)
        if thresholds_to_plot:
            plot_thresholds_distribution(thresholds_to_plot, fig_path=dist_path)
        else:
            logger.warning("No thresholds to plot for alpha=%s, skipping distribution plot.", alpha)

    def _do_final_predictions(self) -> None:
        """Load checkpoints and compute final open-set predictions with threshold rejection."""
        multi = self.multi

        if multi.batch_type == "mix_tp_at_pred":
            self._do_final_predictions_mix_tp()
            return

        openset_dict = self._load_openset_dict()
        alphas_iterations, completed_alphas, checkpoint_dir = self._setup_alpha_iterations()

        for alpha in alphas_iterations:
            if (
                multi.alpha_trade_off_show
                and alpha in completed_alphas
                and not self._replot_mode
            ):
                logger.info(f"Skipping alpha={alpha} (already completed)")
                continue

            logger.info(f"Processing alpha={alpha}")
            thresholds_to_plot: Dict[str, list] = {}

            for tp_name, ds_result in openset_dict.items():
                thresholds_to_plot[tp_name] = []

                for iteration in ds_result.iterations:
                    final_outer_probas = iteration.final_outer_probas
                    final_known_val = iteration.final_known_val
                    max_probas = iteration.max_probas
                    label_map = iteration.label_map

                    thresholds = extract_thresholds(
                        max_probas, fig_path=self._batch_type_fig_path(), alpha=alpha
                    )
                    thresholds_to_plot[tp_name].append(thresholds)

                    y_true_known, y_pred_known, _ = (
                        multi._batch_predict_token_pairs_wise(
                            final_outer_probas["Known"],
                            final_known_val,
                            token_pairs=[tp_name],
                            thresholds=thresholds,
                        )
                    )

                    y_true_unknown, y_pred_unknown, _ = (
                        multi._batch_predict_token_pairs_wise(
                            final_outer_probas["Unknown"],
                            final_known_val,
                            token_pairs=[tp_name],
                            thresholds=thresholds,
                            mode="Unknown",
                        )
                    )

                    y_true = {
                        clf: {
                            bs: np.concatenate(
                                [
                                    origin_label(
                                        y_true_known[clf][bs], label_map
                                    ),
                                    y_true_unknown[clf][bs],  # all -1
                                ]
                            )
                            for bs in multi.batch_prediction_sizes
                        }
                        for clf in multi.classifier_types
                    }

                    y_pred = {
                        clf: {
                            bs: np.concatenate(
                                [
                                    origin_label(
                                        y_pred_known[clf][bs], label_map
                                    ),
                                    origin_label(
                                        y_pred_unknown[clf][bs], label_map
                                    ),
                                ]
                            )
                            for bs in multi.batch_prediction_sizes
                        }
                        for clf in multi.classifier_types
                    }

                    multi._save_prediction_results(y_true, y_pred, tp=tp_name)

            self._finalize_alpha_step(
                alpha, list(openset_dict.keys()), checkpoint_dir, thresholds_to_plot,
            )

        if multi.alpha_trade_off_show:
            plot_alpha_tradeoff(
                multi.alpha_results,
                self._batch_type_fig_path(),
                multi.batch_prediction_sizes,
            )

    def _do_final_predictions_mix_tp(self) -> None:
        """Final predictions for mix_tp_at_pred open-set: per-uplet thresholding."""
        multi = self.multi
        mix_result = self._load_mix_tp_result()
        if mix_result is None:
            logger.warning("No mix_tp_at_pred checkpoint found, cannot compute final predictions.")
            return

        alphas_iterations, completed_alphas, checkpoint_dir = self._setup_alpha_iterations()

        for alpha in alphas_iterations:
            if (
                multi.alpha_trade_off_show
                and alpha in completed_alphas
                and not self._replot_mode
            ):
                logger.info(f"Skipping alpha={alpha} (already completed)")
                continue

            logger.info(f"Processing alpha={alpha} (mix_tp_at_pred)")
            thresholds_to_plot: Dict[str, list] = {}
            all_uplet_names: list = []

            for iteration in mix_result.iterations:
                label_map = iteration.label_map

                # Organize per-bs data for this iteration
                organized: Dict[int, tuple] = {}
                for bs in multi.batch_prediction_sizes:
                    if not multi.tp_uplet_dict.get(bs):
                        continue
                    bs_data = iteration.final_outer_probas_per_tp_bs.get(bs)
                    if not bs_data:
                        continue

                    probas_known_bs = {
                        tp: bs_data[tp]["Known"] for tp in bs_data
                    }
                    probas_unknown_bs = {
                        tp: bs_data[tp]["Unknown"] for tp in bs_data
                    }
                    y_vals_bs = {
                        tp: iteration.final_known_val_per_tp_bs[bs][tp]
                        for tp in bs_data
                    }
                    all_tps = list(bs_data.keys())

                    uc_known = multi._resolve_unique_classes(
                        y_vals_bs, all_tps, None, tp_keyed=True
                    )
                    uc_unknown = np.array([-1])
                    org_k = multi._organize_probas(
                        probas_known_bs, y_vals_bs, all_tps,
                        uc_known, None, tp_keyed=True,
                    )
                    org_u = multi._organize_probas(
                        probas_unknown_bs, y_vals_bs, all_tps,
                        uc_unknown, "Unknown", tp_keyed=True,
                    )
                    organized[bs] = (org_k, org_u, uc_known, uc_unknown)

                # Per-uplet predictions
                for bs in multi.batch_prediction_sizes:
                    if bs not in organized:
                        continue
                    org_k, org_u, uc_known, uc_unknown = organized[bs]

                    for tp_uplet in multi.tp_uplet_dict.get(bs, []):
                        uplet_name = get_tp_uplet_name(tp_uplet)

                        uplet_mp = iteration.max_probas_per_uplet.get(uplet_name)
                        if uplet_mp is None:
                            logger.warning(
                                "No inner-loop max_probas for uplet %s, skipping.",
                                uplet_name,
                            )
                            continue

                        thresholds = extract_thresholds(
                            uplet_mp, fig_path=self._batch_type_fig_path(), alpha=alpha
                        )
                        if uplet_name not in thresholds_to_plot:
                            thresholds_to_plot[uplet_name] = []
                        thresholds_to_plot[uplet_name].append(thresholds)

                        if uplet_name not in all_uplet_names:
                            all_uplet_names.append(uplet_name)

                        y_true_uplet: Dict = {
                            clf: {} for clf in multi.classifier_types
                        }
                        y_pred_uplet: Dict = {
                            clf: {} for clf in multi.classifier_types
                        }

                        for clf in multi.classifier_types:
                            threshold_val = thresholds[clf][bs]

                            preds_k, trues_k, _ = multi._batch_predict_mixed_single(
                                clf_name=clf,
                                tp_uplet=tp_uplet,
                                batch_size=bs,
                                unique_classes=uc_known,
                                probas_by_class_tp=org_k[clf],
                                threshold=threshold_val,
                                mode=None,
                            )

                            preds_u, trues_u, _ = multi._batch_predict_mixed_single(
                                clf_name=clf,
                                tp_uplet=tp_uplet,
                                batch_size=bs,
                                unique_classes=uc_unknown,
                                probas_by_class_tp=org_u[clf],
                                threshold=threshold_val,
                                mode="Unknown",
                            )

                            y_true_uplet[clf][bs] = np.concatenate([
                                origin_label(np.array(trues_k), label_map),
                                np.array(trues_u),  # all -1
                            ])
                            y_pred_uplet[clf][bs] = np.concatenate([
                                origin_label(np.array(preds_k), label_map),
                                origin_label(np.array(preds_u), label_map),
                            ])

                        multi._save_prediction_results(
                            y_true_uplet, y_pred_uplet, tp=uplet_name
                        )

            self._finalize_alpha_step(
                alpha, all_uplet_names, checkpoint_dir, thresholds_to_plot,
            )

        if multi.alpha_trade_off_show:
            plot_alpha_tradeoff(
                multi.alpha_results,
                self._batch_type_fig_path(),
                multi.batch_prediction_sizes,
            )

    def _compute_alpha_metrics(
        self,
        result_keys: list,
        summary: Dict,
    ) -> Dict[str, list]:
        """Compute global/unseen/known accuracy mean and std across batch sizes.

        Parameters
        ----------
        result_keys : list
            Token-pair names (tp_wise) or uplet names (mix_tp_at_pred) to
            iterate over in the summary dict.

        Returns dict with keys ``global_accuracy_{mean,std}``,
        ``unseen_accuracy_{mean,std}``, ``known_accuracy_{mean,std}``,
        each a list with one entry per batch_size.
        """
        multi = self.multi
        current_alpha_metrics: Dict[str, list] = {
            "global_accuracy_mean": [],
            "global_accuracy_std": [],
            "unseen_accuracy_mean": [],
            "unseen_accuracy_std": [],
            "known_accuracy_mean": [],
            "known_accuracy_std": [],
        }

        for bs in multi.batch_prediction_sizes:
            global_accs, unseen_accs, known_accs = [], [], []

            for key_name in result_keys:
                bt_summary = summary.get(multi.batch_type, {}).get(bs, {})
                if key_name not in bt_summary:
                    continue
                for clf in multi.classifier_types:
                    clf_summary = bt_summary[key_name][clf]
                    global_accs.append(clf_summary["accuracy_mean"])

                    confusion_mean = clf_summary["confusion_matrix_mean"]
                    row_sums = confusion_mean.sum(axis=1)

                    # Unseen accuracy (last class)
                    if row_sums[-1] > 0:
                        unseen_accs.append(
                            confusion_mean[-1, -1] / row_sums[-1]
                        )
                    else:
                        unseen_accs.append(0.0)

                    # Known accuracy (all classes except last)
                    known_class_accs = [
                        confusion_mean[i, i] / row_sums[i]
                        for i in range(len(confusion_mean) - 1)
                        if row_sums[i] > 0
                    ]
                    known_accs.append(
                        np.mean(known_class_accs) if known_class_accs else 0.0
                    )

            current_alpha_metrics["global_accuracy_mean"].append(np.mean(global_accs))
            current_alpha_metrics["global_accuracy_std"].append(np.std(global_accs))
            current_alpha_metrics["unseen_accuracy_mean"].append(np.mean(unseen_accs))
            current_alpha_metrics["unseen_accuracy_std"].append(np.std(unseen_accs))
            current_alpha_metrics["known_accuracy_mean"].append(np.mean(known_accs))
            current_alpha_metrics["known_accuracy_std"].append(np.std(known_accs))

        return current_alpha_metrics

    def _compute_and_plot_roc_curves(self) -> None:
        """Compute and plot ROC curves for unseen-vs-known binary classification.

        For each outer iteration, predicts **without thresholds** to obtain
        raw ``max_avg_prob`` scores.  The ROC sweeps its own thresholds over
        these scores — different from the alpha-quantile calibration thresholds.
        """
        from sklearn.metrics import roc_curve, auc

        multi = self.multi

        # {clf: {bs: [(fpr, tpr, auroc), ...]}}
        all_roc_curves = {
            clf: {bs: [] for bs in multi.batch_prediction_sizes}
            for clf in multi.classifier_types
        }
        # Parallel accumulator for raw per-iteration max-prob arrays, used to draw
        # unseen+global P/R-vs-confidence curves further down.
        all_pr_curves = {
            clf: {bs: [] for bs in multi.batch_prediction_sizes}
            for clf in multi.classifier_types
        }

        if multi.batch_type == "mix_tp_at_pred":
            self._collect_roc_scores_mix_tp(all_roc_curves, all_pr_curves)
        else:
            self._collect_roc_scores_tp_wise(all_roc_curves, all_pr_curves)

        total_curves = sum(
            len(all_roc_curves[clf][bs])
            for clf in multi.classifier_types
            for bs in multi.batch_prediction_sizes
        )
        if total_curves == 0:
            logger.warning(
                "No ROC curves accumulated — all data was empty or had < 2 unique labels. "
                "Check that Known and Unknown sets have sufficient data."
            )
            return
        logger.info("Accumulated %d ROC curve entries.", total_curves)

        # Aggregate: interpolate to common FPR grid, compute mean/std
        common_fpr = np.linspace(0.0, 1.0, 200)
        roc_data: dict = {}

        for bs in multi.batch_prediction_sizes:
            roc_data[bs] = {}
            for clf in multi.classifier_types:
                curves = all_roc_curves[clf][bs]
                if not curves:
                    continue

                tpr_interps = []
                auroc_vals = []
                precision_vals = []
                recall_vals = []
                f1_vals = []
                for fpr_i, tpr_i, auroc_i, prec_i, rec_i, f1_i in curves:
                    tpr_interp = np.interp(common_fpr, fpr_i, tpr_i)
                    tpr_interp[0] = 0.0
                    tpr_interps.append(tpr_interp)
                    auroc_vals.append(auroc_i)
                    precision_vals.append(prec_i)
                    recall_vals.append(rec_i)
                    f1_vals.append(f1_i)

                mean_tpr = np.mean(tpr_interps, axis=0)
                std_tpr = np.std(tpr_interps, axis=0)
                mean_tpr[-1] = 1.0

                roc_data[bs][clf] = {
                    "mean_fpr": common_fpr,
                    "mean_tpr": mean_tpr,
                    "std_tpr": std_tpr,
                    "mean_auroc": float(np.mean(auroc_vals)),
                    "std_auroc": float(np.std(auroc_vals)),
                    "mean_precision": float(np.mean(precision_vals)),
                    "std_precision": float(np.std(precision_vals)),
                    "mean_recall": float(np.mean(recall_vals)),
                    "std_recall": float(np.std(recall_vals)),
                    "mean_f1": float(np.mean(f1_vals)),
                    "std_f1": float(np.std(f1_vals)),
                }

        self.roc_data = roc_data
        bt_fig_path = self._batch_type_fig_path()

        plot_openset_roc_curves(
            roc_data,
            fig_save_path=bt_fig_path,
            batch_prediction_sizes=multi.batch_prediction_sizes,
        )

        # Metrics table is alpha-dependent → always write, use alpha-keyed path.
        save_openset_metrics_table(
            roc_data,
            fig_save_path=self._alpha_fig_path(multi.alpha_quantile_threshold),
            batch_prediction_sizes=multi.batch_prediction_sizes,
        )

        # Transpose {clf: {bs: [...]}} → {bs: {clf: [...]}} for the P/R plots.
        pr_curves_by_bs: dict = {bs: {} for bs in multi.batch_prediction_sizes}
        for clf in multi.classifier_types:
            for bs in multi.batch_prediction_sizes:
                if all_pr_curves[clf][bs]:
                    pr_curves_by_bs[bs][clf] = all_pr_curves[clf][bs]

        plot_unseen_and_global_pr_vs_confidence(
            pr_curves_by_bs,
            fig_save_path=bt_fig_path,
            batch_prediction_sizes=multi.batch_prediction_sizes,
            unseen_prevalence=getattr(multi, "m_test_size", None),
        )

        plot_unseen_and_global_pr_vs_alpha(
            pr_curves_by_bs,
            fig_save_path=bt_fig_path,
            batch_prediction_sizes=multi.batch_prediction_sizes,
        )

        plot_alpha_roc_curves(
            pr_curves_by_bs,
            fig_save_path=bt_fig_path,
            batch_prediction_sizes=multi.batch_prediction_sizes,
        )

        plot_roc_curves_overlay(
            pr_curves_by_bs,
            roc_data,
            fig_save_path=bt_fig_path,
            batch_prediction_sizes=multi.batch_prediction_sizes,
        )

        logger.info("ROC curves saved to %s", bt_fig_path)

        for bs in multi.batch_prediction_sizes:
            for clf, roc_info in roc_data.get(bs, {}).items():
                logger.info(
                    "ROC summary: bs=%s clf=%s AUROC=%.4f (± %.4f)",
                    bs, clf, roc_info["mean_auroc"], roc_info["std_auroc"],
                )

        if getattr(multi, "openset_fig_cache", False):
            self._save_roc_figscore_cache(roc_data, pr_curves_by_bs)

    def _save_roc_figscore_cache(self, roc_data: dict, pr_curves_by_bs: dict) -> None:
        """Persist the small plot-input structures so the openset figures regenerate fast.

        Writes ``roc_data`` + ``pr_curves_by_bs`` (tens of MB, vs the multi-GB score
        pickle) to ``roc_figscore_cache.pkl`` in the batch_type fig dir. The standalone
        ``scripts/fig_scripts/preview_openset_roc.py`` reloads this to redraw the ROC/PR
        figures without touching the big pickle or re-predicting. ``alpha`` is recorded
        because the prec/rec/f1 inside ``roc_data`` (metrics table only) are alpha-fixed.
        """
        multi = self.multi
        payload = {
            "roc_data": roc_data,
            "pr_curves_by_bs": pr_curves_by_bs,
            "batch_prediction_sizes": list(multi.batch_prediction_sizes),
            "unseen_prevalence": getattr(multi, "m_test_size", None),
            "alpha": multi.alpha_quantile_threshold,
        }
        path = self._batch_type_fig_path() / "roc_figscore_cache.pkl"
        joblib.dump(payload, path, compress=3)
        logger.info("Saved openset ROC fig-score cache to %s", path)

    def _collect_roc_scores_tp_wise(self, all_roc_curves: Dict, all_pr_curves: Dict) -> None:
        """Collect ROC scores for tp_wise open-set."""
        multi = self.multi
        openset_dict = self._load_openset_dict()

        for tp_name, ds_result in openset_dict.items():
            for iteration in ds_result.iterations:
                final_outer_probas = iteration.final_outer_probas
                final_known_val = iteration.final_known_val

                _, _, probs_known = multi._batch_predict_token_pairs_wise(
                    final_outer_probas["Known"],
                    final_known_val,
                    token_pairs=[tp_name],
                    thresholds=None,
                )
                _, _, probs_unknown = multi._batch_predict_token_pairs_wise(
                    final_outer_probas["Unknown"],
                    final_known_val,
                    token_pairs=[tp_name],
                    thresholds=None,
                    mode="Unknown",
                )

                thresholds = extract_thresholds(
                    iteration.max_probas, alpha=multi.alpha_quantile_threshold
                )
                self._accumulate_roc_scores(all_roc_curves, all_pr_curves,
                                            probs_known, probs_unknown,
                                            thresholds=thresholds)

    def _collect_roc_scores_mix_tp(self, all_roc_curves: Dict, all_pr_curves: Dict) -> None:
        """Collect ROC scores for mix_tp_at_pred open-set (aggregated across uplets)."""
        multi = self.multi
        mix_result = self._load_mix_tp_result()
        if mix_result is None:
            return

        for iteration in mix_result.iterations:
            for bs in multi.batch_prediction_sizes:
                if not multi.tp_uplet_dict.get(bs):
                    continue
                bs_data = iteration.final_outer_probas_per_tp_bs.get(bs)
                if not bs_data:
                    continue

                probas_known_bs = {tp: bs_data[tp]["Known"] for tp in bs_data}
                probas_unknown_bs = {tp: bs_data[tp]["Unknown"] for tp in bs_data}
                y_vals_bs = {
                    tp: iteration.final_known_val_per_tp_bs[bs][tp] for tp in bs_data
                }
                all_tps = list(bs_data.keys())

                uc_known = multi._resolve_unique_classes(
                    y_vals_bs, all_tps, None, tp_keyed=True
                )
                uc_unknown = np.array([-1])
                org_k = multi._organize_probas(
                    probas_known_bs, y_vals_bs, all_tps,
                    uc_known, None, tp_keyed=True,
                )
                org_u = multi._organize_probas(
                    probas_unknown_bs, y_vals_bs, all_tps,
                    uc_unknown, "Unknown", tp_keyed=True,
                )

                for tp_uplet in multi.tp_uplet_dict.get(bs, []):
                    uplet_name = get_tp_uplet_name(tp_uplet)
                    uplet_mp = iteration.max_probas_per_uplet.get(uplet_name)
                    if uplet_mp is None:
                        logger.warning(
                            "No inner-loop max_probas for uplet %s, skipping P/R/F1.",
                            uplet_name,
                        )
                        uplet_thresholds = None
                    else:
                        uplet_thresholds = extract_thresholds(
                            uplet_mp, alpha=multi.alpha_quantile_threshold
                        )

                    for clf in multi.classifier_types:
                        _, _, probs_k = multi._batch_predict_mixed_single(
                            clf_name=clf, tp_uplet=tp_uplet, batch_size=bs,
                            unique_classes=uc_known,
                            probas_by_class_tp=org_k[clf],
                            threshold=None, mode=None,
                        )
                        _, _, probs_u = multi._batch_predict_mixed_single(
                            clf_name=clf, tp_uplet=tp_uplet, batch_size=bs,
                            unique_classes=uc_unknown,
                            probas_by_class_tp=org_u[clf],
                            threshold=None, mode="Unknown",
                        )

                        threshold = uplet_thresholds[clf][bs] if uplet_thresholds else 0.0
                        self._accumulate_roc_scores_single(
                            all_roc_curves, all_pr_curves, clf, bs, probs_k, probs_u,
                            threshold=threshold,
                        )

    def _accumulate_roc_scores(self, all_roc_curves, all_pr_curves, probs_known, probs_unknown, thresholds: dict = None):
        """Accumulate ROC curve data from probs dicts (tp_wise format)."""
        multi = self.multi
        for clf in multi.classifier_types:
            for bs in multi.batch_prediction_sizes:
                threshold = thresholds[clf][bs] if thresholds else 0.0
                self._accumulate_roc_scores_single(
                    all_roc_curves, all_pr_curves, clf, bs,
                    probs_known[clf][bs], probs_unknown[clf][bs],
                    threshold=threshold,
                )

    def _accumulate_roc_scores_single(self, all_roc_curves, all_pr_curves, clf, bs, probs_k, probs_u, threshold: float = 0.0):
        """Accumulate one (clf, bs) ROC entry from known/unknown prob dicts."""
        from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score

        known_scores = probs_k["correct"] + probs_k["wrong"]
        unknown_scores = probs_u["correct"] + probs_u["wrong"]

        scores = np.array(known_scores + unknown_scores)
        labels = np.array(
            [1] * len(known_scores) + [0] * len(unknown_scores)
        )

        if len(np.unique(labels)) < 2 or len(scores) == 0:
            return

        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        auroc_val = auc(fpr, tpr)

        y_pred = (scores >= threshold).astype(int)
        precision = precision_score(labels, y_pred, pos_label=0, zero_division=0)
        recall = recall_score(labels, y_pred, pos_label=0, zero_division=0)
        f1 = f1_score(labels, y_pred, pos_label=0, zero_division=0)

        all_roc_curves[clf][bs].append((fpr, tpr, auroc_val, precision, recall, f1))
        all_pr_curves[clf][bs].append({
            "kc": np.asarray(probs_k["correct"], dtype=float),
            "kw": np.asarray(probs_k["wrong"], dtype=float),
            "u": np.asarray(unknown_scores, dtype=float),
        })

    def _load_openset_dict(self) -> Dict[str, OpenSetTokenPairResult]:
        """Load per-dataset open-set checkpoint files (tp_wise mode).

        Returns
        -------
        dict
            ``{tp_name: OpenSetTokenPairResult}``
        """
        result: Dict[str, OpenSetTokenPairResult] = {}
        for tp_name in self.multi.token_pairs:
            path = self._get_openset_ds_path(tp_name)
            if path.exists():
                result[tp_name] = joblib.load(path)
            else:
                logger.warning(
                    f"Warning: OpenSet split for dataset {tp_name} not found at {path}"
                )

        return result

    def _load_mix_tp_result(self):
        """Load the mix_tp_at_pred open-set checkpoint.

        Returns
        -------
        OpenSetMixTPResult or None
        """
        path = self._get_openset_ds_path(f"mix_tp_at_pred_utp{self.multi.unique_tp_in_mix}")
        if path.exists():
            return joblib.load(path)
        return None

    def _batch_type_fig_path(self) -> Path:
        """Return a batch-type-specific subdirectory under fig_save_path."""
        multi = self.multi
        if multi.batch_type in ("mix_tp_at_pred", "mix_tp_at_train"):
            suffix = f"{multi.batch_type}_utp{multi.unique_tp_in_mix}"
        else:
            suffix = multi.batch_type
        path = Path(multi.fig_save_path) / suffix
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _alpha_fig_path(self, alpha: float) -> Path:
        """Alpha-keyed subdirectory under the batch_type fig path.

        Used for figures/tables that depend on a specific alpha value
        (e.g. openset_metrics_bs_*.md, threshold histograms). Alpha-independent
        figures (sweeps over alpha or raw confidence) live one level up at
        ``_batch_type_fig_path()``.
        """
        path = self._batch_type_fig_path() / f"alpha_{_alpha_str(alpha)}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get_openset_ds_path(self, tp_name: str) -> Path:
        """Path to the per-dataset open-set checkpoint pickle file."""
        checkpoint_dir = Path(self.multi.fig_save_path) / "OpenSetXP_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir / f"{tp_name}_OpenSetXP.pkl"

    def _save_openset_split(self, dataset_result, tp_name: str) -> None:
        """Persist one dataset's (or mix_tp_at_pred's) open-set results to disk."""
        path = self._get_openset_ds_path(tp_name)
        joblib.dump(dataset_result, path, compress=3)
        logger.info(f"Saved OpenSet split for dataset {tp_name} at {path}")

    def _upstream_meta(self) -> dict:
        """Snapshot of upstream-relevant config used to validate score-checkpoint reuse.

        Excludes alpha (alpha-only re-runs are explicitly supported) and pure
        plotting toggles. On mismatch, the score pickle must be rebuilt.
        """
        multi = self.multi
        return {
            "train_size": multi.train_size,
            "n_splits": multi.n_splits,
            "test_size": multi.test_size,
            "m_test_size": getattr(multi, "m_test_size", None),
            "openset_m_splits": multi.openset_m_splits,
            "force_class_size": getattr(multi, "force_class_size", None),
            "splitter_type": multi.splitter_type,
            "random_seed": multi.random_seed,
            "token_pairs": sorted(list(multi.token_pairs)),
            "batch_prediction_sizes": sorted(list(multi.batch_prediction_sizes)),
            "unique_tp_in_mix": getattr(multi, "unique_tp_in_mix", None),
            "max_nb_of_uplet": getattr(multi, "max_nb_of_uplet", None),
            "classifier_types": sorted(list(multi.classifier_types)),
            "batch_type": multi.batch_type,
        }

    def _meta_path(self) -> Path:
        """Path to the per-batch_type upstream-meta JSON file inside the checkpoint dir.

        Per-batch_type because a single xp run may iterate `batch_types: [...]`,
        switching `multi.batch_type` on each pass. The score pickles are also
        per-batch_type (different filenames), so the metas track them 1:1.
        """
        checkpoint_dir = Path(self.multi.fig_save_path) / "OpenSetXP_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir / f"OpenSetXP_meta_{self.multi.batch_type}.json"

    def _verify_or_write_meta(self) -> None:
        """Compare current upstream meta to the saved one; raise on mismatch.

        First call (no meta file): writes it. Subsequent calls: compare and raise
        a clear error naming the differing field(s) if upstream params changed.
        """
        meta_path = self._meta_path()
        current = self._upstream_meta()
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    saved = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Could not read %s (%s); rewriting.", meta_path, e)
                with open(meta_path, "w") as f:
                    json.dump(current, f, indent=2, sort_keys=True, default=str)
                return
            diffs = {
                k: (saved.get(k), current.get(k))
                for k in set(saved) | set(current)
                if saved.get(k) != current.get(k)
            }
            if diffs:
                raise RuntimeError(
                    "OpenSet score checkpoint exists but upstream parameters changed. "
                    f"Differing fields (saved → current): {diffs}. "
                    f"To rebuild scores, delete the checkpoint dir at "
                    f"{meta_path.parent} (alpha_quantile_threshold is intentionally "
                    "not part of the check)."
                )
        else:
            with open(meta_path, "w") as f:
                json.dump(current, f, indent=2, sort_keys=True, default=str)
            logger.info("Wrote OpenSet upstream meta to %s", meta_path)
