"""Single-token_pair classification pipeline.

Provides ``SingleTokenPairClassification``, a pipeline for classifying models
using features from a single token-pair token_pair with multi-split cross-validation,
NaN handling, feature normalization, and optional batch prediction.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from audit_llm.Classification.classification_constants import (
    CLASSIFIER_METRICS,
    CLASSIFIER_METRICS_FUN_MAP,
    CLASSIFIERS_TEMPLATES_MAP,
    SPLITTER_MAP,
)
from audit_llm.Classification.Preprocessing_data import (
    FeatureNormalizer,
    fit_transform_normalize,
    preprocess_X,
)
from audit_llm.plot_configs import FIG_CONFIG_SIMPLE_COL_BIG
from audit_llm.plotting.figure_io import save_fig_and_show

import logging
logger = logging.getLogger(__name__)


def persist_token_pair_ban(token_pair: str, token_pairs_banned_path: Path) -> None:
    """Write a token-pair ban to the shared JSON file."""
    import json
    with open(token_pairs_banned_path) as f:
        banned = json.load(f)
    banned[token_pair] = True
    with open(token_pairs_banned_path, "w") as f:
        json.dump(banned, f, indent=4)


class SingleTokenPairClassification:
    """classify models using features across multiple samples with advanced normalization.

    Input data shape: ``(n_samples, n_models, n_features)``.
    The pipeline classifies which model (0 to n_models-1) a set of features belongs to.
    """

    def __init__(
        self,
        features_index: Dict[str, int],
        new_models_idx: Dict[int, str],
        config: Dict[str, Any],
        xp_config: Dict,
        token_pair: str,
        token_pairs_banned_path: Optional[Path] = None,
        model_groups_config: Optional[Dict[str, Any]] = None,
    ):
        self.classifier_types: List[str] = config["classifiers"]
        self.classifier_metrics = config.get("classifier_metrics", CLASSIFIER_METRICS)
        self.n_splits: int = config.get("n_splits", 1)
        self.normalization_methods: Dict[str, str] = config.get("normalization_methods", {})
        self.default_normalization: str = config.get("default_normalization", "auto")
        self.splitter_type: str = config.get("splitter_type", "StratifiedShuffleSplit")
        self.test_size: int = config.get("test_size", 64)
        self.random_seed: int = config.get("random_seed", 42)
        self.force_class_size: Union[int, str, None] = config.get("force_class_size", "auto")

        self.batch_prediction_size: int = config.get("batch_prediction_size", 1)
        self.batch_concat: bool = config.get("batch_concat", False)
        self.prediction_aggregator: str = config.get("prediction_aggregator", "soft_voting")

        self.results: Dict = {}
        self.confusion_matrices: Dict = {}
        self.features_index = features_index
        self.new_models_idx = new_models_idx
        self.normalizer: Optional[FeatureNormalizer] = None
        self.feature_importances: Dict = {}

        self.token_pair = token_pair
        self.token_pairs_banned_path = token_pairs_banned_path

        self.verbose = config.get("verbose", False)

        self.show = xp_config.get("show", False)
        self.save_path = xp_config.get("save_path", "")

        self.already_trained = False
        self.cross_classif = False
        self.dataset_banned = False
        self.ban_reason = ""

        # Group-based classification: remap per-model labels to per-group labels
        self.model_groups_config = model_groups_config
        self.model_to_group: Optional[Dict[int, int]] = None
        self.group_names: Optional[Dict[int, str]] = None
        if model_groups_config is not None:
            from audit_llm.Classification.model_grouping import build_group_mapping
            self.model_to_group, self.group_names = build_group_mapping(
                new_models_idx, model_groups_config
            )

    def _ban_token_pair(self, reason: str = "") -> None:
        """Mark this token pair as banned (no I/O — caller persists)."""
        self.dataset_banned = True
        self.ban_reason = reason
        logger.warning("Token pair '%s' banned: %s", self.token_pair, reason)

    def prepare_data_common(
        self,
        X: np.ndarray,
        strategy: str = "mean",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """NaN handling, flatten 3D->2D, drop all-NaN rows, re-index labels. No balancing or normalization.

        When ``model_groups_config`` is set, per-model labels are remapped to
        per-group labels and models not in any group are excluded.
        """
        # 1) Clean NaNs
        X_cleaned = preprocess_X(X, strategy=strategy, verbose=self.verbose)
        n_samples, n_models, n_features = X_cleaned.shape

        # 2) Flatten & original labels
        X_flat = X_cleaned.reshape(n_samples * n_models, n_features)
        y_flat = np.tile(np.arange(n_models), n_samples)

        # 2b) Group-based classification: remap model labels to group labels
        #     and exclude models not in any group
        if self.model_to_group is not None:
            # Build mask: True for rows whose model is in a group
            group_mask = np.array([mid in self.model_to_group for mid in y_flat])
            X_flat = X_flat[group_mask]
            y_flat = np.array([self.model_to_group[mid] for mid in y_flat[group_mask]], dtype=int)
            # Replace new_models_idx with group names for downstream display
            self.new_models_idx = self.group_names.copy()
            logger.info("Group-based classification: %d rows kept, %d excluded",
                        group_mask.sum(), (~group_mask).sum())

        # 3) Drop rows that are all NaN
        mask_all_nan = np.all(np.isnan(X_flat), axis=1)
        self.removed_model_sample_idx = [(idx // n_models, idx % n_models) for idx in np.where(mask_all_nan)[0]]
        X_no_nan = X_flat[~mask_all_nan]
        y_no_nan = y_flat[~mask_all_nan]

        # 4) Identify removed classes (models or groups)
        present = sorted(np.unique(y_no_nan))
        if self.model_to_group is not None:
            all_expected = sorted(self.group_names.keys())
        else:
            all_expected = list(range(n_models))
        removed = sorted(set(all_expected) - set(present))
        self.removed_models_idx = removed
        if removed:
            removed_names = [self.new_models_idx[idx] for idx in removed]
            self._ban_token_pair(reason=f"all-NaN classes: {removed_names}")

        # 5) Re-index labels contiguously
        mapping = {old: new for new, old in enumerate(present)}
        y_reindexed = np.array([mapping[i] for i in y_no_nan], dtype=int)

        # 6) Save original imbalance
        self.data_for_class_imbalance = y_reindexed.copy()
        counts = Counter(y_reindexed)
        if self.verbose:
            logger.debug(f"Original class distribution: {counts}")

        return X_no_nan, y_reindexed

    def balance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_balancing_mode: str = "none",
        cross_mode: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resample classes to target size for class balancing."""
        from sklearn.utils import resample

        counts = Counter(y)

        if self.force_class_size == "auto":
            # Balance to the smallest class: undersample only, never oversample
            # (no duplicated rows, no leakage). Computed on already-cleaned labels.
            target_size = min(counts.values())
        elif self.force_class_size is not None:
            target_size = self.force_class_size
        elif class_balancing_mode == "undersample":
            target_size = min(counts.values())
        elif class_balancing_mode == "oversample":
            target_size = max(counts.values())
        else:
            target_size = None

        if target_size is None:
            return X, y

        X_pieces, y_pieces = [], []
        oversampled = []  # (cls, len(X_cls)) for classes below target_size
        for cls in sorted(counts):
            mask = y == cls
            X_cls = X[mask]
            y_cls = y[mask]
            replace = len(X_cls) < target_size
            if replace:
                oversampled.append((cls, len(X_cls)))
            resampled = resample(
                X_cls,
                y_cls,
                n_samples=target_size,
                replace=replace,
            )
            X_res, y_res = resampled  # type: ignore
            X_pieces.append(X_res)
            y_pieces.append(y_res)
        if oversampled:
            sizes = [n for _, n in oversampled]
            severe = [(c, n) for c, n in oversampled if n < target_size // 2]
            severe_suffix = (
                "; severe (<%d): " % (target_size // 2)
                + ", ".join(f"cls{c}={n}" for c, n in severe)
                if severe else ""
            )
            logger.debug(
                "Oversampled %d/%d classes to target_size=%d (min=%d, max=%d, median=%d)%s",
                len(oversampled), len(counts), target_size,
                min(sizes), max(sizes), int(np.median(sizes)),
                severe_suffix,
            )
        X_balanced = np.vstack(X_pieces)
        y_balanced = np.concatenate(y_pieces)
        if self.verbose:
            logger.debug(f"Balanced class distribution: {Counter(y_balanced)}")

        return X_balanced, y_balanced

    def _batch_predict(self, clf, X_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
        """Batch predictions where each batch contains elements from the same class."""
        aggregator = getattr(self, "prediction_aggregator", "soft_voting")
        batch_size = getattr(self, "batch_prediction_size", 1)

        if batch_size <= 1:
            return clf.predict(X_val)

        y_pred = np.zeros(len(X_val), dtype=y_val.dtype)
        unique_classes = np.unique(y_val)

        for class_label in unique_classes:
            class_indices = np.where(y_val == class_label)[0]

            for i in range(0, len(class_indices), batch_size):
                batch_indices = class_indices[i : i + batch_size]
                X_batch = X_val[batch_indices]

                if len(X_batch) == 0:
                    continue

                batch_pred = self._aggregate_from_classifier(clf, X_batch, aggregator)
                y_pred[batch_indices] = batch_pred

        return y_pred

    def _aggregate_from_classifier(self, clf, X_batch: np.ndarray, aggregator: str):
        """Aggregate predictions for a batch of samples using a trained sklearn classifier."""
        if aggregator == "soft_voting":
            if hasattr(clf, "predict_proba"):
                try:
                    probas = clf.predict_proba(X_batch)
                    mean_proba = np.mean(probas, axis=0)
                    return clf.classes_[np.argmax(mean_proba)]
                except Exception as e:
                    logger.warning(f"predict_proba failed for clf {type(clf).__name__}: {e}")

            # SVC without probability=True
            if hasattr(clf, "decision_function") and not hasattr(clf, "predict_proba"):
                try:
                    decision_scores = clf.decision_function(X_batch)
                    if decision_scores.ndim == 1:
                        mean_score = np.mean(decision_scores)
                        return clf.classes_[1] if mean_score > 0 else clf.classes_[0]
                    else:
                        mean_scores = np.mean(decision_scores, axis=0)
                        return clf.classes_[np.argmax(mean_scores)]
                except Exception as e:
                    logger.warning(f"decision_function failed for clf {type(clf).__name__}: {e}")

        elif aggregator == "decision_function":
            if hasattr(clf, "decision_function"):
                try:
                    decision_scores = clf.decision_function(X_batch)
                    if decision_scores.ndim == 1:
                        mean_score = np.mean(decision_scores)
                        return clf.classes_[1] if mean_score > 0 else clf.classes_[0]
                    else:
                        mean_scores = np.mean(decision_scores, axis=0)
                        return clf.classes_[np.argmax(mean_scores)]
                except Exception as e:
                    logger.warning(f"decision_function failed for clf {type(clf).__name__}: {e}")

        # Fall back to hard voting
        if aggregator in ["hard_voting", "soft_voting", "decision_function"]:
            predictions = clf.predict(X_batch)
            unique_preds, counts = np.unique(predictions, return_counts=True)
            return unique_preds[np.argmax(counts)]

        else:
            raise ValueError(
                f"Unknown aggregator: {aggregator}. Supported: 'soft_voting', 'hard_voting', 'mean_proba', 'decision_function'"
            )

    def _train_and_load_clfs(self, X_tr: np.ndarray, y_tr: np.ndarray, le: Optional[LabelEncoder] = None) -> None:
        """Clone, fit all classifiers with balanced class weights."""
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_tr)
        y_tr_enc = le.transform(y_tr) if le is not None else y_tr
        self.clfs = {}
        for clf_name in self.classifier_types:
            clf = clone(CLASSIFIERS_TEMPLATES_MAP[clf_name])
            if hasattr(clf, "class_weight"):
                clf.set_params(class_weight="balanced")
                clf.fit(X_tr, y_tr_enc)
            else:
                try:
                    clf.fit(X_tr, y_tr_enc, sample_weight=sample_weights)
                except TypeError:
                    clf.fit(X_tr, y_tr_enc)
            self.clfs[clf_name] = clf

    def _predict_probas(self, X_val: np.ndarray, le: Optional[LabelEncoder] = None) -> Dict[str, np.ndarray]:
        """Get predicted probabilities (used by MultiTokenPairClassification)."""
        y_pred_probas = {}
        for clf_name, clf in self.clfs.items():
            y_pred_probas[clf_name] = clf.predict_proba(X_val)
        return y_pred_probas

    def _predict(self, X_val: np.ndarray, y_val: np.ndarray, X_tr: Optional[np.ndarray] = None, save_results: bool = True) -> None:
        self.y_pred = {}
        for clf_name, clf in self.clfs.items():
            if self.batch_prediction_size > 1:
                self.y_pred[clf_name] = self._batch_predict(clf, X_val, y_val)
            else:
                self.y_pred[clf_name] = clf.predict(X_val)  # type: ignore

        if save_results:
            self._save_prediction_results(y_val)

    def _save_prediction_results(self, y_val: np.ndarray) -> None:
        for clf_name, clf in self.clfs.items():
            y_pred = self.y_pred[clf_name]

            for clf_metric in self.classifier_metrics:
                self.results[clf_name][clf_metric].append(CLASSIFIER_METRICS_FUN_MAP[clf_metric](y_val, y_pred))

            self.confusion_matrices[clf_name].append(confusion_matrix(y_val, y_pred))
            imp = self._extract_importances(clf)
            if imp is None and hasattr(clf, "named_steps"):
                imp = self._extract_importances(list(clf.named_steps.values())[-1])
            if imp is not None:
                self.feature_importances[clf_name].append(imp)

    def fit_evaluate(self, X: np.ndarray, X_test: Optional[np.ndarray] = None) -> Dict:
        """Fit and evaluate classifiers with proper train/test separation (no data leakage)."""
        n_features = X.shape[2]
        assert n_features == len(self.features_index), f"{n_features = }, {len(self.features_index) = }"

        valid_aggregators = ["soft_voting", "hard_voting", "decision_function"]
        if self.prediction_aggregator not in valid_aggregators:
            raise ValueError(f"prediction_aggregator must be one of {valid_aggregators}")

        self.dataset_banned = False

        X_common, y_common = self.prepare_data_common(X)

        # Initialize results containers
        self.results = {name: {m: [] for m in self.classifier_metrics} for name in self.classifier_types}
        self.confusion_matrices = {name: [] for name in self.classifier_types}
        self.feature_importances = {name: [] for name in self.classifier_types}

        if self.dataset_banned:
            return self.results

        if X_test is not None:
            # --- Cross-token_pair evaluation ---
            X_train_balanced, y_train_balanced = self.balance(X_common, y_common)

            # Test data: common preprocessing, NO balancing
            X_test_common, y_test = self.prepare_data_common(X_test)

            X_train_norm, X_test_norm, self.normalizer = fit_transform_normalize(
                X_train_balanced, X_test_common,
                self.features_index, self.normalization_methods, self.default_normalization,
            )
            self.detected_normalization_method_index = self.normalizer.get_normalization_methods_index()

            if not (self.already_trained and self.cross_classif):
                self._train_and_load_clfs(X_train_norm, y_train_balanced)
                self.already_trained = True

            self._predict(X_test_norm, y_test)

        else:
            # --- Multi train/test splits ---
            n_classes = len(np.unique(y_common))
            splitter = SPLITTER_MAP[self.splitter_type](
                n_splits=self.n_splits,
                random_state=self.random_seed,
                test_size=self.test_size * n_classes,
            )

            for split_idx, (train_idx, test_idx) in enumerate(splitter.split(X_common, y_common)):
                X_tr_raw, y_tr_raw = X_common[train_idx], y_common[train_idx]
                X_val_raw, y_val = X_common[test_idx], y_common[test_idx]

                # Balance ONLY training fold
                X_tr_balanced, y_tr = self.balance(X_tr_raw, y_tr_raw)

                # Fit normalizer on training fold, transform both
                X_tr, X_val, self.normalizer = fit_transform_normalize(
                    X_tr_balanced, X_val_raw,
                    self.features_index, self.normalization_methods, self.default_normalization,
                )
                self.detected_normalization_method_index = self.normalizer.get_normalization_methods_index()

                if self.verbose:
                    logger.debug(f"{self.token_pair=}")
                    logger.debug(f"Split {split_idx + 1}/{self.n_splits}:")
                    logger.debug(f"X_tr shape: {X_tr.shape}, X_val shape: {X_val.shape}")
                    logger.debug(f"y_tr shape: {y_tr.shape}, y_val shape: {y_val.shape}")
                    logger.debug(f"Number of NaNs in X_tr: {np.isnan(X_tr).sum()}, X_val: {np.isnan(X_val).sum()}")

                self._train_and_load_clfs(X_tr, y_tr)
                self._predict(X_val, y_val)

                if "accuracy" in self.classifier_metrics:
                    for clf_name in self.classifier_types:
                        split_acc = self.results[clf_name]["accuracy"][-1]
                        logger.debug(
                            "[%s] split %d/%d — %s accuracy: %.4f",
                            self.token_pair, split_idx + 1, self.n_splits, clf_name, split_acc,
                        )

            if self.cross_classif:
                self.already_trained = False

        self.feature_importances = {name: imps for name, imps in self.feature_importances.items() if imps}
        return self.results

    # --- Display / results methods ---

    def _extract_importances(self, estimator):
        """Extract feature importances or coefficients from a fitted estimator."""
        if hasattr(estimator, "feature_importances_"):
            return estimator.feature_importances_
        if hasattr(estimator, "coef_"):
            coef = estimator.coef_
            return np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        return None

    def get_normalization_methods(self) -> Dict:
        """Print and return the resolved normalization methods per feature."""
        features_index_reversed = {value: key for key, value in self.features_index.items()}
        logger.info("Normalization methods:")
        for feature_idx, method in self.detected_normalization_method_index.items():
            feature_name = features_index_reversed[feature_idx]
            if self.normalization_methods.get(feature_name, self.default_normalization) == "auto":
                logger.info(f"(Auto-detected) {feature_name}: {method}")
            else:
                logger.info(f"{feature_name}: {method}")
        return self.detected_normalization_method_index

    def get_summary_results(self) -> pd.DataFrame:
        """Return mean and std of metrics for each classifier as a DataFrame."""
        summary = self.get_raw_summary_results()
        return pd.DataFrame(summary).T

    def get_raw_summary_results(self) -> Dict:
        """Return ``{clf_name: {metric_mean: val, metric_std: val}}``."""
        summary: Dict = {}
        for clf_name in self.classifier_types:
            summary[clf_name] = {}
            for metric in self.classifier_metrics:
                values = self.results[clf_name][metric]
                summary[clf_name][f"{metric}_mean"] = np.mean(values)
                summary[clf_name][f"{metric}_std"] = np.std(values)
        return summary

    def get_confusion_matrices(self) -> Dict:
        """Return raw confusion matrices dict."""
        return self.confusion_matrices

    def plot_results(self, metric: str = "accuracy") -> None:
        """Box plot of metric scores across splits."""
        plt.figure(figsize=(8, 6))

        data = []
        for clf_name in self.classifier_types:
            for i, value in enumerate(self.results[clf_name][metric]):
                data.append({"Classifier": clf_name, "Split": i + 1, metric.capitalize(): value})

        df = pd.DataFrame(data)

        sns.boxplot(x="Classifier", y=metric.capitalize(), data=df)
        plt.title(f"{metric.capitalize()} Scores Across {self.n_splits} Splits")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()

        save_fig_and_show(self.save_path, self.show, "results.pdf")

    def plot_confusion_matrix(self, classifier_name: str = None) -> None:  # type: ignore
        """Plot the mean confusion matrix for a classifier."""
        if not self.confusion_matrices:
            raise ValueError("No confusion matrices available. Run fit_evaluate first.")

        if classifier_name is None:
            summary = self.get_summary_results()
            classifier_name = summary["f1_mean"].idxmax()

        if classifier_name not in self.confusion_matrices:
            raise ValueError(f"No confusion matrix available for classifier '{classifier_name}'")

        mean_cm = np.mean(self.confusion_matrices[classifier_name], axis=0)

        # Bug fix: FIG_CONFIG was undefined — replaced with FIG_CONFIG_SIMPLE_COL_BIG
        plt.figure(**FIG_CONFIG_SIMPLE_COL_BIG)
        sns.heatmap(mean_cm, annot=True, fmt=".1f", cmap="Blues", cbar=True)
        plt.title(f"Mean Confusion Matrix for {classifier_name}")
        plt.xlabel("Predicted Model")
        plt.ylabel("True Model")
        plt.tight_layout()
        if self.save_path:
            spec_save_fig_path = Path(self.save_path) / classifier_name
            spec_save_fig_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(Path(spec_save_fig_path) / "confusion_matrix.pdf")
        if self.show:
            plt.show()
        plt.close()

    def plot_feature_importances(self, classifier_name: str = None, top_n: int = 10) -> None:  # type: ignore
        """Plot top-N feature importances for a classifier."""
        if not self.feature_importances:
            raise ValueError(
                "No feature importances available. "
                "Run fit_evaluate first with importances-capable classifiers."
            )

        if classifier_name is None:
            summary = self.get_summary_results()
            candidates = set(self.feature_importances) & set(summary.index)
            if not candidates:
                classifier_name = next(iter(self.feature_importances))
            else:
                classifier_name = summary.loc[list(candidates), "f1_mean"].idxmax()

        if classifier_name not in self.feature_importances:
            raise ValueError(f"No feature importances stored for '{classifier_name}'")

        imps = np.vstack(self.feature_importances[classifier_name])
        mean_imps = imps.mean(axis=0)

        top_idx = np.argsort(mean_imps)[-top_n:]
        top_vals = mean_imps[top_idx]

        labels = []
        inv_map = {v: k for k, v in self.features_index.items()}
        for idx in top_idx:
            labels.append(inv_map.get(idx, f"Feature {idx}"))

        # Bug fix: FIG_CONFIG was undefined — replaced with FIG_CONFIG_SIMPLE_COL_BIG
        plt.figure(**FIG_CONFIG_SIMPLE_COL_BIG)
        plt.barh(range(top_n), top_vals, align="center")
        plt.yticks(range(top_n), labels)
        plt.xlabel("Mean importance")
        plt.title(f"Top {top_n} features for {classifier_name}")
        plt.tight_layout()
        if self.save_path:
            spec_save_fig_path = Path(self.save_path) / classifier_name
            spec_save_fig_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(Path(spec_save_fig_path) / "feature_importance.pdf")
        if self.show:
            plt.show()
        plt.close()

    def plot_importance_heatmap(self, normalize: bool = True) -> None:
        """Plot a heatmap of mean feature importances for each classifier."""
        clf_names = list(self.feature_importances.keys())
        imps = np.array(
            [np.mean(self.feature_importances[name], axis=0) for name in clf_names]
        )

        if normalize:
            row_sums = imps.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            imps = imps / row_sums

        inv_map = {idx: name for name, idx in self.features_index.items()}
        feature_labels = [inv_map[i] if i in inv_map else f"f{i}" for i in range(imps.shape[1])]

        # Bug fix: FIG_CONFIG was undefined — replaced with FIG_CONFIG_SIMPLE_COL_BIG
        fig, ax = plt.subplots(**FIG_CONFIG_SIMPLE_COL_BIG)
        cax = ax.imshow(imps, aspect="auto", interpolation="nearest")
        ax.set_yticks(np.arange(len(clf_names)))
        ax.set_yticklabels(clf_names)
        ax.set_xticks(np.arange(len(feature_labels)))
        ax.set_xticklabels(feature_labels, rotation=90)
        ax.set_xlabel("Feature")
        ax.set_ylabel("Classifier")
        ax.set_title("Mean Feature Importances (normalized)" if normalize else "Mean Feature Importances")
        fig.colorbar(cax, ax=ax, label="Importance")
        plt.tight_layout()

        if self.save_path:
            plt.savefig(Path(self.save_path) / "importance_heatmap.pdf")
        if self.show:
            plt.show()
        plt.close()

    def plot_model_index(self, verbose: bool = False) -> List[str]:
        """Display kept and removed models."""
        kept_models_idx = [
            model_idx for model_idx in self.new_models_idx.keys() if model_idx not in self.removed_models_idx
        ]

        kept_model_names_display = [
            f"{fig_idx}: {self.new_models_idx[model_idx]}" for fig_idx, model_idx in enumerate(kept_models_idx, start=0)
        ]

        removed_model_names_display = [self.new_models_idx[model_idx] for model_idx in self.removed_models_idx]

        model_index_text = "Displayed Models:\n"
        model_index_text += "\n".join(kept_model_names_display) or "None"
        model_index_text += "\n\nRemoved Models:\n"
        model_index_text += "\n".join(removed_model_names_display) or "None"

        if verbose:
            logger.debug(model_index_text)

        return kept_model_names_display
