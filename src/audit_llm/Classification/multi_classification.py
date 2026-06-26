"""Multi-token_pair classification pipeline (closed-set and shared infrastructure).

Provides ``MultiTokenPairClassification`` for batch classification across
multiple token_pairs with support for token_pair-wise and across-token_pair batch
prediction, token_pair mixing, and DCA analysis.

Open-set classification is handled by ``OpenSetClassification`` in
``openset_classification.py`` via composition.
"""

from __future__ import annotations

import pickle
import logging
from collections import Counter, defaultdict
from pathlib import Path
from random import sample
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

from audit_llm.Classification.classification_constants import (
    CLASSIFIER_METRICS,
    CLASSIFIER_METRICS_FUN_MAP,
    CLASSIFIERS_TEMPLATES_MAP,
    SPLITTER_MAP,
)
from audit_llm.Classification.batch_results import spill_full_probas, summarize_metrics
from audit_llm.Classification.token_pair_mixing import mix_by_class_multi_tp, sequential_concat_multi_tp
from audit_llm.Classification.Preprocessing_data import FeatureNormalizer, fit_transform_normalize
from audit_llm.Classification.single_classification import SingleTokenPairClassification, persist_token_pair_ban
from audit_llm.plotting.figure_io import save_fig_and_show
from audit_llm.xp_tools.model_filtering import (
    full_var_model_name_to_original_model_name,
    original_model_name_to_safe_var_model_idx_mapper,
)
from audit_llm.system_utils import print_list_diff
from audit_llm.xp_tools import (
    get_tp_uplet_name,
    get_tp_uplets_dict_from_group,
)


logger = logging.getLogger(__name__)


class MultiTokenPairClassification:
    """Batch classification across multiple token_pairs.

    Supports ``tp_wise`` (predict per-token_pair) and ``mix_tp_at_pred`` (mixed-token_pair
    batches) modes. Open-set experiments are delegated to ``OpenSetClassification``.
    """

    def __init__(
        self,
        X_s: Dict[str, np.ndarray],
        features_index: Dict[str, int],
        new_models_idx: Dict[int, str],
        config: Dict[str, Any],
        batch_type: str,
        token_pairs: List[str],
        xp_config: Dict,
        fig_save_path: Path,
        train_size: int = 200,
        token_pairs_banned_path: Optional[Path] = None,
        model_groups_config: Optional[Dict[str, Any]] = None,
    ):
        self.batch_type = batch_type
        self.classifier_types: List[str] = config["classifiers"]
        self.classifier_metrics = config.get("classifier_metrics", CLASSIFIER_METRICS)

        self.batch_prediction_sizes: List[int] = list(config.get("batch_prediction_sizes", [1]))
        if self.batch_type in ("mix_tp_at_pred", "mix_tp_at_train") and 1 in self.batch_prediction_sizes:
            self.batch_prediction_sizes.remove(1)

        self.prediction_aggregator: str = config.get("prediction_aggregator", "soft_voting")

        self.n_splits: int = config.get("n_splits", 1)
        self.splitter_type: str = config.get("splitter_type", "StratifiedShuffleSplit")
        self.train_size: int = train_size
        self.test_size: int = config.get("test_size", 64)
        self.random_seed: int = config.get("random_seed", 42)
        self._rng = np.random.default_rng(self.random_seed)
        self.top_k = config.get("top_k_token_pair_to_show", 3)
        self.force_class_size: Union[int, str, None] = config.get("force_class_size", "auto")
        self.openset: bool = config.get("openset", False)
        self.m_test_size: float = config.get("m_test_size", 0.5)
        self.openset_m_splits: int = config.get("openset_m_splits", 2)
        self.alpha_quantile_threshold: float = config.get("alpha_quantile_threshold", 0.05)
        self.alpha_trade_off_show: bool = config.get("alpha_trade_off_show", False)
        self.roc_curve_show: bool = config.get("roc_curve_show", False)
        self.store_prediction_probas: bool = config.get("store_prediction_probas", False)
        self.openset_fig_cache: bool = config.get("openset_fig_cache", False)

        self.plot_all_token_pairs = config.get("plot_all_token_pairs", True)

        self.features_index = features_index
        self.normalization_methods: Dict[str, str] = config.get("normalization_methods", {})
        self.default_normalization: str = config.get("default_normalization", "auto")
        self.normalizer = None
        self.feature_importances: Dict = {}

        self.unique_tp_in_mix: Optional[Union[int, str]] = config.get("unique_tp_in_mix", None)
        if self.batch_type in ("mix_tp_at_pred", "mix_tp_at_train"):
            assert self.unique_tp_in_mix is not None, f"unique_tp_in_mix must be specified for {self.batch_type} batch type"
            assert config.get("max_nb_of_uplet") is not None, f"max_nb_of_uplet must be specified for {self.batch_type} batch type"
            self.tp_uplet_dict: Dict[int, List[List[str]]] = get_tp_uplets_dict_from_group(
                group="FLiPS",
                max_nb_of_uplet=config.get("max_nb_of_uplet", 10),
                token_pairs=token_pairs,
                batch_prediction_sizes=self.batch_prediction_sizes,
                unique_elements=self.unique_tp_in_mix,
            )
        else:
            self.tp_uplet_dict = None

        self.xp_config = xp_config
        self.verbose = config.get("verbose", False)
        self.show = xp_config.get("show", False)
        self.LLMmap_compare = xp_config.get("LLMmap_compare", True)
        self.fig_save_path = fig_save_path
        self.token_pairs_banned_path = token_pairs_banned_path

        self.model_index = new_models_idx  # {model_idx: model_name}
        self.model_to_safe_model = original_model_name_to_safe_var_model_idx_mapper(new_models_idx)

        # store config values directly instead of a lambda factory
        self._classif_features_index = features_index
        self._classif_new_models_idx = new_models_idx
        self._classif_config = config
        self._classif_xp_config = xp_config
        self._classif_token_pairs_banned_path = token_pairs_banned_path
        self._model_groups_config = model_groups_config

        # When model grouping is active, update model_index to reflect groups
        if model_groups_config is not None:
            from audit_llm.Classification.model_grouping import build_group_mapping
            _, group_names = build_group_mapping(new_models_idx, model_groups_config)
            self.model_index = group_names

        self.X_s = X_s
        self.token_pairs = list(X_s.keys())

    def _resolve_unique_tp(self, bs: int) -> int:
        """Resolve unique_tp_in_mix value: 'max' → bs, int → min(bs, value), None → bs."""
        if self.unique_tp_in_mix is None or self.unique_tp_in_mix == 'max':
            return bs
        return min(bs, self.unique_tp_in_mix)

    def _create_single_token_pair_classif(self, token_pair: str) -> SingleTokenPairClassification:
        """Instantiate a SingleTokenPairClassification for the given token_pair."""
        return SingleTokenPairClassification(
            self._classif_features_index,
            self._classif_new_models_idx,
            self._classif_config,
            self._classif_xp_config,
            token_pair,
            token_pairs_banned_path=self._classif_token_pairs_banned_path,
            model_groups_config=self._model_groups_config,
        )

    def _create_normalizer(self) -> FeatureNormalizer:
        """Create a new FeatureNormalizer with the current config."""
        return FeatureNormalizer(
            self.features_index, self.normalization_methods, self.default_normalization
        )

    def _prepare_data(
        self, X: np.ndarray, batch_type: str, batch_size: Optional[int] = None, tp_name: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Common preprocessing + truncation (no balancing, no normalization).

        Balancing is the caller's responsibility and must only be applied to the
        training fold after the train/test split to avoid data leakage.
        """
        tp_key = tp_name if tp_name is not None else next(iter(self.X_s.keys()))
        blank_classifier = self._create_single_token_pair_classif(tp_key)
        X_common, y_common = blank_classifier.prepare_data_common(X)

        if blank_classifier.dataset_banned and self.token_pairs_banned_path is not None:
            persist_token_pair_ban(tp_key, self.token_pairs_banned_path)

        # Truncate each class to at most force_class_size (no upsampling => no leakage).
        # "auto" resolves to the smallest class count of this (cleaned) token-pair.
        truncation = self._resolve_force_class_size(y_common, tp_name=tp_key)
        X_trunc, y_trunc = self._reducing_token_pair(X_common, y_common, truncation=truncation)
        new_model_index = blank_classifier.plot_model_index()

        if hasattr(self, "model_index_consistency_check") and self.model_index_consistency_check is not None:
            if new_model_index != self.model_index_consistency_check:
                print_list_diff(new_model_index, self.model_index_consistency_check)
                raise ValueError(
                    f"Model index mismatch!: {new_model_index = }, {self.model_index_consistency_check =}"
                )
        else:
            self.model_index_consistency_check = new_model_index

        return X_trunc, y_trunc

    def _aggregate_from_probabilities(self, batch_prediction_probas: np.ndarray, threshold: Optional[float] = None) -> int:
        """Aggregate batch of probability arrays via soft/hard voting."""
        aggregator = self.prediction_aggregator

        if aggregator == "soft_voting":
            mean_proba = np.mean(batch_prediction_probas, axis=0)
            if threshold is not None:
                max_proba = np.max(mean_proba)
                if max_proba < threshold:
                    return -1  # indicating "unknown"
            return int(np.argmax(mean_proba))

        elif aggregator == "hard_voting":
            hard_votes = np.argmax(batch_prediction_probas, axis=1)
            return int(np.bincount(hard_votes).argmax())

        else:
            raise ValueError(f"Unknown aggregator: {aggregator}. Supported: 'soft_voting', 'hard_voting'")

    def _train_and_predict_proba(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: Union[np.ndarray, Dict[str, np.ndarray]],
        tp_name: str,
        le: Optional[LabelEncoder] = None,
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """Train on X_tr and return predicted probabilities on X_val."""
        single = self._create_single_token_pair_classif(tp_name)
        single._train_and_load_clfs(X_tr, y_tr, le=le)
        if isinstance(X_val, Dict):
            probas = {key: single._predict_probas(X_val_value, le=le) for key, X_val_value in X_val.items()}
        else:
            probas = single._predict_probas(X_val, le=le)
        return probas

    def batch_classification(self) -> None:
        """Run batch classification and store results for downstream analysis."""
        self.openset_roc_data: dict | None = None
        if self.openset:
            # Delegate open-set to OpenSetClassification via composition
            from audit_llm.Classification.openset_classification import OpenSetClassification
            openset_clf = OpenSetClassification(self)
            openset_clf.run()
            self.openset_roc_data = openset_clf.roc_data or None
        elif self.batch_type == "mix_tp_at_train":
            results_map, batch_size_confusion_matrices_map, probs_save_map = self._batch_classification_mix_tp_at_train()
            self.batch_size_results_map = results_map
            self.batch_size_confusion_matrices_map = batch_size_confusion_matrices_map
            self.probs_save_map = probs_save_map
        else:
            results_map, batch_size_confusion_matrices_map, probs_save_map = self._batch_classification_across_token_pairs(
                batch_type=self.batch_type, tp_uplet_dict=self.tp_uplet_dict
            )
            self.batch_size_results_map = results_map
            self.batch_size_confusion_matrices_map = batch_size_confusion_matrices_map
            self.probs_save_map = probs_save_map
            try:
                self._plot_max_predict_probas(probs_save_map, fig_name="tp_wise_max_pred_probs.pdf")
            except Exception as e:
                logger.warning(f"Could not plot max predict probas: {e}")

    def _resolve_force_class_size(self, y: np.ndarray, tp_name: Optional[str] = None) -> Optional[int]:
        """Resolve ``force_class_size`` to a concrete per-class cap.

        - ``int``: used as-is (cap each class to at most this many rows).
        - ``"auto"``: the minimum class count of *y* — balance to the smallest class
          via truncation only (no oversampling, no leakage). *y* must already be
          cleaned (post NaN-drop / group-remap), which it is at every call site.
        - ``None``: no cap.

        Raises ``ValueError`` when the resolved cap is not strictly greater than
        ``test_size`` (the ``StratifiedShuffleSplit`` feasibility invariant), turning a
        cryptic deep-sklearn crash into an actionable, token-pair-named error.
        """
        fcs = self.force_class_size
        if fcs == "auto":
            counts = Counter(y)
            resolved: Optional[int] = min(counts.values()) if counts else None
        else:
            resolved = fcs
        if (
            resolved is not None
            and isinstance(self.test_size, int)
            and not isinstance(self.test_size, bool)
            and resolved <= self.test_size
        ):
            where = f" for token_pair '{tp_name}'" if tp_name else ""
            raise ValueError(
                f"force_class_size resolved to {resolved} per class{where}, which is "
                f"<= test_size ({self.test_size}); StratifiedShuffleSplit cannot carve a "
                "valid test fold. Increase the available data, lower test_size, or set an "
                "explicit force_class_size > test_size."
            )
        return resolved

    def _reducing_token_pair(self, X: np.ndarray, y: np.ndarray, truncation: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Truncate each class to at most *truncation* samples."""
        if truncation is None:
            return X, y

        X_parts = []
        y_parts = []
        for label in np.unique(y):
            mask = y == label
            X_lbl = X[mask][:truncation]
            y_lbl = y[mask][:truncation]
            X_parts.append(X_lbl)
            y_parts.append(y_lbl)

        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        return X, y

    def _class_split(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split unique classes into train/test sets for open-set experiments."""
        classes = np.unique(y)
        nb_of_test_models = int(self.m_test_size * len(classes))
        logger.info(f"{nb_of_test_models =} (in openset setting)")
        test_classes = self._rng.choice(classes, size=nb_of_test_models, replace=False)
        train_classes = np.setdiff1d(classes, test_classes)
        return train_classes, test_classes

    def _initialize_results_and_conf_structure(self) -> None:
        """Initialize nested dicts for results and confusion matrices."""
        self.results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        self.confusion_matrices = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        if self.batch_type in ("mix_tp_at_pred", "tp_wise", "mix_tp_at_train"):
            self.results = {
                bs: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
                for bs in self.batch_prediction_sizes
            }
            self.confusion_matrices = {
                bs: defaultdict(lambda: defaultdict(list)) for bs in self.batch_prediction_sizes
            }
        else:
            raise ValueError(f"Unknown batch_type: {self.batch_type}. Supported: 'mix_tp_at_pred', 'tp_wise', 'mix_tp_at_train'")

    def _train_predict_single_tp_split(
        self,
        tp_name: str,
        X_p: np.ndarray,
        y_p: np.ndarray,
        tr_idx: np.ndarray,
        te_idx: np.ndarray,
        truncation: int,
    ) -> Tuple[Union[Dict[str, np.ndarray], np.ndarray], np.ndarray]:
        """Split, balance, reduce, normalize, and predict probabilities for one token-pair fold.

        Returns (probas, y_val).
        """
        X_tr, X_val = X_p[tr_idx], X_p[te_idx]
        y_tr, y_val = y_p[tr_idx], y_p[te_idx]
        logger.debug("X_tr.shape = %s, X_val.shape = %s", X_tr.shape, X_val.shape)

        # Balance training fold only (after split to avoid leakage)
        _blank = self._create_single_token_pair_classif(tp_name)
        X_tr, y_tr = _blank.balance(X_tr, y_tr)

        X_tr, y_tr = self._reducing_token_pair(X_tr, y_tr, truncation=truncation)

        # Normalize: fit on train, transform both
        X_tr, X_val, _ = fit_transform_normalize(
            X_tr, X_val,
            self.features_index, self.normalization_methods, self.default_normalization,
        )

        probas = self._train_and_predict_proba(X_tr, y_tr, X_val, tp_name)
        if isinstance(probas, dict):
            probas = {k: v.astype(np.float32, copy=False) for k, v in probas.items()}
        else:
            probas = probas.astype(np.float32, copy=False)
        return probas, y_val

    def _batch_classification_across_token_pairs(
        self, batch_type: str, tp_uplet_dict: Optional[Dict[int, List[List[str]]]] = None
    ):
        """Closed-set classification across token_pairs (open-set delegated to OpenSetClassification)."""
        self._initialize_results_and_conf_structure()

        n_classes = len(self.model_index)
        splitter = SPLITTER_MAP[self.splitter_type](
            n_splits=self.n_splits,
            random_state=self.random_seed,
            test_size=self.test_size * n_classes,
        )

        # Initialize dictionaries based on batch_type
        if self.batch_type == "tp_wise":
            y_vals = {i: {} for i in range(self.n_splits)}
            y_preds_probas = {i: {} for i in range(self.n_splits)}
        elif self.batch_type == "mix_tp_at_pred":
            y_vals = {i: {bs: {} for bs in self.batch_prediction_sizes} for i in range(self.n_splits)}
            y_preds_probas = {i: {bs: {} for bs in self.batch_prediction_sizes} for i in range(self.n_splits)}

        # train on each token_pair, each split
        for tp_name, X in self.X_s.items():
            if logger.isEnabledFor(logging.DEBUG):
                n_samples, n_models_col = X.shape[0], X.shape[1]
                logger.debug(f"\n=== DEBUG NaN diagnostic for token_pair '{tp_name}' ===")
                logger.debug(f"X shape: {X.shape}")
                for col_idx in range(n_models_col):
                    col_data = X[:, col_idx, :]
                    nan_pct = np.isnan(col_data).mean() * 100
                    all_nan_rows = np.all(np.isnan(col_data), axis=1).sum()
                    model_name = self.model_index.get(col_idx, f"col_{col_idx}")
                    logger.debug(f"  col {col_idx} ({model_name}): {nan_pct:.1f}% NaN, {all_nan_rows}/{n_samples} all-NaN rows")
                logger.debug("=== END DEBUG ===\n")

            X_p, y_p = self._prepare_data(X, batch_type=batch_type, tp_name=tp_name)

            for split_idx, (tr_idx, te_idx) in enumerate(splitter.split(X_p, y_p)):
                if self.batch_type == "tp_wise":
                    probas, y_val = self._train_predict_single_tp_split(
                        tp_name, X_p, y_p, tr_idx, te_idx, truncation=self.train_size,
                    )
                    y_preds_probas[split_idx][tp_name] = probas
                    y_vals[split_idx][tp_name] = y_val

                elif self.batch_type == "mix_tp_at_pred":
                    for bs in self.batch_prediction_sizes:
                        assert self.unique_tp_in_mix is not None
                        if not self.tp_uplet_dict.get(bs):
                            continue  # skip batch sizes with no valid uplets

                        probas, y_val = self._train_predict_single_tp_split(
                            tp_name, X_p, y_p, tr_idx, te_idx,
                            truncation=self.train_size // self._resolve_unique_tp(bs),
                        )
                        y_preds_probas[split_idx][bs][tp_name] = probas
                        y_vals[split_idx][bs][tp_name] = y_val

        # now do the batch predictions & record metrics/confusions
        probs_save_all_splits = []
        for split_idx in range(self.n_splits):

            y_true_map, y_pred_map, probs_save = self._batch_predict_no_mix_train(
                y_preds_probas, y_vals, split_idx, tp_uplet_dict
            )

            probs_save_all_splits.append(probs_save)

            for tp in y_true_map:
                self._save_prediction_results(y_true_map[tp], y_pred_map[tp], tp=tp)

            del y_preds_probas[split_idx]
            del y_vals[split_idx]

        batch_size_results_map = self.results
        batch_size_confusion_matrix_map = self.confusion_matrices

        if self.batch_type == "mix_tp_at_pred":
            # Structure: {tp_uplet: {clf: {bs: probs}}}
            # Each tp_uplet only has data for one bs (len(uplet) == bs).
            # Reshape to {bs: {tp_uplet: {clf: {key: concat_across_splits}}}}
            probs_save_map = {}
            if probs_save_all_splits:
                for tp_uplet in probs_save_all_splits[0]:
                    for clf in probs_save_all_splits[0][tp_uplet]:
                        for bs in probs_save_all_splits[0][tp_uplet][clf]:
                            cell_per_split = [s[tp_uplet][clf][bs] for s in probs_save_all_splits]
                            cell_id = f"uplet_{tp_uplet}__{clf}__bs{bs}"
                            probs_save_map.setdefault(bs, {}).setdefault(tp_uplet, {})[clf] = (
                                self._concat_cell_across_splits(cell_per_split, cell_id)
                            )
        elif self.batch_type == "tp_wise":
            probs_save_map = {}
            if probs_save_all_splits:
                for tp in probs_save_all_splits[0]:
                    for clf in probs_save_all_splits[0][tp]:
                        for bs in probs_save_all_splits[0][tp][clf]:
                            cell_per_split = [s[tp][clf][bs] for s in probs_save_all_splits]
                            cell_id = f"tp_{tp}__{clf}__bs{bs}"
                            probs_save_map.setdefault(bs, {}).setdefault(tp, {})[clf] = (
                                self._concat_cell_across_splits(cell_per_split, cell_id)
                            )
        else:
            probs_save_map = {}

        return batch_size_results_map, batch_size_confusion_matrix_map, probs_save_map

    def _save_prediction_results(
        self,
        y_true: dict,
        y_pred: dict,
        tp: str,
    ) -> None:
        """Record metrics and confusion matrices for one token_pair."""
        for clf in self.classifier_types:
            for bs in self.batch_prediction_sizes:
                if bs not in y_true[clf]:
                    assert self.batch_type != "tp_wise"
                    continue
                for metric in self.classifier_metrics:
                    fn = CLASSIFIER_METRICS_FUN_MAP[metric]
                    val = fn(y_true[clf][bs], y_pred[clf][bs])
                    self.results[bs][tp][clf][metric].append(val)

                cm = confusion_matrix(
                    y_true[clf][bs],
                    y_pred[clf][bs],
                    labels=(
                        np.append(np.arange(len(self.model_index)), -1)
                        if self.openset
                        else np.arange(len(self.model_index))
                    ),
                )
                self.confusion_matrices[bs][tp][clf].append(cm)

    def _batch_predict_no_mix_train(self, y_preds_probas, y_vals, split_idx, tp_uplet_dict):
        """Route to token_pair-wise or mixed-wise batch prediction."""
        if self.batch_type == "tp_wise":
            y_true_map, y_pred_map, probs_save = self._batch_predict_token_pairs_wise(
                y_preds_probas[split_idx], y_vals[split_idx], split_idx=split_idx,
            )
        elif self.batch_type == "mix_tp_at_pred":
            assert tp_uplet_dict is not None
            y_true_map, y_pred_map, probs_save = self._batch_predict_mixed_wise(
                y_preds_probas[split_idx], y_vals[split_idx], tp_uplet_dict=tp_uplet_dict,
                split_idx=split_idx,
            )
        else:
            raise ValueError("Unknown batch type: %s" % self.batch_type)
        return y_true_map, y_pred_map, probs_save

    def _resolve_unique_classes(
        self, y_vals, token_pairs: List[str], mode: Optional[str], tp_keyed: bool = False
    ) -> np.ndarray:
        """Determine which class labels to iterate over.

        Parameters
        ----------
        tp_keyed : bool
            When True, ``y_vals`` is a dict keyed by token-pair name even in
            open-set mode (used by mix_tp_at_pred + open-set).
        """
        if mode == "Unknown":
            return np.array([-1])
        if self.openset and not tp_keyed:
            return np.unique(y_vals)
        if not token_pairs:
            raise ValueError(
                "token_pairs list is empty in _resolve_unique_classes. "
                "This likely means tp_uplet_dict has no uplets for the current batch_size. "
                "Check that batch_prediction_sizes and unique_tp_in_mix are compatible with the available token pairs."
            )
        return np.unique(y_vals[token_pairs[0]])

    def _organize_probas(self, y_preds_probas, y_vals, token_pairs, unique_classes, mode, tp_keyed: bool = False):
        """Reorganize raw probabilities into ``result[clf][class_label][token_pair]``.

        Parameters
        ----------
        tp_keyed : bool
            When True, ``y_preds_probas`` and ``y_vals`` are dicts keyed by
            token-pair name even in open-set mode (used by mix_tp_at_pred +
            open-set).
        """
        result: Dict = {}
        for clf_name in self.classifier_types:
            result[clf_name] = {}
            for class_label in unique_classes:
                result[clf_name][class_label] = {}
                for token_pair in token_pairs:
                    if mode == "Unknown":
                        if tp_keyed or not self.openset:
                            probas = y_preds_probas[token_pair][clf_name]
                        else:
                            probas = y_preds_probas[clf_name]
                    elif self.openset and not tp_keyed:
                        assert len(token_pairs) == 1, "Openset mode supports only one token_pair at a time."
                        indices = np.where(y_vals == class_label)[0]
                        probas = y_preds_probas[clf_name][indices]
                    else:
                        indices = np.where(y_vals[token_pair] == class_label)[0]
                        probas = y_preds_probas[token_pair][clf_name][indices]
                    result[clf_name][class_label][token_pair] = probas
        return result

    def _probas_sidecar_dir(self) -> Path:
        """Directory for spilled ``full_probas`` / ``full_y_true`` sidecar files.

        Namespaced by ``batch_type`` (and ``unique_tp_in_mix`` for mixed batch types) so that
        successive steps within the same train_size never overwrite each other's sidecars.
        """
        sub = self.batch_type
        if self.batch_type in ("mix_tp_at_pred", "mix_tp_at_train"):
            sub = f"{self.batch_type}_utp{self.unique_tp_in_mix}"
        return Path(self.fig_save_path) / "probas_sidecars" / sub

    def _concat_cell_across_splits(self, cell_per_split, cell_id: str) -> Dict:
        """Concat one ``(uplet|tp, clf, bs)`` cell's per-split dicts into one final dict.

        For inline-array keys (``correct``, ``wrong``, also ``full_probas``/``full_y_true``
        when openset paths leave them inline), uses ``np.concatenate``. For sidecar-stored
        keys (``full_probas_path`` present), loads each per-split ``.npz``, concatenates the
        arrays, writes one final ``.npz`` keyed by ``cell_id`` (without ``split{i}__`` prefix),
        deletes the per-split sidecars, and stores the final path in the returned dict.
        """
        result: Dict = {}
        for key in cell_per_split[0].keys():
            if key == "full_probas_path":
                per_split_paths = [c["full_probas_path"] for c in cell_per_split]
                fps, fys = [], []
                for p in per_split_paths:
                    with np.load(p) as data:
                        fps.append(data["full_probas"])
                        fys.append(data["full_y_true"])
                concat_fp = np.concatenate(fps)
                concat_fy = np.concatenate(fys)
                final_path = self._probas_sidecar_dir() / f"{cell_id}.npz"
                np.savez(final_path, full_probas=concat_fp, full_y_true=concat_fy)
                for p in per_split_paths:
                    if str(p) != str(final_path):
                        Path(p).unlink(missing_ok=True)
                result["full_probas_path"] = str(final_path)
            else:
                result[key] = np.concatenate([c[key] for c in cell_per_split])
        return result

    def _batch_predict_token_pairs_wise(
        self,
        y_preds_probas,
        y_vals,
        token_pairs: Optional[List[str]] = None,
        thresholds: Optional[Dict[str, Dict[int, float]]] = None,
        mode: Optional[str] = None,
        *,
        split_idx: Optional[int] = None,
    ):
        """Batch-predict across token_pairs, classifiers, and batch sizes."""
        token_pairs = list(self.X_s.keys()) if token_pairs is None else token_pairs
        unique_classes = self._resolve_unique_classes(y_vals, token_pairs, mode)

        probas_by_clf_class_tp = self._organize_probas(y_preds_probas, y_vals, token_pairs, unique_classes, mode)

        y_pred: Dict = {}
        y_true: Dict = {}
        probs_save: Dict = {}

        for token_pair in token_pairs:
            y_pred[token_pair] = {}
            y_true[token_pair] = {}
            probs_save[token_pair] = {}

            for clf_name in self.classifier_types:
                y_pred[token_pair][clf_name] = {}
                y_true[token_pair][clf_name] = {}
                probs_save[token_pair][clf_name] = {}

                for bs in self.batch_prediction_sizes:
                    preds, trues, probs = self._batch_predict_single(
                        clf_name=clf_name,
                        token_pair=token_pair,
                        batch_size=bs,
                        unique_classes=unique_classes,
                        probas_by_class=probas_by_clf_class_tp[clf_name],
                        threshold=thresholds[clf_name][bs] if thresholds else None,
                        mode=mode,
                        split_idx=split_idx,
                    )
                    y_pred[token_pair][clf_name][bs] = preds
                    y_true[token_pair][clf_name][bs] = trues
                    probs_save[token_pair][clf_name][bs] = probs

        # Flatten if single token_pair
        if len(token_pairs) == 1:
            key = token_pairs[0]
            y_true, y_pred, probs_save = y_true[key], y_pred[key], probs_save[key]

        return y_true, y_pred, probs_save

    def _batch_predict_single(self, clf_name, token_pair, batch_size, unique_classes, probas_by_class, threshold, mode, *, split_idx: Optional[int] = None):
        """Run batch prediction for one (clf, token_pair, batch_size) combination."""
        save_dca = self.xp_config.get("save_dca_showcase_data", False)
        preds = []
        trues = []
        probs: Dict = {"correct": [], "wrong": []}
        if self.store_prediction_probas:
            probs["full_probas"] = []
            probs["full_y_true"] = []

        for class_label in unique_classes:
            class_probas = probas_by_class[class_label][token_pair]
            n_samples = len(class_probas)
            true_label = -1 if mode == "Unknown" else class_label

            dca_true, dca_safe, dca_top1 = [], [], []

            for i in range(0, n_samples, batch_size):
                batch = class_probas[i : i + batch_size]
                if len(batch) < batch_size:
                    break

                batch_pred = self._aggregate_from_probabilities(batch, threshold=threshold)
                preds.append(batch_pred)
                trues.append(true_label)

                avg_probs = np.mean(batch, axis=0)
                max_avg_prob = np.max(avg_probs)
                bucket = "correct" if batch_pred == class_label else "wrong"
                probs[bucket].append(max_avg_prob)

                if self.store_prediction_probas:
                    probs["full_probas"].append(avg_probs)
                    probs["full_y_true"].append(true_label)

                if save_dca:
                    original_name = full_var_model_name_to_original_model_name(self.model_index[class_label])
                    safe_label = self.model_to_safe_model[original_name]
                    dca_safe.append(avg_probs[safe_label])
                    dca_true.append(avg_probs[class_label])
                    dca_top1.append(self.model_index[int(batch_pred)])

            if save_dca:
                class_key = self.model_index[class_label]
                probs[f"{class_key}"] = dca_true
                probs[f"safe_{class_key}"] = dca_safe
                probs[f"top1_pred_{class_key}"] = dca_top1

        if self.store_prediction_probas:
            probs["full_probas"] = np.array(probs["full_probas"], dtype=np.float32)  # (n_preds, n_classes)
            probs["full_y_true"] = np.array(probs["full_y_true"], dtype=np.int32)    # (n_preds,)
            if split_idx is not None:
                spill_full_probas(
                    probs,
                    self._probas_sidecar_dir(),
                    f"split{split_idx}__tp_{token_pair}__{clf_name}__bs{batch_size}",
                )

        return preds, trues, probs

    def _batch_predict_mixed_wise(
        self,
        y_preds_probas,
        y_vals,
        tp_uplet_dict: Dict[int, List[List[str]]],
        thresholds: Optional[Dict[str, Dict[int, float]]] = None,
        mode: Optional[str] = None,
        *,
        split_idx: Optional[int] = None,
    ):
        """Batch-predict with mixed-token_pair batches (one sample per token_pair per uplet)."""
        y_pred: Dict = {}
        y_true: Dict = {}
        probs_save: Dict = {}

        for bs in self.batch_prediction_sizes:
            tp_uplets = tp_uplet_dict.get(bs, [])
            if not tp_uplets:
                continue  # skip batch sizes with no valid uplets (e.g. bs < unique_tp_in_mix)

            y_preds_probas_bs = y_preds_probas[bs]
            y_vals_bs = y_vals[bs]

            all_token_pairs_bs = list({tp for uplet in tp_uplets for tp in uplet})
            tp_keyed = True  # mixed-wise data is always TP-keyed
            unique_classes = self._resolve_unique_classes(y_vals_bs, all_token_pairs_bs, mode, tp_keyed=tp_keyed)

            probas_by_clf_class_tp = self._organize_probas(
                y_preds_probas_bs, y_vals_bs, all_token_pairs_bs, unique_classes, mode, tp_keyed=tp_keyed
            )

            for tp_uplet in tp_uplets:
                assert len(tp_uplet) == bs, f"tp_uplet length {len(tp_uplet)} != batch size {bs}"

                tp_uplet_name = get_tp_uplet_name(tp_uplet)

                if tp_uplet_name not in y_pred:
                    y_pred[tp_uplet_name] = {}
                    y_true[tp_uplet_name] = {}
                    probs_save[tp_uplet_name] = {}

                for clf_name in self.classifier_types:
                    threshold = thresholds[clf_name][bs] if thresholds else None
                    preds, trues, probs = self._batch_predict_mixed_single(
                        clf_name=clf_name,
                        tp_uplet=tp_uplet,
                        batch_size=bs,
                        unique_classes=unique_classes,
                        probas_by_class_tp=probas_by_clf_class_tp[clf_name],
                        threshold=threshold,
                        mode=mode,
                        split_idx=split_idx,
                    )

                    if clf_name not in y_pred[tp_uplet_name]:
                        y_pred[tp_uplet_name][clf_name] = {}
                        y_true[tp_uplet_name][clf_name] = {}
                        probs_save[tp_uplet_name][clf_name] = {}

                    y_pred[tp_uplet_name][clf_name].update({bs: preds})
                    y_true[tp_uplet_name][clf_name].update({bs: trues})
                    probs_save[tp_uplet_name][clf_name].update({bs: probs})

        return y_true, y_pred, probs_save

    def _batch_predict_mixed_single(
        self,
        clf_name,
        tp_uplet,
        batch_size,
        unique_classes,
        probas_by_class_tp,
        threshold,
        mode,
        *,
        split_idx: Optional[int] = None,
    ):
        """Mixed-token_pair batch prediction for one (clf, tp_uplet) combination."""
        save_dca = self.xp_config.get("save_dca_showcase_data", False)
        preds = []
        trues = []
        probs: Dict = {"correct": [], "wrong": []}
        if self.store_prediction_probas:
            probs["full_probas"] = []
            probs["full_y_true"] = []

        for class_label in unique_classes:
            true_label = -1 if mode == "Unknown" else class_label

            per_tp_probas = [probas_by_class_tp[class_label][tp] for tp in tp_uplet]
            n_batches = min(len(p) for p in per_tp_probas)

            dca_true, dca_safe, dca_top1 = [], [], []

            for i in range(n_batches):
                batch = np.stack([per_tp_probas[j][i] for j in range(batch_size)])

                batch_pred = self._aggregate_from_probabilities(batch, threshold=threshold)
                preds.append(batch_pred)
                trues.append(true_label)

                avg_probs = np.mean(batch, axis=0)
                max_avg_prob = np.max(avg_probs)
                bucket = "correct" if batch_pred == class_label else "wrong"
                probs[bucket].append(max_avg_prob)

                if self.store_prediction_probas:
                    probs["full_probas"].append(avg_probs)
                    probs["full_y_true"].append(true_label)

                if save_dca:
                    original_name = full_var_model_name_to_original_model_name(self.model_index[class_label])
                    safe_label = self.model_to_safe_model[original_name]
                    dca_safe.append(avg_probs[safe_label])
                    dca_true.append(avg_probs[class_label])
                    dca_top1.append(self.model_index[int(batch_pred)])

            if save_dca:
                class_key = self.model_index[class_label]
                probs[f"{class_key}"] = dca_true
                probs[f"safe_{class_key}"] = dca_safe
                probs[f"top1_pred_{class_key}"] = dca_top1

        if self.store_prediction_probas:
            probs["full_probas"] = np.array(probs["full_probas"], dtype=np.float32)  # (n_preds, n_classes)
            probs["full_y_true"] = np.array(probs["full_y_true"], dtype=np.int32)    # (n_preds,)
            if split_idx is not None:
                spill_full_probas(
                    probs,
                    self._probas_sidecar_dir(),
                    f"split{split_idx}__uplet_{get_tp_uplet_name(tp_uplet)}__{clf_name}__bs{batch_size}",
                )

        return preds, trues, probs

    def get_raw_summary_results(self) -> Dict:
        """Summary dict: ``summary[batch_type][batch_size][token_pair][clf][metric]``."""
        summary: Dict[Any, Any] = {}
        summarize_metrics(self.batch_size_results_map, summary, self.batch_type)
        summarize_metrics(self.batch_size_confusion_matrices_map, summary, self.batch_type, confusion_matrix=True)
        summarize_metrics(self.probs_save_map, summary, self.batch_type, probs_save=True)
        return summary

    def _plot_max_predict_probas(self, probs_save_map, fig_name: str, bins: int = 10, figsize_per_plot: Tuple[int, int] = (4, 3)) -> None:
        """Histogram of max predicted probabilities (correct vs wrong)."""
        batch_sizes = sorted(probs_save_map.keys())
        n_bs = len(batch_sizes)

        def gather_values(bs, key):
            vals = []
            for tp_dict in probs_save_map[bs].values():
                for clf_dict in tp_dict.values():
                    for arr in clf_dict[key]:
                        vals.append(arr.ravel() if hasattr(arr, 'ravel') else np.array([arr]))
            return np.concatenate(vals) if vals else np.array([])

        def make_figure(kind: str):
            fig, axes = plt.subplots(n_bs, 1, figsize=(figsize_per_plot[0], figsize_per_plot[1] * n_bs), squeeze=False)
            axes = axes.flatten()

            for ax, bs in zip(axes, batch_sizes):
                correct_vals = gather_values(bs, "correct")
                wrong_vals = gather_values(bs, "wrong")

                if kind == "correct":
                    vals = correct_vals
                elif kind == "wrong":
                    vals = wrong_vals
                elif kind == "both":
                    if correct_vals.size > 0:
                        ax.hist(correct_vals, bins=bins, range=(0, 1), alpha=0.6, label="correct", edgecolor="k")
                    if wrong_vals.size > 0:
                        ax.hist(wrong_vals, bins=bins, range=(0, 1), alpha=0.6, label="wrong", edgecolor="k")
                    ax.legend()
                    vals = None
                else:
                    raise ValueError("kind must be 'correct', 'wrong', or 'both'")

                if vals is not None and vals.size > 0:
                    ax.hist(vals, bins=bins, range=(0, 1), alpha=0.7, edgecolor="k")

                ax.set_xlim(0, 1)
                ax.set_xlabel("Max predicted probability")
                ax.set_ylabel("Count")
                ax.set_title(f"{kind.capitalize()} predictions (bs={bs})")

            plt.tight_layout()
            save_fig_and_show(save_path=self.fig_save_path, show=self.show, fig_name=f"{fig_name}_{kind}.pdf")

        make_figure("correct")
        make_figure("wrong")
        make_figure("both")

    def _batch_classification_mix_tp_at_train(self):
        """Run mix_tp_at_train classification: train on concatenated features from multiple token pairs."""
        self._initialize_results_and_conf_structure()

        n_classes = len(self.model_index)
        splitter = SPLITTER_MAP[self.splitter_type](
            n_splits=self.n_splits,
            random_state=self.random_seed,
            test_size=self.test_size * n_classes,
        )

        for batch_size in self.batch_prediction_sizes:
            if not self.tp_uplet_dict.get(batch_size):
                continue  # skip batch sizes with no valid uplets
            self._batch_concat_train_predict_across_tp(splitter, batch_size, self.tp_uplet_dict)

        batch_size_results_map = self.results
        batch_size_confusion_matrix_map = self.confusion_matrices
        probs_save_map = {}

        return batch_size_results_map, batch_size_confusion_matrix_map, probs_save_map

    def _batch_concat_train_predict_across_tp(self, splitter, batch_size, tp_uplet_dict: Dict[int, List[List[str]]]) -> None:
        """Concat-based training across token pairs: mix features at training time."""
        token_pairs = list(self.X_s.keys())

        X_splits, y_splits = self._prepare_and_split_all(
            token_pairs, splitter, batch_type="mix_tp_at_train", batch_size=batch_size
        )

        for split_idx in range(self.n_splits):
            X_tr_tp, y_tr_tp = X_splits[split_idx]["train"], y_splits[split_idx]["train"]
            X_val_tp, y_val_tp = X_splits[split_idx]["val"], y_splits[split_idx]["val"]
            logger.debug(f"{X_val_tp[token_pairs[0]].shape = }, {self.test_size = }, {X_tr_tp[token_pairs[0]].shape = } ")

            for tp_uplet in tp_uplet_dict[batch_size]:
                tp_uplet_name = get_tp_uplet_name(tp_uplet)
                logger.info(f"{batch_size = }, {tp_uplet_name =}")

                effective_bs = self._resolve_unique_tp(batch_size)
                X_mixed_tr, y_mixed_tr = mix_by_class_multi_tp(
                    X_tr_tp, y_tr_tp, tp_uplet, max_combinations=300, extra_samples=self.train_size % effective_bs
                )

                X_mixed_val, y_mixed_val = sequential_concat_multi_tp(X_val_tp, y_val_tp, tp_uplet)

                logger.debug(f"{X_mixed_tr.shape =}, {X_mixed_val.shape =}")

                # Normalize: fit on mixed train, transform both
                X_mixed_tr, X_mixed_val, _ = fit_transform_normalize(
                    X_mixed_tr, X_mixed_val,
                    self.features_index, self.normalization_methods, self.default_normalization,
                )

                y_pred_probas = self._train_and_predict_proba(
                    X_mixed_tr, y_mixed_tr, X_mixed_val, tp_name=tp_uplet_name
                )

                # Wrap with batch_size key to match _save_prediction_results expected structure
                y_pred = {clf_name: {batch_size: np.argmax(y_pred_probas[clf_name], axis=1)} for clf_name in self.classifier_types}
                y_true = {clf_name: {batch_size: y_mixed_val} for clf_name in self.classifier_types}

                self._save_prediction_results(y_true, y_pred, tp=tp_uplet_name)

    def _prepare_and_split_all(
        self,
        token_pairs: List[str],
        splitter,
        batch_type: str,
        batch_size: int,
        no_trunc_test: bool = True,
    ):
        """Prepare and split each token_pair, returning per-split train/val dicts."""
        X_splits = [{"train": {}, "val": {}} for _ in range(self.n_splits)]
        y_splits = [{"train": {}, "val": {}} for _ in range(self.n_splits)]

        effective_bs = self._resolve_unique_tp(batch_size)
        train_truncation = self.train_size // effective_bs
        if self.train_size % effective_bs > 0:
            train_truncation += 1

        test_truncation = self.test_size // batch_size
        if self.test_size % batch_size > 0:
            test_truncation += 1

        if no_trunc_test:
            test_truncation = self.test_size

        logger.debug(f"{train_truncation = }, {test_truncation = }")
        for tp in token_pairs:
            X_p, y_p = self._prepare_data(self.X_s[tp], batch_type=batch_type, batch_size=batch_size, tp_name=tp)
            nb_of_class = len(np.unique(y_p))
            for idx, (tr_idx, te_idx) in enumerate(splitter.split(X_p, y_p)):
                X_tr, y_tr = X_p[tr_idx], y_p[tr_idx]
                X_val, y_val = X_p[te_idx], y_p[te_idx]

                # Balance training fold only (after split to avoid leakage)
                _blank = self._create_single_token_pair_classif(tp)
                X_tr, y_tr = _blank.balance(X_tr, y_tr)

                if len(X_tr) / nb_of_class < train_truncation:
                    raise ValueError(
                        f"Not enough samples per class in training set of token_pair {tp} for batch_size {batch_size}. "
                        f"Got {len(X_tr)/nb_of_class} samples per class, need at least {train_truncation}. "
                        f"Try reducing train_size or batch_size."
                    )
                if len(X_val) < test_truncation:
                    raise ValueError(
                        f"Not enough samples in test set of token_pair {tp} for batch_size {batch_size}. "
                        f"Got {len(X_val)} samples, need at least {test_truncation}. "
                        f"Try reducing test_size or batch_size."
                    )

                X_tr, y_tr = self._reducing_token_pair(X_tr, y_tr, truncation=train_truncation)
                X_val, y_val = self._reducing_token_pair(X_val, y_val, truncation=test_truncation)

                X_splits[idx]["train"][tp] = X_tr
                X_splits[idx]["val"][tp] = X_val
                y_splits[idx]["train"][tp] = y_tr
                y_splits[idx]["val"][tp] = y_val

        return X_splits, y_splits
