import warnings
import logging
logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple, Union

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler


def preprocess_X(X, strategy="mean", n_neighbors=5, p=0.8, K=2, verbose: bool = False):
    """
    Handles NaNs in a 3D array X (samples, models, features) with custom logic.

    Logic:
    - If a model-sample has all NaN features, leave it unchanged.
    - If a model-sample has > p fraction of NaN features, set all its features to NaN.
    - Otherwise, impute missing values using other samples from the same model.
      If a feature is entirely missing across valid samples, fill it with 0.0.
    - If after imputation fewer than K samples remain (i.e., not all-NaN), ditch the entire model.
    - Enforce "all-or-nothing" at the end and assert no partial NaN rows remain.

    Prints detailed logs for each model.
    """
    X = X.copy()
    # Treat ±inf as missing so the standard NaN-imputation path handles them.
    # Without this, sklearn classifiers (LDA/LR/SVM/KNN/MLP/RF/GB) reject the
    # data at predict time; only XGBoost is inf-tolerant.
    n_inf = int(np.isinf(X).sum())
    if n_inf > 0:
        if verbose:
            logger.debug("preprocess_X: replacing %d non-finite (inf) entries with NaN", n_inf)
        X = np.where(np.isfinite(X), X, np.nan)
    n_samples, n_models, n_features = X.shape

    # Select the sklearn imputer
    if strategy in ("mean", "median", "most_frequent"):
        Imputer = lambda: SimpleImputer(strategy=strategy)
    elif strategy == "knn":
        Imputer = lambda: KNNImputer(n_neighbors=n_neighbors)
    elif strategy == "iterative":
        Imputer = lambda: IterativeImputer(random_state=0)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    for m in range(n_models):
        M = X[:, m, :]

        # 1) Categorize samples
        is_all_nan = np.isnan(M).all(axis=1)
        nan_counts = np.isnan(M).sum(axis=1)
        over_thresh = nan_counts > (p * n_features)
        to_impute = (~is_all_nan) & (~over_thresh) & (nan_counts > 0)

        # 2) Nullify over-threshold samples
        if over_thresh.any() and verbose:
            logger.debug(f"Model {m}: nullifying {np.sum(over_thresh)} samples (> {p*100}% NaNs)")
        M[over_thresh, :] = np.nan

        # 3) Impute remaining samples
        if to_impute.any():
            valid = (~is_all_nan) & (~over_thresh)
            Y = M[valid, :]

            # Track entirely-missing features
            feature_all_nan = np.isnan(Y).all(axis=0)
            if feature_all_nan.any():
                missing_feats = np.where(feature_all_nan)[0]
                if verbose:
                    logger.debug(
                        f"Model {m}: features {missing_feats.tolist()} unobserved across valid samples; will fill with 0.0"
                    )

            imputer = Imputer()
            # Suppress the sklearn warning for all-NaN features — we handle them below
            # via re-expansion and zero-filling (lines after this block).
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="Skipping features without any observed values",
                )
                Y_imputed = imputer.fit_transform(Y)

            # Re-expand dropped columns if any
            if Y_imputed.shape[1] != n_features:
                kept = ~feature_all_nan
                full = np.full((Y_imputed.shape[0], n_features), np.nan, dtype=Y_imputed.dtype)
                full[:, kept] = Y_imputed
                Y_imputed = full

            # Zero-fill any columns still all NaN
            col_all_nan = np.isnan(Y_imputed).all(axis=0)
            if col_all_nan.any():
                zero_feats = np.where(col_all_nan)[0]
                if verbose:
                    logger.debug(f"Model {m}: zero-filling features {zero_feats.tolist()} after imputation")
                Y_imputed[:, col_all_nan] = 0.0

            M[valid, :] = Y_imputed


        # 5) Ditch entire model if too few valid samples remain
        survivors = np.sum(~np.isnan(M).all(axis=1))
        if survivors < K:
            if verbose:
                logger.debug(f"Model {m}: only {survivors} samples remain (< K={K}); ditching entire model")
            M[:, :] = np.nan

        # Write back to X
        X[:, m, :] = M

    # Final validation
    violations = []
    for i in range(n_samples):
        for m in range(n_models):
            row = X[i, m, :]
            n_nan = np.isnan(row).sum()
            if 0 < n_nan < n_features:
                violations.append((i, m, n_nan))
    if violations:
        lines = "\n".join(f" sample={i}, model={m}: {n_nan}/{n_features}" for i, m, n_nan in violations[:10])
        extra = len(violations) - len(violations[:10])
        msg = (
            f"Assumption violation: found {len(violations)} partial-NaN rows.\n"
            "Each model-sample must be either entirely NaN or entirely filled.\n"
            "Examples:\n" + lines + (f"\n...and {extra} more" if extra > 0 else "")
        )
        raise ValueError(msg)

    return X


class FeatureNormalizer(TransformerMixin, BaseEstimator):
    """
    Custom transformer to apply different normalization methods to different features.
    This allows for feature-specific normalization based on data characteristics.
    """

    def __init__(
        self,
        features_index: Dict[str, int],
        normalization_methods: Optional[Dict[str, str]] = None,
        default_normalization: str = "auto",
    ):
        """
        Initialize the feature normalizer.

        Args:
            features_index: Dictionary mapping feature names to feature indices of data_matrix.
            normalization_methods: Dictionary mapping feature names to normalization methods.
                                   Available methods: 'standard', 'minmax', 'robust', 'power', 'quantile', 'log', 'none', 'auto'.
                                   If 'auto', will auto-detect appropriate method based on data characteristics.
            default_normalization: Default normalization method for features not in normalization_methods.
        """
        self.features_index = features_index
        self.normalization_methods = normalization_methods if normalization_methods is not None else {}
        self.default_normalization = default_normalization
        self.transformers: Dict[int, Optional[object]] = {}
        self.log_offsets: Dict[int, float] = {}

    def _resolve_normalization_methods(self, X):
        """
        Args:
            X: Input features of shape (n_samples * n_models, n_features)
        """
        # Map feature indices to methods (default 'auto')
        self.normalization_methods_index = {
            feature_idx: self.normalization_methods.get(name, self.default_normalization)
            for name, feature_idx in self.features_index.items()
        }

        # Detect appropriate method for 'auto' features
        for feature_idx, method in self.normalization_methods_index.items():
            if method == "auto":
                self.normalization_methods_index[feature_idx] = self._detect_normalization_method(X[:, feature_idx])

    def _detect_normalization_method(self, feature_data: np.ndarray) -> str:
        # ensure numeric and drop NaN/inf
        feature_data = np.asarray(feature_data, dtype=float)
        feature_data = feature_data[np.isfinite(feature_data)]

        if feature_data.size == 0:
            return "none"  # nothing to do

        # if constant or near-constant (e.g. from set_constant_seq_length) -> no transform
        std = np.nanstd(feature_data)
        mean_abs = np.abs(np.nanmean(feature_data))
        if std == 0 or (mean_abs > 0 and std / mean_abs < 1e-8):
            return "none"

        skewness = stats.skew(feature_data)
        q1, q3 = np.percentile(feature_data, [25, 75])
        iqr = q3 - q1
        outlier_count = np.sum((feature_data < q1 - 1.5 * iqr) | (feature_data > q3 + 1.5 * iqr))
        outlier_ratio = outlier_count / feature_data.size
        all_non_negative = np.all(feature_data >= 0)
        data_range = np.max(feature_data) - np.min(feature_data)

        if np.isfinite(skewness) and abs(skewness) > 1.0 and all_non_negative:
            return "log"
        elif np.isfinite(skewness) and abs(skewness) > 1.0:
            return "power"
        elif outlier_ratio > 0.05:
            return "robust"
        elif data_range > 10:
            return "standard"
        else:
            return "minmax"

    def _get_transformer(self, method: str):
        """
        Get the appropriate transformer based on the method name.

        Args:
            method: Normalization method name

        Returns:
            Transformer object
        """
        if method == "standard":
            return StandardScaler()
        elif method == "minmax":
            return MinMaxScaler()
        elif method == "robust":
            return RobustScaler()
        elif method == "power":
            return PowerTransformer(method="yeo-johnson")
        elif method == "quantile":
            return QuantileTransformer(output_distribution="normal")
        elif method == "log":
            return None  # Special case, handled separately
        elif method == "none":
            return None
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "FeatureNormalizer":
        """Fit normalizers per-feature based on data characteristics."""
        self._resolve_normalization_methods(X)
        self.log_offsets = {}

        for feature_index, method in self.normalization_methods_index.items():
            if method == "log":
                # Compute and store the offset needed for log transform
                col = np.asarray(X[:, feature_index], dtype=float)
                finite_vals = col[np.isfinite(col)]
                min_val = np.min(finite_vals) if finite_vals.size > 0 else 0.0
                self.log_offsets[feature_index] = (-min_val + 1.0) if min_val <= 0 else 0.0
                continue
            if method == "none":
                continue

            transformer = self._get_transformer(method)
            # prepare column: numeric, finite
            col = np.asarray(X[:, feature_index], dtype=float)
            finite_mask = np.isfinite(col)
            col_finite = col[finite_mask].reshape(-1, 1)

            # if no usable values or constant -> mark 'none'
            if col_finite.size == 0 or np.nanstd(col_finite) == 0:
                # fallback: no transform
                self.normalization_methods_index[feature_index] = "none"
                self.transformers[feature_index] = None
                continue

            # try to fit; if it fails, fallback to standard scaler
            try:
                self.transformers[feature_index] = transformer.fit(col_finite)  # type:ignore
            except Exception as e:
                # fallback strategy: try a safer transformer
                try:
                    fallback = StandardScaler()
                    self.transformers[feature_index] = fallback.fit(col_finite)
                    self.normalization_methods_index[feature_index] = "standard"
                except Exception:
                    # last resort: disable transformation
                    self.transformers[feature_index] = None
                    self.normalization_methods_index[feature_index] = "none"

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using previously fitted normalizers."""
        if not hasattr(self, "normalization_methods_index"):
            raise RuntimeError("FeatureNormalizer has not been fitted. Call fit() before transform().")

        X_transformed = X.copy()

        for feature_idx, method in self.normalization_methods_index.items():
            if method == "log":
                offset = self.log_offsets.get(feature_idx, 0.0)
                # Sanitize the log argument: any value that would make log(.) non-finite
                # (e.g. val + offset <= 0 because val_test < val_train_min) gets clipped
                # to a tiny positive number, so the result is a very negative-but-finite
                # log value rather than -inf or NaN.
                arg = X_transformed[:, feature_idx] + offset
                arg = np.where(arg > 0, arg, 1e-12)
                X_transformed[:, feature_idx] = np.log(arg)
            elif method != "none":
                # Apply fitted transformer
                transformed_feature = self.transformers[feature_idx].transform(
                    X_transformed[:, feature_idx].reshape(-1, 1)
                )
                X_transformed[:, feature_idx] = transformed_feature.ravel()

        # Final defensive sanitization: any non-finite value that leaked through (e.g.
        # PowerTransformer overflow, QuantileTransformer edge cases) is replaced with 0.
        # 0 is roughly the post-normalization median for standard/minmax/robust outputs,
        # so it's a safe neutral fallback that lets downstream sklearn classifiers run.
        non_finite_mask = ~np.isfinite(X_transformed)
        n_bad = int(non_finite_mask.sum())
        if n_bad > 0:
            logger.warning(
                "FeatureNormalizer.transform: %d non-finite values replaced with 0 "
                "(check upstream data quality if this number is large)", n_bad,
            )
            X_transformed[non_finite_mask] = 0.0

        return X_transformed

    def get_normalization_methods_index(self) -> Dict[int, str]:
        """Return the resolved mapping of feature index to normalization method."""
        return self.normalization_methods_index


def fit_transform_normalize(
    X_train: np.ndarray,
    X_test: np.ndarray,
    features_index: Dict[str, int],
    normalization_methods: Optional[Dict[str, str]],
    default_normalization: str,
) -> Tuple[np.ndarray, np.ndarray, FeatureNormalizer]:
    """Create a FeatureNormalizer, fit on training data, transform both."""
    normalizer = FeatureNormalizer(features_index, normalization_methods, default_normalization)
    normalizer.fit(X_train)
    return normalizer.transform(X_train), normalizer.transform(X_test), normalizer
