"""Token pair mixing utilities for multi-token-pair classification.

Provides functions to combine feature samples across multiple token pairs
per class, using combinatorial or random sampling strategies.
"""

import itertools
import logging
import operator
import random
from functools import reduce
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def safe_prod(iterable) -> int:
    """Product using Python big ints (no overflow)."""
    return reduce(operator.mul, iterable, 1)


def generate_product_combinations(
    counts: List[int],
    max_combinations: int,
    max_explicit: int = 10**7,
) -> List[Tuple[int, ...]]:
    """Generate up to *max_combinations* unique index combinations.

    Parameters
    ----------
    counts : list of int
        Number of samples available in each dataset slice.
    max_combinations : int
        Maximum number of unique combinations to return.
    max_explicit : int, optional
        Upper bound on total combinations before raising MemoryError.

    Returns
    -------
    list of tuple
        Unique combinations (tuples of sample indices).
    """
    total_combos = safe_prod(counts)

    if max_combinations >= total_combos:
        if total_combos > max_explicit:
            raise MemoryError(f"Too many combinations ({total_combos:,}) to generate explicitly")
        combos = itertools.product(*[range(c) for c in counts])
        return list(combos)

    # heuristic: if we're sampling more than half of all combos
    if max_combinations > total_combos // 2 and total_combos <= max_explicit:
        combos = itertools.product(*[range(c) for c in counts])
        sample = random.sample(list(combos), max_combinations)
        return sample

    # random sampling with uniqueness
    combos: set = set()
    while len(combos) < max_combinations:
        combo = tuple(random.randint(0, c - 1) for c in counts)
        combos.add(combo)

    return list(combos)


def mix_by_class_multi_tp(
    X_tp: Dict[str, np.ndarray],
    y_tp: Dict[str, np.ndarray],
    token_pairs: List[str],
    extra_samples: int,
    combination: bool = True,
    max_combinations: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """Combine feature samples across token pairs per class.

    For each class, takes samples from each token pair and creates all
    possible horizontal concatenations (or a random subset thereof).

    Parameters
    ----------
    X_tp : dict
        ``{token_pair_name: feature_array}`` with shape ``(n_samples, n_features)``.
    y_tp : dict
        ``{token_pair_name: label_array}`` with shape ``(n_samples,)``.
    token_pairs : list of str
        Token pair names to include in the mix.
    extra_samples : int
        Number of extra single-token-pair samples to include (for padding).
    combination : bool
        If True, use combinatorial mixing; otherwise raise NotImplementedError.
    max_combinations : int
        Cap on the number of generated combinations per class.

    Returns
    -------
    X_mixed : np.ndarray
        Shape ``(n_total, n_features * len(token_pairs))``.
    y_mixed : np.ndarray
        Shape ``(n_total,)``.
    """
    classes = np.unique(y_tp[token_pairs[0]])

    mixed_X_list, mixed_y_list = [], []

    # Adding left self.train_size % batch_size samples from all possible tp:
    if extra_samples > 0:
        selected_tp_idx = random.sample(range(len(token_pairs)), extra_samples)
        selected_tp = [token_pairs[tp_idx] for tp_idx in selected_tp_idx]
        assert len(selected_tp) == extra_samples
    else:
        selected_tp = []

    for cl in classes:
        slices = [
            (
                X_tp[tp][y_tp[tp] == cl][:-1]
                if (extra_samples > 0) and (tp not in selected_tp)
                else X_tp[tp][y_tp[tp] == cl]
            )
            for tp in token_pairs
        ]

        if len(token_pairs) == 1:
            X_cat = slices[0]
            y_cat = np.full(slices[0].shape[0], cl)

        elif combination:
            counts = [s.shape[0] for s in slices]
            all_combinations: list = generate_product_combinations(counts, max_combinations)
            X_combinations = []
            for combo in all_combinations:
                sample_features = []
                for ds_idx, sample_idx in enumerate(combo):
                    sample_features.append(slices[ds_idx][sample_idx])
                combined_sample = np.concatenate(sample_features, axis=0)
                X_combinations.append(combined_sample)

            X_cat = np.array(X_combinations)
            y_cat = np.full(len(all_combinations), cl)
        else:
            raise NotImplementedError("Currently only combination=True is implemented.")

        mixed_X_list.append(X_cat)
        mixed_y_list.append(y_cat)

    X_mixed = np.concatenate(mixed_X_list, axis=0)
    y_mixed = np.concatenate(mixed_y_list, axis=0)

    return X_mixed, y_mixed


def sequential_concat_multi_tp(
    X_tp: Dict[str, np.ndarray],
    y_tp: Dict[str, np.ndarray],
    token_pairs: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate token pair features sequentially (no augmentation).

    For each class, zips samples across token pairs positionally: sample i from TP1
    is concatenated with sample i from TP2, etc. Produces exactly
    ``min(n_samples_per_class across TPs)`` samples per class — no expansion.

    Parameters
    ----------
    X_tp : dict
        ``{token_pair_name: feature_array}`` with shape ``(n_samples, n_features)``.
    y_tp : dict
        ``{token_pair_name: label_array}`` with shape ``(n_samples,)``.
    token_pairs : list of str
        Token pair names to include in the concatenation.

    Returns
    -------
    X_concat : np.ndarray
        Shape ``(n_total, n_features * len(token_pairs))``.
    y_concat : np.ndarray
        Shape ``(n_total,)``.
    """
    classes = np.unique(y_tp[token_pairs[0]])
    X_parts, y_parts = [], []

    for cl in classes:
        slices = [X_tp[tp][y_tp[tp] == cl] for tp in token_pairs]
        n = min(s.shape[0] for s in slices)
        X_cat = np.concatenate([s[:n] for s in slices], axis=1)
        y_parts.append(np.full(n, cl))
        X_parts.append(X_cat)

    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def balance_class_Xy_list(
    X_list: List[np.ndarray],
    y_list: List[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Truncate all class slices to the size of the smallest class."""
    min_Xsample = min(len(slices) for slices in X_list)
    min_ysample = min(len(slices) for slices in y_list)
    assert min_Xsample == min_ysample, f"{min_Xsample = }, {min_ysample = }"
    min_size = min_Xsample
    logger.debug("min_size=%s", min_size)
    balanced_X_list = [slices[:min_size] for slices in X_list]
    balanced_y_list = [slices[:min_size] for slices in y_list]
    return balanced_X_list, balanced_y_list
