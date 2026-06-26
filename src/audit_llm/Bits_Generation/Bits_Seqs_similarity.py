# N: nb_of_samples to compare
# L: seq_lenght

## The following three metrics are O(N**2 * L**2) (too expensive, order of element in sequence matters)

## The following metrics are O(N**2 * L)


from collections import Counter
from itertools import product
from math import sqrt
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.notebook import tqdm


# ----------------------------
# Helper functions
# ----------------------------


def extract_kgrams(seq, k):
    """Extract overlapping k-grams from a sequence."""
    return ["_".join(map(str, seq[i : i + k])) for i in range(len(seq) - k + 1)]


# ----------------------------
# Similarity metrics
# ----------------------------


def jaccard_kgram_sim(a, b, k=1):
    """Set-based Jaccard similarity on k-grams."""
    grams_a = {tuple(a[i : i + k]) for i in range(len(a) - k + 1)}
    grams_b = {tuple(b[i : i + k]) for i in range(len(b) - k + 1)}
    if not grams_a and not grams_b:
        return 1.0
    return len(grams_a & grams_b) / len(grams_a | grams_b)


def multiset_jaccard(a, b, k=3):
    """Multiset Jaccard index, accounting for frequency of k-grams."""
    a_k = Counter(extract_kgrams(a, k))
    b_k = Counter(extract_kgrams(b, k))
    return sum((a_k & b_k).values()) / sum((a_k | b_k).values()) if (a_k | b_k) else 1.0


def kgram_cosine_similarity(a, b, k=3):
    """Cosine similarity between k-gram frequency vectors."""
    a_k = Counter(extract_kgrams(a, k))
    b_k = Counter(extract_kgrams(b, k))
    all_keys = sorted(set(a_k) | set(b_k))
    vec_a = np.array([a_k.get(key, 0) for key in all_keys])
    vec_b = np.array([b_k.get(key, 0) for key in all_keys])
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b)) if norm_a and norm_b else 0.0


def tfidf_kgram_similarity(a, b, k=3):
    """Cosine similarity on TF-IDF vectors of k-grams."""
    corpus = [" ".join(extract_kgrams(a, k)), " ".join(extract_kgrams(b, k))]
    vec = TfidfVectorizer()
    tfidf = vec.fit_transform(corpus)
    return float(cosine_similarity(tfidf[0], tfidf[1])[0, 0])  # type: ignore


def compute_k_scores(a, b, k_values, sim_func):
    return [sim_func(a, b, k) for k in k_values]


# ----------------------------
# Core routines
# ----------------------------

SIM_FUNC_MAP = {
    "Jaccard": jaccard_kgram_sim,
    "Multiset_Jaccard": multiset_jaccard,
    "Cosine": kgram_cosine_similarity,
    "TF_IDF_Cosine": tfidf_kgram_similarity,
}


def compute_similarities(
    sequences,
    metrics: Optional[List[str]] = None,
    k_list: Optional[List[int]] = None,
    strategies: Optional[List[str]] = None,
    disable_tqdm: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    # Set default values for parameters
    if k_list is None:
        k_list = list(range(1, 6))
    if metrics is None:
        metrics = ["Jaccard", "Multiset_Jaccard", "Cosine", "TF_IDF_Cosine"]
    if strategies is None:
        strategies = ["uniform", "linear", "exponential", "inverse"]

    # Create metrics map
    metrics_map = {
        f"{metric_name} (k={k})": lambda a, b, f=SIM_FUNC_MAP[metric_name], _k=k: f(a, b, _k)
        for k in k_list
        for metric_name in metrics
    }

    n = len(sequences)
    scores_matrices = {}

    # Compute similarity matrices
    metrics_bar = tqdm(
        metrics_map.items(), desc=f"Computing similarities, with {len(sequences) = }", position=3, leave=True
    )
    for name, fn in metrics_bar:
        metrics_bar.set_description(f"Computing similarities, with {len(sequences) = }, metric ={name}")
        mat = np.zeros((n, n))
        for i, j in product(range(n), repeat=2):
            mat[i, j] = fn(sequences[i], sequences[j])
        scores_matrices[name] = mat

    # Aggregate scores for the different k values
    aggregated_scores = {}  # Dict[str, np.ndarray(n,n)]

    for strategy in strategies:
        # Compute weights once per strategy
        weights = compute_weights(k_list, strategy)

        for metric_name in metrics:
            # Collect all matrices for this metric across different k values
            metric_matrices = [scores_matrices[f"{metric_name} (k={k})"] for k in k_list]

            # Generate the aggregated score matrix
            agg_name = f"{metric_name}_{strategy}_agg"
            aggregated_scores[agg_name] = aggregate_k_similarities(metric_matrices, weights)

    return scores_matrices, aggregated_scores


def aggregate_k_similarities(scores_matrices, weights):
    """
    Aggregate similarity matrices using weighted combination.

    Args:
        scores_matrices: List of score matrices to aggregate
        weights: List of weights for each matrix

    Returns:
        Aggregated similarity matrix
    """
    # Get dimensions from the first matrix
    n = scores_matrices[0].shape[0]

    # Initialize the aggregation matrix
    agg_matrix = np.zeros((n, n))

    # Vectorized implementation
    for k in range(len(scores_matrices)):
        agg_matrix += weights[k] * scores_matrices[k]

    return agg_matrix


def compute_weights(k_values, strategy="uniform"):
    """
    Compute aggregation weights for a list of k values based on the specified strategy.

    Strategies:
    - uniform: equal weight to each k
    - linear: weights proportional to 1,2,...,m
    - exponential: weights halved each step (most weight on largest k)
    - inverse: weights proportional to 1/k

    Args:
        k_values: List of k values
        strategy: Weight computation strategy

    Returns:
        Normalized weights array
    """
    m = len(k_values)

    if strategy == "uniform":
        return np.full(m, 1.0 / m)
    elif strategy == "linear":
        raw = np.arange(1, m + 1, dtype=float)
        return raw / raw.sum()
    elif strategy == "exponential":
        raw = np.array([0.5 ** (m - i - 1) for i in range(m)], dtype=float)
        return raw / raw.sum()
    elif strategy == "inverse":
        raw = np.array([1.0 / k for k in k_values], dtype=float)
        return raw / raw.sum()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

