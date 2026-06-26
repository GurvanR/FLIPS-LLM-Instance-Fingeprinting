"""Data loading and feature computation functions (extracted from Analysis_Classes.py).

This module handles feature computation, loading, and model preparation for experiments.
"""

import json
import logging
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def _get_rss_mb() -> str:
    """Return current process RSS in MB (Linux only, no dependencies)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return f"{kb / 1024:.0f}MB"
    except (FileNotFoundError, ValueError):
        pass
    return "N/A"

from audit_llm.Bits_Generation.bits_tools import (
    compute_intra_samples_bit_feature_matrix,
    fill_inter_samples_features_map,
    make_random_bit_sequences,
    rng_tests_dict_to_nb_of_tests,
)
from audit_llm.models_management.model_names import PRNG_MODELS
from audit_llm.Bits_Generation.parsing_bits_tools import (
    bits_token_pair_to_scrapper,
    token_pair_name_to_items,
)
from audit_llm.file_io import (
    compute_file_hash,
    load_json,
    load_manifest,
    open_pickle_file,
    read_parquet_dir_or_monolithic,
    sanitize_model_name,
    write_json,
    write_manifest,
)
from audit_llm.data_transforms import preprocess_token_ids_col_of_answers
from audit_llm.file_io import erase_file
from audit_llm.path_utils import get_repository_level_path, make_dir
from audit_llm.polars_tools import join_dataframes
from audit_llm.Tokens_analysis.vocab_analysis import analyze_and_save_vocab_substrings
from audit_llm.xp_init_fun import integrate_nist_test_parameters

# Constants from original Analysis_Classes.py
COMPUTE_METRICS_MAP = {"Bits_Datasets": {}}

TRUE_RNG_GENERATOR_MAP = {
    "Bits": make_random_bit_sequences,
}

INTRA_SAMPLES_FEATURES_COMPUTER_MAP = {
    "Bits": compute_intra_samples_bit_feature_matrix,
}

INTER_SAMPLES_FEATURES_COMPUTER_MAP = {
    "Erdos_Renyi": "",
    "Bits": fill_inter_samples_features_map,
}

DATASET_SORTER_MAP = {
    "Bits": lambda x: x,  # identity
}

SCRAPPER_MAP = {
    "Bits_Datasets": bits_token_pair_to_scrapper,
}

SCRAPPING_RULE = "DEFAULT"


# Token analysis configuration (moved from xp_tools)
COMPLEMENTARY_SUBSTRINGS = []  # Define based on use case


def _analysis_mode(
    Analysis_save_path: str,
    scrapping_rule: str,
    Dataset_path: str,
    seed: Optional[int] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, Path]:
    """Load and prepare answers, token IDs, and main token_pair DataFrames.
    
    Sets the Answers_df which is the df of all collected answers, filtered by models.
    
    Args:
        Analysis_save_path: Path to analysis save directory
        scrapping_rule: Scrapping rule set in run_config
        Dataset_path: Path to the main token_pair CSV
        seed: Optional random seed for reproducibility
    
    Returns:
        Tuple of (Answers_df, TokenIDs_df, MainDataset_df, Answers_df_path)
    """
    base_path = Path(Analysis_save_path) / scrapping_rule

    # Per-model directory > monolithic Parquet > monolithic CSV
    answers_dir = base_path / "answers"
    token_ids_dir = base_path / "token_ids"

    if answers_dir.exists() and any(answers_dir.glob("*.parquet")):
        # Per-model partitioned Parquet
        # Read per-file to normalize schema: old files use "Dataset", new ones
        # use "Token_pair".  A single scan_parquet() on the directory fails with
        # SchemaError when both column names coexist.
        _dfs = []
        for _f in sorted(answers_dir.glob("*.parquet")):
            _df = pl.read_parquet(_f)
            if "Dataset" in _df.columns and "Token_pair" not in _df.columns:
                _df = _df.rename({"Dataset": "Token_pair"})
            _dfs.append(_df)
        Answers_df = pl.concat(_dfs, how="diagonal")
        Answers_df_path = answers_dir
    elif (base_path / "Answers.parquet").exists():
        Answers_df = pl.scan_parquet(base_path / "Answers.parquet").collect()
        Answers_df_path = base_path / "Answers.parquet"
    elif (base_path / "Answers.csv").exists():
        Answers_df = pl.read_csv(
            base_path / "Answers.csv",
            schema_overrides={"Answer": pl.Utf8},
            quote_char='"',
            infer_schema_length=10_000,
        )
        Answers_df_path = base_path / "Answers.csv"
    else:
        raise FileNotFoundError(f"No answers data found in {base_path}")

    logger.info("Answers dataframe loaded.")

    # Backward compat: legacy files have "Dataset" instead of "Token_pair"
    if "Dataset" in Answers_df.columns and "Token_pair" not in Answers_df.columns:
        Answers_df = Answers_df.rename({"Dataset": "Token_pair"})

    # Load Token IDs — per-model directory > monolithic Parquet > monolithic CSV
    open_token_df = False  # Too big for the moment
    if open_token_df:
        if token_ids_dir.exists() and any(token_ids_dir.glob("*.parquet")):
            TokenIDs_df = pl.scan_parquet(token_ids_dir).collect()
        elif (base_path / "TokenIDsAnswers.parquet").exists():
            TokenIDs_df = pl.scan_parquet(base_path / "TokenIDsAnswers.parquet").collect()
        elif (base_path / "TokenIDsAnswers.csv").exists():
            TokenIDs_df = pl.read_csv(base_path / "TokenIDsAnswers.csv")
        else:
            raise FileNotFoundError(f"No TokenIDs data found in {base_path}")

        TokenIDs_df = preprocess_token_ids_col_of_answers(TokenIDs_df)
    else:
        # Create dataframe with empty Token_IDs column
        TokenIDs_df = Answers_df.with_columns(
            pl.lit(None).cast(pl.List(pl.Int32)).alias("Token_IDs")
        ).drop("Answer")

    logger.info("Token_IDs col processed.")

    # Load MainDataset_df
    MainDataset_df = pl.read_csv(Dataset_path)
    logger.info("MainDataset loaded.")

    return Answers_df, TokenIDs_df, MainDataset_df, Answers_df_path


def _set_environment(Analysis_save_path: str, Experiments_path: str):
    """Set up directory structure for analysis.
    
    Args:
        Analysis_save_path: Path to analysis save directory  
        Experiments_path: Path to experiments directory
    """
    make_dir(Analysis_save_path)
    make_dir(os.path.join(Analysis_save_path, "Figures"))

    erase_file(os.path.join(Analysis_save_path, "Answers.csv"))
    erase_file(os.path.join(Analysis_save_path, "global_table.csv"))

    # Make Experiments environment
    make_dir(Experiments_path)


def _prepare_models_with_PRNGs(Answers_df: pl.DataFrame) -> Tuple[List[str], List[str], Dict[str, int]]:
    """Prepare models list including PRNG models for feature computation.
    
    Args:
        Answers_df: Answers DataFrame
    
    Returns:
        Tuple of (token_pairs, models, model_index)
    """
    # Get unique sorted token_pairs and models
    token_pairs = sorted(Answers_df["Token_pair"].unique())

    # Models list preparation
    models = sorted(Answers_df["Model"].unique().to_list())
    models.extend(PRNG_MODELS)  # Add different PRNGs

    # Create and save index mapping for models
    model_index = {model: idx for idx, model in enumerate(models)}

    return token_pairs, models, model_index


def _inter_feats_for_model_to_npz_dict(
    inter_feats: Dict[tuple, Any], model_k: int
) -> Dict[str, np.ndarray]:
    """Extract model *model_k*'s inter-sample features and flatten for npz storage."""
    npz_data: Dict[str, np.ndarray] = {}
    for (k, feat_name), value in inter_feats.items():
        if k != model_k:
            continue
        if isinstance(value, np.ndarray):
            npz_data[feat_name] = value
        elif isinstance(value, dict):
            for sub_key, sub_val in value.items():
                flat_key = f"{feat_name}__{sub_key}"
                npz_data[flat_key] = np.asarray(sub_val)
        else:
            npz_data[feat_name] = np.asarray(value)
    return npz_data


def _npz_to_inter_feats_for_model(
    npz_path: Path, model_k: int
) -> Dict[tuple, Any]:
    """Load npz and reconstruct {(k, feat_name): value} for a single model."""
    result: Dict[tuple, Any] = {}
    nested_groups: Dict[str, Dict[str, Any]] = {}

    data = np.load(npz_path, allow_pickle=False)
    for key in data.files:
        arr = data[key]
        if "__" in key:
            base, sub = key.split("__", 1)
            nested_groups.setdefault(base, {})[sub] = arr.item() if arr.ndim == 0 else arr
        else:
            result[(model_k, key)] = arr.item() if arr.ndim == 0 else arr

    for base, sub_dict in nested_groups.items():
        result[(model_k, base)] = sub_dict

    return result


def _compute_single_token_pair(
    token_pair: str,
    xp_data_path: Path,
    models: List[str],
    sanitized_models: List[str],
    compute_config: Dict[str, Any],
    max_tokens: int,
    N_iter: int,
    source_csv_hash: str,
    Answers_df: pl.DataFrame,
    TokenIDs_df: pl.DataFrame,
    compute_inter_features: bool = False,
) -> None:
    """Compute and cache features for a single token pair.

    This is a top-level function (not a closure) so it can be pickled
    by ProcessPoolExecutor.

    Args:
        token_pair: Token pair name to process
        xp_data_path: Base path for feature computation data
        models: List of all model names (LLM + PRNG)
        sanitized_models: Filesystem-safe model names
        compute_config: Feature computation configuration
        max_tokens: Maximum number of tokens / bits per sequence
        N_iter: Number of dataset questions (MainDataset_df.height)
        source_csv_hash: SHA-256 hash of the source dataset CSV
        Answers_df: Answers DataFrame
        TokenIDs_df: Token IDs DataFrame
        compute_inter_features: Whether to compute inter-sample features
    """
    # Ensure logging works in subprocess (handlers may not survive fork)
    _pkg = logging.getLogger("audit_llm")
    if not _pkg.handlers:
        _h = logging.StreamHandler()
        _h.setLevel(logging.INFO)
        _h.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        ))
        _pkg.addHandler(_h)
        _pkg.setLevel(logging.INFO)

    tp_start_time = time.time()
    logger.info("[%s] Starting computation... (RSS=%s)", token_pair, _get_rss_mb())

    token_pair_dir = xp_data_path / token_pair
    intra_dir = token_pair_dir / "intra"
    manifest_path = token_pair_dir / "manifest.json"

    intra_dir.mkdir(parents=True, exist_ok=True)

    if compute_inter_features:
        inter_dir = token_pair_dir / "inter"
        inter_dir.mkdir(parents=True, exist_ok=True)

    # Per-model caching: check if all models are already computed
    all_intra_done = all(
        (intra_dir / f"{sm}.npy").exists()
        for sm in sanitized_models
    )
    if compute_inter_features:
        inter_dir = token_pair_dir / "inter"
        all_inter_done = all(
            (inter_dir / f"{sm}.npz").exists()
            for sm in sanitized_models
        )
    else:
        all_inter_done = True

    if all_intra_done and all_inter_done and manifest_path.exists():
        logger.info("[%s] CACHE HIT — all %d models already computed, skipping.", token_pair, len(models))
        return

    token_union_set = []

    # Calculate number of features
    nb_of_features = (
        2 * rng_tests_dict_to_nb_of_tests(compute_config["features"]["nist_tests"])
        + len(compute_config["features"]["intra_samples_features"])
        + 2 * len(token_union_set)
    )

    # Build inter-sample feature index
    if compute_inter_features:
        inter_samples_features_list = list(compute_config["features"]["inter_samples_features"]) + [
            "nb_of_sequences_after_seq_length_filter"
        ]
        inter_samples_feature_index = {name: idx for idx, name in enumerate(inter_samples_features_list)}
    else:
        inter_samples_feature_index = {}

    intra_samples_feature_index: Optional[Dict[str, int]] = None

    # Seed for PRNG models — deterministic per token pair
    # Use hash of token_pair name to get a unique but reproducible seed offset
    sub_seed = 1

    n_cached = 0
    n_computed = 0
    logger.info("[%s] Processing %d models (inter_features=%s)...", token_pair, len(models), compute_inter_features)

    for k, model in enumerate(models):
        sanitized = sanitized_models[k]
        intra_model_path = intra_dir / f"{sanitized}.npy"

        if compute_inter_features:
            inter_model_path = inter_dir / f"{sanitized}.npz"
            model_cached = intra_model_path.exists() and inter_model_path.exists()
        else:
            model_cached = intra_model_path.exists()

        # Per-model caching
        if model_cached:
            if model in PRNG_MODELS:
                sub_seed += N_iter  # Keep seed consistent even when skipping
            n_cached += 1
            continue

        model_start_time = time.time()

        # Get data for this specific configuration
        if model in PRNG_MODELS:
            extracted_seq_answers_dict = make_random_bit_sequences(
                model, max_tokens, N=N_iter, seed=sub_seed
            )
            extracted_bit_seqs = extracted_seq_answers_dict["extracted_bit_seqs"]
            sub_seed += N_iter
        else:
            # Answers_df and TokenIDs_df are already pre-filtered by token_pair
            filtered_answer_df = Answers_df.filter(pl.col("Model") == model)
            filtered_token_ids_df = TokenIDs_df.filter(pl.col("Model") == model)
            filtered_df = join_dataframes(filtered_answer_df, filtered_token_ids_df)

            # Ordering by col: Dataset_Question Index to make sure index of extracted items are aligned with index of MainDataset
            filtered_df = filtered_df.sort("Dataset_Question Index")

            # Initialize lists with empty strings for all N_iter positions
            extracted_bit_seqs = [""] * N_iter
            extracted_token_ids = [[]] * N_iter

            # Fill in the actual values at their correct indices
            for row in filtered_df.iter_rows(named=True):
                idx = row["Dataset_Question Index"]
                extracted_bit_seqs[idx] = row["Answer"] if isinstance(row["Answer"], str) else ""
                extracted_token_ids[idx] = row["Token_IDs"]

            extracted_seq_answers_dict = {
                "extracted_bit_seqs": extracted_bit_seqs,
                "extracted_seq_token_ids": extracted_token_ids,
            }

        # Make a local copy of compute_config with current model
        local_compute_config = compute_config.copy()
        local_compute_config["model"] = model

        # Compute features if we have answers
        if extracted_bit_seqs:
            # 1. Compute Inter-sample features (if enabled)
            if compute_inter_features and not inter_model_path.exists():
                inter_start = time.time()
                inter_feats = fill_inter_samples_features_map(k, extracted_seq_answers_dict, local_compute_config)
                npz_data = _inter_feats_for_model_to_npz_dict(inter_feats, k)
                np.savez(inter_model_path, **npz_data)
                logger.info("[%s] model %d/%d (%s): inter-features computed in %.1fs",
                            token_pair, k + 1, len(models), model, time.time() - inter_start)

            # 2. Compute Intra-sample features
            if not intra_model_path.exists():
                intra_start = time.time()
                intra_features = compute_intra_samples_bit_feature_matrix(
                    extracted_seq_answers_dict,
                    N_iter,
                    local_compute_config,
                    token_pair,
                )
                model_features_2d, intra_samples_feature_index = intra_features
                np.save(intra_model_path, model_features_2d)
                logger.info("[%s] model %d/%d (%s): intra-features computed in %.1fs",
                            token_pair, k + 1, len(models), model, time.time() - intra_start)

                if "abliterated" in model:
                    logger.debug(
                        "for model=%s, nan percentage = %s%%", model, np.isnan(model_features_2d).mean()*100
                    )
        else:
            if not intra_model_path.exists():
                logger.warning("[%s] All bit none for model: %s", token_pair, model)
                np.save(intra_model_path, np.full((N_iter, nb_of_features), np.nan))
            if compute_inter_features and not inter_model_path.exists():
                np.savez(inter_model_path)  # empty npz for model with no data

        n_computed += 1
        model_elapsed = time.time() - model_start_time
        logger.info("[%s] model %d/%d (%s): total %.1fs", token_pair, k + 1, len(models), model, model_elapsed)

    # Write manifest
    manifest: Dict[str, Any] = {
        "models": sanitized_models,
        "feature_index": intra_samples_feature_index,
        "inter_feature_index": inter_samples_feature_index,
        "source_csv_hash": source_csv_hash,
    }
    write_manifest(manifest, manifest_path)
    tp_elapsed = time.time() - tp_start_time
    logger.info("[%s] DONE — %d cached, %d computed, total %.1fs (%.1f min) (RSS=%s)",
                token_pair, n_cached, n_computed, tp_elapsed, tp_elapsed / 60, _get_rss_mb())


def _resolve_feature_workers() -> int:
    """Resolve the number of worker processes for per-token-pair feature computation.

    Defaults to ``min(8, cpu_count - 2)`` — leave 2 cores for the OS / main process
    (and to curb OpenBLAS/NumPy thread thrashing), and cap at 8 so a high-core host
    does not spawn a huge pool. The cap also bounds peak RAM: with the streaming
    submission below, only ``workers + 1`` per-token-pair slices are materialized at
    once, so fewer workers => lower memory.

    Override with ``FLIPS_FEATURE_WORKERS`` (a positive int, clamped to ``[1, 4*cpu]``).
    Useful on HPC, where the node's total core count is unrelated to a single job's
    allocation, and to trade memory against parallelism on either side. The upper clamp
    guards against a typo (or a pasted total-node-core count) spawning a huge pool and
    re-creating the OOM this streaming path exists to avoid.
    """
    default = min(8, max(1, (os.cpu_count() or 1) - 2))
    raw = os.environ.get("FLIPS_FEATURE_WORKERS")
    if raw is None:
        return default
    try:
        n = int(raw)
    except ValueError:
        logger.warning("Invalid FLIPS_FEATURE_WORKERS=%r; using default %d.", raw, default)
        return default
    if n < 1:
        logger.warning("FLIPS_FEATURE_WORKERS=%d is < 1; clamping to 1.", n)
        return 1
    ceiling = 4 * (os.cpu_count() or 1)
    if n > ceiling:
        logger.warning(
            "FLIPS_FEATURE_WORKERS=%d exceeds %d (4x cores); clamping to %d.", n, ceiling, ceiling
        )
        return ceiling
    return n


def _compute_save_load_experiments(
    Experiments_path: str,
    Answers_df: pl.DataFrame,
    TokenIDs_df: pl.DataFrame,
    MainDataset_df: pl.DataFrame,
    max_tokens: int,
    compute_config: Dict[str, Any],
    Dataset_path: str = "",
    compute_inter_features: bool = False,
) -> Tuple[Dict, Dict, Dict[str, int], Dict, Dict, Dict]:
    """Compute, save, or load experiment data for token_pairs and models.

    Uses per-model .npy / .npz files with JSON manifests.
    Token pairs are computed in parallel using ProcessPoolExecutor.

    Args:
        Experiments_path: Path to experiments directory
        Answers_df: Answers DataFrame
        TokenIDs_df: Token IDs DataFrame
        MainDataset_df: Main token_pair DataFrame
        max_tokens: Maximum number of tokens
        compute_config: Configuration parameters for the experiments
        Dataset_path: Path to source token_pair CSV (for SHA-256 hash validation)
        compute_inter_features: Whether to compute inter-sample features.
            Set to False to skip inter-sample features and save time.

    Returns:
        Tuple containing intra_samples_features_dict, inter_samples_features_map,
        model_index, intra_samples_feature_index_dict, inter_samples_feature_index_dict,
        and compute_config
    """
    total_start_time = time.time()

    # Ensure logs have timestamps even before setup_experiment_logging() is called.
    pkg_logger = logging.getLogger("audit_llm")
    if not pkg_logger.handlers:
        _early_handler = logging.StreamHandler()
        _early_handler.setLevel(logging.INFO)
        _early_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        ))
        pkg_logger.addHandler(_early_handler)
        pkg_logger.setLevel(logging.INFO)
    else:
        _early_handler = None

    logger.info("=== _compute_save_load_experiments START (RSS=%s) ===", _get_rss_mb())
    logger.info("  inter_features=%s", compute_inter_features)

    token_pairs, models, model_index = _prepare_models_with_PRNGs(Answers_df)
    logger.info("  %d token pairs, %d models (incl. PRNGs)", len(token_pairs), len(models))

    # Prepare directory structure
    constant = compute_config.get('set_constant_seq_length')
    suffix = f"_const{constant}" if constant else ""
    xp_data_path = Path(Experiments_path) / f"feature_computation_data{suffix}"
    xp_data_path.mkdir(parents=True, exist_ok=True)

    # model_index as JSON (replaces model_index.pickle)
    model_index_json_path = Path(Experiments_path) / "model_index.json"
    if model_index_json_path.exists():
        existing = load_json(path=model_index_json_path)
        if existing != model_index:
            logger.warning("Existing model_index differs from the newly created one. Overwriting.")
    write_json(model_index, model_index_json_path)

    # Integrate nist test parameters from config
    compute_config = integrate_nist_test_parameters(compute_config)

    # Sanitized model names for file I/O
    sanitized_models = [sanitize_model_name(m) for m in models]

    compute_config["token_stats_dict"] = None

    # Compute token_pair source hash for manifest
    source_csv_hash = compute_file_hash(Dataset_path) if Dataset_path else ""

    N_iter = MainDataset_df.height
    max_workers = _resolve_feature_workers()

    for attempt in range(2):
        # === Computation phase (parallelized across token pairs) ===
        logger.info(
            "Computation phase (attempt %d): processing %d token pairs with %d workers.",
            attempt + 1, len(token_pairs), max_workers,
        )

        ctx = mp.get_context("forkserver")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            # Stream submissions with a sliding window: keep at most `max_workers + 1`
            # token-pair slices in flight, backfilling as workers finish. (The old code
            # pre-filtered and submitted ALL token pairs up front, so the executor's
            # pending-work queue transiently held ~one extra full copy of the row data.)
            # Peak RAM here is (two resident full frames) + (workers+1 in-flight slices);
            # this bounds only the second, *transient* term — from ~len(token_pairs)
            # slices down to workers+1. It does NOT shrink the resident-frame floor: the
            # full Answers/TokenIDs frames stay live for on-demand slicing AND the caller
            # reuses them after this returns, so lowering the worker count trims only the
            # slice window (an OOM on the resident frames is a separate, larger problem).
            # Workers still receive only their own slice, so the per-subprocess
            # serialization saving is preserved.
            futures = {}
            tp_queue = iter(token_pairs)

            # _submit_next mutates `futures` and the shared `tp_queue` iterator without a
            # lock — safe ONLY because it is called from the main thread (the priming loop
            # and the post-result() backfill below). Do NOT wire it to
            # future.add_done_callback(...): that runs on an executor-internal thread and
            # would race on both.
            def _submit_next() -> bool:
                tp = next(tp_queue, None)
                if tp is None:
                    return False
                tp_answers = Answers_df.filter(pl.col("Token_pair") == tp)
                tp_token_ids = TokenIDs_df.filter(pl.col("Token_pair") == tp)
                fut = executor.submit(
                    _compute_single_token_pair,
                    token_pair=tp,
                    xp_data_path=xp_data_path,
                    models=models,
                    sanitized_models=sanitized_models,
                    compute_config=compute_config,
                    max_tokens=max_tokens,
                    N_iter=N_iter,
                    source_csv_hash=source_csv_hash,
                    Answers_df=tp_answers,
                    TokenIDs_df=tp_token_ids,
                    compute_inter_features=compute_inter_features,
                )
                futures[fut] = tp
                return True

            for _ in range(max_workers + 1):
                if not _submit_next():
                    break

            with tqdm(total=len(token_pairs), desc="Processing token_pairs", leave=True) as pbar:
                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        tp = futures.pop(future)
                        try:
                            future.result()
                        except BrokenProcessPool as e:
                            logger.error(
                                "ProcessPool broken while computing token_pair %s. "
                                "This often means a worker was killed by the OS (OOM killer). "
                                "Try reducing FLIPS_FEATURE_WORKERS or freeing RAM. (RSS=%s)",
                                tp, _get_rss_mb(),
                            )
                            raise RuntimeError(
                                f"Worker for token_pair '{tp}' was killed (likely OOM). "
                                f"Current main-process RSS: {_get_rss_mb()}"
                            ) from e
                        except Exception:
                            logger.exception("Error computing token_pair %s (RSS=%s)", tp, _get_rss_mb())
                            raise
                        pbar.update(1)
                        # Backfill the window now that a slot is free.
                        _submit_next()

        # === Loading phase (sequential — I/O-bound, fast) ===
        load_start_time = time.time()
        logger.info("Loading phase: reading cached features for %d token pairs...", len(token_pairs))
        intra_samples_features_dict: Dict[str, np.ndarray] = {}
        intra_samples_feature_index_dict: Dict[str, Any] = {}
        inter_samples_features_map_dict: Dict[str, Dict[tuple, Any]] = {}
        inter_samples_feature_index_dict: Dict[str, Any] = {}
        retry_needed = False

        for token_pair in token_pairs:
            token_pair_dir = xp_data_path / token_pair
            manifest_path = token_pair_dir / "manifest.json"

            try:
                manifest = load_manifest(manifest_path)

                # Validate source hash
                if Dataset_path and manifest.get("source_csv_hash"):
                    current_hash = compute_file_hash(Dataset_path)
                    if current_hash != manifest["source_csv_hash"]:
                        logger.warning(
                            "Dataset CSV hash mismatch for %s. Data may be stale.",
                            token_pair,
                        )

                # Load intra features: reassemble 3D from per-model 2D
                intra_dir = token_pair_dir / "intra"
                slices = []
                for sm in manifest["models"]:
                    npy_path = intra_dir / f"{sm}.npy"
                    try:
                        slices.append(np.load(npy_path))
                    except EOFError:
                        npy_path.unlink()
                        manifest_path.unlink(missing_ok=True)
                        logger.warning(
                            "Corrupted file %s, deleted for recompute on retry.", npy_path
                        )
                        retry_needed = True
                        raise
                stacked = np.stack(slices, axis=1)
                intra_samples_features_dict[token_pair] = stacked
                intra_samples_feature_index_dict[token_pair] = manifest["feature_index"]
                logger.info("  Loaded intra features for %s: shape %s", token_pair, stacked.shape)

                # Load inter features (only if enabled)
                if compute_inter_features:
                    inter_dir = token_pair_dir / "inter"
                    inter_feats_all: Dict[tuple, Any] = {}
                    for k_idx, sm in enumerate(manifest["models"]):
                        npz_path = inter_dir / f"{sm}.npz"
                        if npz_path.exists():
                            model_feats = _npz_to_inter_feats_for_model(npz_path, k_idx)
                            inter_feats_all.update(model_feats)
                    inter_samples_features_map_dict[token_pair] = inter_feats_all
                    inter_samples_feature_index_dict[token_pair] = manifest.get("inter_feature_index", {})
                else:
                    inter_samples_features_map_dict[token_pair] = {}
                    inter_samples_feature_index_dict[token_pair] = {}

            except EOFError:
                pass  # file deleted above; will recompute on retry
            except (FileNotFoundError, IOError) as e:
                logger.warning("Could not load data for token_pair %s: %s", token_pair, e)

        load_elapsed = time.time() - load_start_time
        logger.info("Loading phase completed in %.1fs (RSS=%s)", load_elapsed, _get_rss_mb())

        if not retry_needed:
            break

    total_elapsed = time.time() - total_start_time
    logger.info("=== _compute_save_load_experiments DONE in %.1fs (%.1f min) (RSS=%s) ===",
                total_elapsed, total_elapsed / 60, _get_rss_mb())

    # Clean up early handler if we added one
    if _early_handler is not None:
        pkg_logger.removeHandler(_early_handler)
        _early_handler.close()

    return (
        intra_samples_features_dict,
        inter_samples_features_map_dict,
        model_index,
        intra_samples_feature_index_dict,
        inter_samples_feature_index_dict,
        compute_config,
    )
