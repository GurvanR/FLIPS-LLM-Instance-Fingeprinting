"""One-time migration utilities for storage format changes.

Functions
---------
migrate_run_config_pickle_to_json      — Convert run_config.pickle to run_config.json.
migrate_generation_checkpoints         — Convert generation pickle checkpoints to Parquet.
migrate_monolithic_parquet             — Split monolithic Parquet into per-model files.
migrate_3d_npy_to_per_model            — Split 3D .npy matrices into per-model .npy + .npz files.
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from audit_llm.file_io import (
    compute_file_hash,
    load_json,
    sanitize_model_name,
    write_json,
    write_manifest,
    write_per_model_parquet,
)


# ------------------------------------------------------------------
# run_config.pickle → run_config.json
# ------------------------------------------------------------------


def migrate_run_config_pickle_to_json(run_path: Path) -> None:
    """Convert run_config.pickle to run_config.json in *run_path*.

    Skips if run_config.json already exists. Leaves the pickle file in place.
    """
    run_path = Path(run_path)
    json_path = run_path / "run_config.json"
    pickle_path = run_path / "run_config.pickle"

    if json_path.exists():
        logger.info("  run_config.json already exists in %s, skipping.", run_path)
        return

    if not pickle_path.exists():
        logger.info("  No run_config.pickle found in %s, skipping.", run_path)
        return

    with open(pickle_path, "rb") as f:
        run_config: dict = pickle.load(f)  # noqa: S301

    write_json(run_config, json_path)
    logger.info("  Migrated %s -> %s", pickle_path, json_path)


# ------------------------------------------------------------------
# Generation checkpoints pickle → Parquet
# ------------------------------------------------------------------


def migrate_generation_checkpoints(generations_path: Path) -> None:
    """Convert generation pickle checkpoints to Parquet in *generations_path*.

    Walks recursively through model generation directories and converts
    *_generations*.pickle files to Parquet.
    """
    generations_path = Path(generations_path)
    if not generations_path.exists():
        logger.info("  Path %s does not exist, skipping.", generations_path)
        return

    for root, _dirs, files in os.walk(generations_path):
        for filename in files:
            if "generations" in filename and filename.endswith(".pickle"):
                pickle_file = Path(root) / filename
                parquet_file = pickle_file.with_suffix(".parquet")

                if parquet_file.exists():
                    continue

                try:
                    with open(pickle_file, "rb") as f:
                        generations = pickle.load(f)  # noqa: S301

                    # Import here to avoid circular imports at module level
                    from audit_llm.LLM_Classes.General_LLM_Class import (
                        LLM_Generation,
                    )

                    if generations and isinstance(generations[0], LLM_Generation):
                        rows = [gen.to_row_dict() for gen in generations]
                        df = pd.DataFrame(rows)
                        df.to_parquet(parquet_file, index=False)
                        logger.info("  Migrated %s -> %s", pickle_file, parquet_file)
                    else:
                        logger.info("  Skipping %s: not a list of LLM_Generation", pickle_file)

                except Exception as e:  # pylint: disable=broad-except
                    logger.error("  Error migrating %s: %s", pickle_file, e)


# ------------------------------------------------------------------
# Monolithic Parquet → per-model partitioned Parquet
# ------------------------------------------------------------------


def migrate_monolithic_parquet(base_path: Path) -> None:
    """Split monolithic Answers/TokenIDs/Logprobs Parquet into per-model files.

    Creates ``answers/``, ``token_ids/``, and ``logprobs/`` subdirectories
    under *base_path* with one Parquet file per model.
    """
    base_path = Path(base_path)

    _split_monolithic(
        monolithic_path=base_path / "Answers.parquet",
        output_dir=base_path / "answers",
        model_col="Model",
    )

    _split_monolithic(
        monolithic_path=base_path / "TokenIDsAnswers.parquet",
        output_dir=base_path / "token_ids",
        model_col="Model",
    )

    logprobs_path = base_path / "Logprobs.parquet"
    if logprobs_path.exists():
        _split_monolithic(
            monolithic_path=logprobs_path,
            output_dir=base_path / "logprobs",
            model_col="model_name",
        )


def _split_monolithic(monolithic_path: Path, output_dir: Path, model_col: str) -> None:
    """Split a monolithic Parquet file into per-model files."""
    if not monolithic_path.exists():
        logger.info("  %s not found, skipping.", monolithic_path)
        return

    if output_dir.exists() and any(output_dir.glob("*.parquet")):
        logger.info("  %s already has per-model files, skipping.", output_dir)
        return

    df = pd.read_parquet(monolithic_path)
    if model_col not in df.columns:
        logger.warning("  Column '%s' not found in %s, skipping.", model_col, monolithic_path)
        return

    for model_name, model_df in df.groupby(model_col):
        write_per_model_parquet(model_df, output_dir, str(model_name))

    logger.info("  Split %s into %d per-model files in %s", monolithic_path, len(df[model_col].unique()), output_dir)


# ------------------------------------------------------------------
# 3D .npy → per-model .npy + .npz + JSON manifests
# ------------------------------------------------------------------


def migrate_3d_npy_to_per_model(
    xp_data_path: Path,
    model_index_path: Optional[Path] = None,
    dataset_csv_path: Optional[Path] = None,
) -> None:
    """Split 3D .npy matrices and pickle inter-feature maps into per-model files.

    Reads old-style directories:
      - ``intra_samples_features_N/{dataset}.npy``
      - ``intra_samples_features_N/intra_samples_feature_index_{dataset}.pickle``
      - ``inter_samples_features_map_M/{dataset}.pickle``
      - ``inter_samples_features_map_M/inter_samples_feature_index_{dataset}.pickle``

    Creates new-style per-model directories:
      - ``{dataset}/intra/{sanitized_model}.npy``
      - ``{dataset}/inter/{sanitized_model}.npz``
      - ``{dataset}/manifest.json``

    Args:
        xp_data_path: Path to ``feature_computation_data/`` directory.
        model_index_path: Path to ``model_index.pickle`` (or ``.json``).
        dataset_csv_path: Optional path to source CSV for hash computation.
    """
    xp_data_path = Path(xp_data_path)

    # Load model_index
    model_index: Dict[str, int] = {}
    if model_index_path is not None:
        model_index_path = Path(model_index_path)
        if model_index_path.suffix == ".json":
            model_index = load_json(path=model_index_path)
        elif model_index_path.suffix == ".pickle":
            with open(model_index_path, "rb") as f:
                model_index = pickle.load(f)  # noqa: S301
            # Also save as JSON
            json_path = model_index_path.with_suffix(".json")
            write_json(model_index, json_path)
            logger.info("  Migrated model_index to JSON: %s", json_path)

    if not model_index:
        logger.warning("  No model_index found, cannot migrate 3D npy files.")
        return

    # Invert model_index: {idx: model_name}
    idx_to_model = {v: k for k, v in model_index.items()}
    models_ordered = [idx_to_model[i] for i in range(len(idx_to_model))]
    sanitized_models = [sanitize_model_name(m) for m in models_ordered]

    # Source CSV hash
    source_csv_hash = compute_file_hash(dataset_csv_path) if dataset_csv_path else ""

    # Find intra directories (intra_samples_features_N)
    intra_dirs = sorted(xp_data_path.glob("intra_samples_features_*"))
    inter_dirs = sorted(xp_data_path.glob("inter_samples_features_map_*"))

    for intra_dir in intra_dirs:
        for npy_file in intra_dir.glob("*.npy"):
            dataset = npy_file.stem
            logger.info("  Migrating intra features for dataset: %s", dataset)

            token_pair_dir = xp_data_path / dataset
            new_intra_dir = token_pair_dir / "intra"
            new_intra_dir.mkdir(parents=True, exist_ok=True)

            # Load 3D matrix: (N_iter, N_models, N_features)
            matrix_3d = np.load(npy_file)

            if matrix_3d.shape[1] != len(models_ordered):
                logger.warning(
                    "3D matrix has %d models but model_index has %d. Skipping.",
                    matrix_3d.shape[1],
                    len(models_ordered),
                )
                continue

            # Split into per-model 2D files
            for k, sanitized in enumerate(sanitized_models):
                out_path = new_intra_dir / f"{sanitized}.npy"
                if not out_path.exists():
                    np.save(out_path, matrix_3d[:, k, :])

            # Load feature index
            feature_index: Optional[Dict[str, int]] = None
            index_pickle = intra_dir / f"intra_samples_feature_index_{dataset}.pickle"
            if index_pickle.exists():
                with open(index_pickle, "rb") as f:
                    feature_index = pickle.load(f)  # noqa: S301

            # Load inter features if available
            inter_feature_index: Optional[Dict[str, int]] = None
            for inter_dir in inter_dirs:
                inter_pickle = inter_dir / f"{dataset}.pickle"
                inter_idx_pickle = inter_dir / f"inter_samples_feature_index_{dataset}.pickle"

                if inter_pickle.exists():
                    logger.info("  Migrating inter features for dataset: %s", dataset)
                    with open(inter_pickle, "rb") as f:
                        inter_map: Dict[tuple, Any] = pickle.load(f)  # noqa: S301

                    new_inter_dir = token_pair_dir / "inter"
                    new_inter_dir.mkdir(parents=True, exist_ok=True)

                    # Split per-model
                    for k, sanitized in enumerate(sanitized_models):
                        out_path = new_inter_dir / f"{sanitized}.npz"
                        if out_path.exists():
                            continue
                        npz_data: Dict[str, np.ndarray] = {}
                        for (mk, feat_name), value in inter_map.items():
                            if mk != k:
                                continue
                            if isinstance(value, np.ndarray):
                                npz_data[feat_name] = value
                            elif isinstance(value, dict):
                                for sub_key, sub_val in value.items():
                                    npz_data[f"{feat_name}__{sub_key}"] = np.asarray(sub_val)
                            else:
                                npz_data[feat_name] = np.asarray(value)
                        np.savez(out_path, **npz_data)

                if inter_idx_pickle.exists():
                    with open(inter_idx_pickle, "rb") as f:
                        inter_feature_index = pickle.load(f)  # noqa: S301

            # Write manifest
            manifest: Dict[str, Any] = {
                "models": sanitized_models,
                "feature_index": feature_index,
                "inter_feature_index": inter_feature_index,
                "source_csv_hash": source_csv_hash,
            }
            write_manifest(manifest, token_pair_dir / "manifest.json")
            logger.info("  Wrote manifest for %s", dataset)
