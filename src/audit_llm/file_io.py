# SPDX-FileCopyrightText: 2024 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""File I/O utilities for CSV, Parquet, JSON, pickle, INI, and per-model storage formats."""

import configparser
import csv
import hashlib
import json
import logging

logger = logging.getLogger(__name__)
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

from audit_llm.config import _ROOT_PATH


def write_data(
    header: list[str],
    data: list[list[Any]],
    path: str,
    file_format: str = "csv",
) -> None:
    """Save data as CSV or Parquet."""
    if file_format == "csv":
        mode = "a" if os.path.exists(path) else "w"
        with open(path, mode, newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            if mode == "w":
                csv_writer.writerow(header)
            csv_writer.writerows(data)

    elif file_format == "parquet":
        df = pd.DataFrame(data, columns=header)

        if os.path.exists(path):
            # Parquet append -> read + concat + overwrite
            existing_df = pd.read_parquet(path)
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_parquet(path, index=False)

    else:
        raise ValueError(f"Unsupported file_format: {file_format}")


def erase_file(file_path: str) -> bool:
    """Delete a file, returning True on success."""
    try:
        os.remove(file_path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        logger.error("An error occurred while erasing the file %s: %s", file_path, exc)
        return False
    logger.info("File %s has been erased successfully.", file_path)
    return True


def open_pickle_file(pickle_path: Union[str, Path]) -> Any:
    """Load and return a pickled object."""
    with open(pickle_path, "rb") as pickle_file:
        return pickle.load(pickle_file)  # noqa: S301


def save_pickle_file(obj: Any, pickle_path: Union[str, Path]) -> None:
    """Save an object to a pickle file."""
    with open(pickle_path, "wb") as pickle_file:
        pickle.dump(obj, pickle_file)


def load_json(
    filename: str = "",
    path: Path = _ROOT_PATH / "datasets",
    set_keys_as_int: bool = False,
) -> Union[dict, list]:
    """Load JSON data, optionally converting keys to int."""
    if filename:
        path = path / f"{filename}.json"

    with open(path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    if set_keys_as_int:
        data_list = {int(k): v for k, v in data_list.items()}

    return data_list


def write_json(data: Any, path: Path) -> None:
    """Write data to a JSON file with indent=4."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# ------------------------------------------------------------------
# Model name sanitization
# ------------------------------------------------------------------


def sanitize_model_name(name: str) -> str:
    """Replace '/' with '__' for safe filenames."""
    return name.replace("/", "__")


def unsanitize_model_name(name: str) -> str:
    """Reverse sanitize_model_name: '__' back to '/'."""
    return name.replace("__", "/")


QUANTIZATION_SEPARATOR = "@@"


def get_base_model_name(name: str) -> str:
    """Strip the @@quantization suffix, if present, returning the base HF model ID."""
    return name.split(QUANTIZATION_SEPARATOR, 1)[0]


def get_quantization_suffix(name: str) -> str | None:
    """Return the quantization suffix after @@, or None if absent."""
    parts = name.split(QUANTIZATION_SEPARATOR, 1)
    return parts[1] if len(parts) == 2 else None


# ------------------------------------------------------------------
# File hashing
# ------------------------------------------------------------------


def compute_file_hash(path: Union[str, Path]) -> str:
    """Compute SHA-256 hash of a file's contents."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()}"


# ------------------------------------------------------------------
# Per-model Parquet I/O
# ------------------------------------------------------------------


def write_per_model_parquet(
    df: pd.DataFrame,
    directory: Path,
    model_name: str,
) -> None:
    """Write a DataFrame as a per-model Parquet file.

    Args:
        df: DataFrame to write (already filtered to one model).
        directory: Directory to write into (e.g., answers/).
        model_name: Unsanitized model name (e.g., 'meta-llama/Llama-3.1-8B').
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sanitized = sanitize_model_name(model_name)
    path = directory / f"{sanitized}.parquet"
    df.to_parquet(path, index=False)


def read_parquet_dir_or_monolithic(
    directory: Path,
    monolithic_path: Path,
) -> pl.DataFrame:
    """Read per-model Parquet directory, falling back to monolithic file.

    Backward compatibility: if per-model directory exists
    and contains .parquet files, read them all; otherwise read the monolithic file.
    """
    directory = Path(directory)
    monolithic_path = Path(monolithic_path)

    if directory.exists() and any(directory.glob("*.parquet")):
        return pl.scan_parquet(directory).collect()

    if monolithic_path.exists():
        return pl.scan_parquet(monolithic_path).collect()

    raise FileNotFoundError(
        f"Neither per-model directory {directory} nor monolithic file {monolithic_path} found."
    )


# ------------------------------------------------------------------
# JSON manifests
# ------------------------------------------------------------------


def write_manifest(manifest: dict, path: Union[str, Path]) -> None:
    """Write a JSON manifest file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest, path)


def load_manifest(path: Union[str, Path]) -> dict:
    """Load a JSON manifest file."""
    return load_json(path=Path(path))


# ------------------------------------------------------------------
# Per-model feature I/O
# ------------------------------------------------------------------


def save_intra_features(
    dataset_dir: Path,
    model_name: str,
    features: np.ndarray,
) -> None:
    """Save per-model intra-sample features as .npy.

    Args:
        dataset_dir: e.g., feature_computation_data/{dataset}/
        model_name: Unsanitized model name.
        features: 2D array of shape (N_iter, N_features).
    """
    intra_dir = Path(dataset_dir) / "intra"
    intra_dir.mkdir(parents=True, exist_ok=True)
    sanitized = sanitize_model_name(model_name)
    np.save(intra_dir / f"{sanitized}.npy", features)


def load_intra_features(dataset_dir: Path) -> Tuple[np.ndarray, Dict[str, int]]:
    """Load all per-model intra features and stack into 3D array.

    Returns:
        (X, feature_index) where X has shape (N_iter, N_models, N_features).
    """
    dataset_dir = Path(dataset_dir)
    manifest = load_manifest(dataset_dir / "manifest.json")
    feature_index: Dict[str, int] = manifest["feature_index"]
    models: list[str] = manifest["models"]

    slices = []
    for model in models:
        path = dataset_dir / "intra" / f"{model}.npy"
        slices.append(np.load(path))

    X = np.stack(slices, axis=1)  # (N_iter, N_models, N_features)
    return X, feature_index


def save_inter_features(
    dataset_dir: Path,
    model_name: str,
    npz_data: Dict[str, np.ndarray],
) -> None:
    """Save per-model inter-sample features as .npz.

    Args:
        dataset_dir: e.g., feature_computation_data/{dataset}/
        model_name: Unsanitized model name.
        npz_data: Dict of {array_name: np.ndarray} to save.
    """
    inter_dir = Path(dataset_dir) / "inter"
    inter_dir.mkdir(parents=True, exist_ok=True)
    sanitized = sanitize_model_name(model_name)
    np.savez(inter_dir / f"{sanitized}.npz", **npz_data)


def load_inter_features(dataset_dir: Path) -> Dict[tuple, Any]:
    """Load all per-model inter features into dict keyed by (model_idx, feature_name).

    Reconstructs the format expected by downstream consumers.
    """
    dataset_dir = Path(dataset_dir)
    manifest = load_manifest(dataset_dir / "manifest.json")
    models: list[str] = manifest["models"]

    inter_dir = dataset_dir / "inter"
    inter_map: Dict[tuple, Any] = {}

    for k, model in enumerate(models):
        npz_path = inter_dir / f"{model}.npz"
        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=False)
            for key in data.files:
                inter_map[(k, key)] = data[key]

    return inter_map


def write_dict_on_file(experiment_config: dict[str, Any], file_path: Path) -> None:
    """Write a summary of the experiment_config dict to file_path."""
    fp = Path(file_path)
    fp.parent.mkdir(parents=True, exist_ok=True)

    with fp.open("w", encoding="utf-8") as f:
        # --- analysis_df ---
        df = experiment_config.get("analysis_df", None)
        f.write("=== analysis_df ===\n")
        if df is not None:
            cols = df.columns if hasattr(df, "columns") else list(df.schema.keys())
            f.write("Columns: " + ", ".join(cols) + "\n")

            if isinstance(df, pl.DataFrame):
                sample_df = df.sample(n=min(10, df.height), seed=0)
                pdf = sample_df.to_pandas()
            else:
                sample_df = df.sample(n=min(10, len(df)), random_state=0) if hasattr(df, "sample") else df
                pdf = sample_df if isinstance(sample_df, pd.DataFrame) else pd.DataFrame(sample_df)
            f.write(pdf.to_string(index=False))
        else:
            f.write("None\n")
        f.write("\n\n")

        # --- intra_samples_features_dict keys ---
        f.write("=== intra_samples_features_dict keys ===\n")
        intra_dict = experiment_config.get("intra_samples_features_dict", None)
        if intra_dict is not None:
            for k in intra_dict:
                f.write(f"- {k}\n")
        else:
            f.write("None\n")
        f.write("\n")

        # --- inter_samples_features_map keys ---
        f.write("=== inter_samples_features_map keys ===\n")
        inter_map = experiment_config.get("inter_samples_features_map", None)
        if inter_map is not None:
            for k in inter_map:
                f.write(f"- {k}\n")
        else:
            f.write("None\n")
        f.write("\n")

        # --- token_stats_dict first 5 union ---
        f.write("=== token_stats_dict (first 5 of ['union']) ===\n")
        token_stats_dict = experiment_config.get("token_stats_dict", None)
        if token_stats_dict is not None:
            for dataset, stats in token_stats_dict.items():
                union = stats.get("union", []) if stats else []
                f.write(f"{dataset}: {union[:5]}\n")
        else:
            f.write("None\n")
        f.write("\n")

        # --- pretty-print xp_config only ---
        f.write("=== xp_config ===\n")
        f.write(
            json.dumps(
                experiment_config.get("xp_config", None),
                indent=2,
                sort_keys=True,
            )
        )
        f.write("\n\n")

        # --- the remaining simple dumps ---
        for key in ("save_fig_path", "temperatures", "datasets", "models"):
            f.write(f"=== {key} ===\n" f"{repr(experiment_config.get(key, None))}\n\n")


def ini_to_dict(file_path: str) -> dict:
    """Parse an INI file to a nested dict with type coercion."""
    config = configparser.ConfigParser()
    config.read(file_path)

    data: dict[str, Any] = {}
    for section in config.sections():
        data[section] = dict(config.items(section))

        # Convert numeric values
        if "time_scaling" in data[section]:
            data[section]["time_scaling"] = float(data[section]["time_scaling"])
        if "nb_of_gpu" in data[section]:
            data[section]["nb_of_gpu"] = int(data[section]["nb_of_gpu"])
        if "models" in data[section]:
            data[section]["models"] = [model.strip() for model in data[section]["models"].split(",")]
        if "model_big_set" in data[section]:
            data[section]["model_big_set"] = data[section]["model_big_set"].split(",")
        if "working_lib" in data[section]:
            data[section]["working_lib"] = data[section]["working_lib"].split(",")
        if "filename" in data[section]:
            data[section]["filename"] = str(data[section]["filename"])
        if "local_cache" in data[section]:
            data[section]["local_cache"] = data[section]["local_cache"].lower() == "true"
        if "quantization" in data[section]:
            data[section]["quantization"] = [q.strip() for q in data[section]["quantization"].split(",")]

    return data
