"""Run configuration assembly, validation, and persistence.

Functions
---------
make_run_config    — Create a unified run configuration dictionary.
update_run_config  — Persist progress after each model completes.
add_model_to_database      — Append a model entry to the specs CSV.
filter_model_specifications — Filter the model-specifications DataFrame.
"""

import logging
logger = logging.getLogger(__name__)

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TypedDict

import polars as pl

from audit_llm.file_io import load_json, open_pickle_file, write_json
from audit_llm.models_management.model_names import (
    vllm_model_names_to_path,
)
from audit_llm.path_utils import get_repository_level_path


# ---------------------------------------------------------------------------
# Shared type: run_config dict structure
# ---------------------------------------------------------------------------


class RunConfigDict(TypedDict, total=False):
    """Structure of the run_config dict created by make_run_config and consumed
    by inference_runner, General_LLM_Class, and vLLM_Classes."""

    run_name: str
    scrapping_rule: str
    min_seq_length: int
    dyn_checking_batch_size: int
    TOKEN_PAIRS_SET: str
    MAX_TOKENS: int
    vllm_model_config: dict[str, Any]
    hf_model_config: dict[str, Any]
    openrouter_model_config: dict[str, Any]
    Dataset_relative_path: str
    hours_delay: float
    outlines: bool
    model_path_JZ: str | None
    created_at: str
    Initial_checkpoint: bool
    vllm_models: dict[str, bool]
    hf_models: dict[str, bool]
    openrouter_models: dict[str, bool]
    vllm_model_path: dict[str, str]
    hf_model_path: dict[str, str | None]
    openrouter_model_path: dict[str, str | None]
    vllm_models_progression: str
    hf_models_progression: str
    openrouter_models_progression: str
    environment: dict[str, str | None]
    quantization_map: dict[str, str]


def add_model_to_database(
    model_specifications: dict[str, Any],
    model_specifications_df_path: str = ("datasets/model_specifications_df/model_specifications_df_1.csv"),
) -> None:
    """Add or update a model entry in the model-specifications CSV."""
    model_specifications_default: dict[str, Any] = {
        "model_name": "NAME ?",
        "language": "english",
        "more_than_14b": 0,
        "vllm": False,
        "on_Lab": False,
        "license": "OK",
        "ChatBot_Finetuned": False,
        "checked_working": False,
        "working": True,
        "satisfying_results": False,
    }

    updated_specifications = {
        **model_specifications_default,
        **model_specifications,
    }

    new_model_df = pl.DataFrame([updated_specifications])
    model_specifications_df = pl.read_csv(model_specifications_df_path)

    # Remove the model if already existed.
    if updated_specifications["model_name"] in model_specifications_df["model_name"].unique():
        model_specifications_df = model_specifications_df.filter(
            pl.col("model_name") != updated_specifications["model_name"]
        )

    model_specifications_df = pl.concat([model_specifications_df, new_model_df])
    model_specifications_df.write_csv(model_specifications_df_path)


def filter_model_specifications(model_specifications_df: pl.DataFrame, filters: dict[str, Any]) -> pl.DataFrame:
    """Filter model-specifications DataFrame by column conditions."""
    conditions: list[pl.Expr] = []
    clone_filters = filters.copy()
    clone_filters.pop("path")  # not a column

    for key, value in clone_filters.items():
        if isinstance(value, list):
            condition = pl.col(key).is_in(value)
        else:
            condition = pl.col(key) == value
        conditions.append(condition)

    combined_condition = conditions[0]
    for condition in conditions[1:]:
        combined_condition = combined_condition & condition

    return model_specifications_df.filter(combined_condition)


# ------------------------------------------------------------------
# Lazy import helper — avoids circular import with inference_runner
# ------------------------------------------------------------------


def _vllm_supported_models(model_names: list[str]) -> list[str]:
    """Resolve vLLM-supported models (delegates to inference_runner)."""
    # Import here to avoid circular dependency at module level
    from audit_llm.LLM_Classes.inference_runner import (  # pylint: disable=import-outside-toplevel
        vllm_supported_models,
    )

    return vllm_supported_models(model_names)


# ------------------------------------------------------------------
# Environment metadata for reproducibility
# ------------------------------------------------------------------


def _collect_env_metadata() -> dict[str, str | None]:
    """Capture environment versions for reproducibility.

    Returns a dict with keys: python_version, vllm_version,
    torch_version, cuda_version, gpu_type.  Missing packages
    are recorded as ``None``.
    """
    import platform
    import subprocess as _sp

    metadata: dict[str, str | None] = {
        "python_version": platform.python_version(),
    }

    # torch
    try:
        import torch
        metadata["torch_version"] = torch.__version__
        metadata["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            metadata["gpu_type"] = torch.cuda.get_device_name(0)
        else:
            metadata["gpu_type"] = None
    except ImportError:
        metadata["torch_version"] = None
        metadata["cuda_version"] = None
        metadata["gpu_type"] = None

    # vllm
    try:
        import vllm
        metadata["vllm_version"] = vllm.__version__
    except ImportError:
        metadata["vllm_version"] = None

    return metadata


# ------------------------------------------------------------------
# make_run_config
# ------------------------------------------------------------------


def make_run_config(  # noqa: C901  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    scrapping_rule: str,
    run_name: str,
    Inf_config: dict[str, Any],
    models_configs: dict[str, Any],
    vllm_model_config: dict[str, Any],
    hf_model_config: dict[str, Any],
    open_router_config: dict[str, Any],
    Dataset_relative_path: str,
    Productions_relative_path: str,
    hours_delay: float,
    model_names: Optional[list[str]] = None,
    banned_models: Optional[list[str]] = None,
    model_specifications_filters: Optional[dict[str, Any]] = None,
    outlines: bool = False,
    model_cache_path: Optional[str] = None,
    working_lib: Optional[list[str]] = None,
    ALL_OPEN_ROUTER: bool = False,
    quantization_map: Optional[dict[str, str]] = None,
) -> str:
    """Create a unified run configuration and save it as a pickle.

    Returns the *run_name* for convenience.
    """
    if model_names is None:
        model_names = []
    if quantization_map is None:
        quantization_map = {mn: "no_quantized" for mn in model_names}

    # Models that work with vLLM but not on the HPC cluster → force HF pipeline
    hf_forced_models = [
        "meta-llama/Llama-3.3-70B-Instruct",
        "tiiuae/falcon-40b-instruct",
        "tiiuae/falcon-40b",
        "tiiuae/falcon-7b-instruct",
        "tiiuae/falcon-7b",
        "google/recurrentgemma-2b-it",
        "facebook/opt-66b",
        "facebook/opt-6.7b",
    ]

    repository_level = get_repository_level_path()
    productions_path = os.path.join(repository_level, Productions_relative_path)
    run_path = os.path.join(productions_path, run_name)

    if model_specifications_filters is not None:
        path_to_model_specifications_df = model_specifications_filters["path"]
        all_model_specifications_df = pl.read_csv(path_to_model_specifications_df)
        model_specifications_df = filter_model_specifications(all_model_specifications_df, model_specifications_filters)

        if model_names:
            model_specifications_df = model_specifications_df.filter(pl.col("model_name").is_in(model_names))
        if banned_models:
            model_specifications_df = model_specifications_df.filter(~pl.col("model_name").is_in(banned_models))

        model_specifications_df = model_specifications_df.sort(pl.col("model_name"))

        vllm_models_for_run: list[str] = list(
            model_specifications_df.filter(pl.col("vllm") == True)["model_name"].unique()  # noqa: E712
        )
        hf_models_for_run: list[str] = list(
            model_specifications_df.filter(pl.col("vllm") == False)["model_name"].unique()  # noqa: E712
        )
        openrouter_models_for_run: list[str] = []
    else:
        all_hf_forcing = False
        openrouter_models_for_run = []
        if ALL_OPEN_ROUTER:
            openrouter_models_for_run = list(model_names)
            vllm_models_for_run = []
            hf_models_for_run = []
        elif all_hf_forcing:
            vllm_models_for_run = []
            hf_models_for_run = list(model_names)
        elif working_lib is not None:
            working_lib_lower = [lib.lower() for lib in working_lib]
            if any("vllm" in lib for lib in working_lib_lower):
                vllm_models_for_run = list(model_names)
                hf_models_for_run = []
            elif any("transformers" in lib for lib in working_lib_lower):
                vllm_models_for_run = []
                hf_models_for_run = list(model_names)
            else:
                raise ValueError(f"unidentified woorking lib ({working_lib}), " "moving to automatic lib choice")
        else:
            vllm_models_for_run = _vllm_supported_models(model_names)
            for model in hf_forced_models:
                if model in vllm_models_for_run:
                    vllm_models_for_run.remove(model)
            hf_models_for_run = [mn for mn in model_names if mn not in vllm_models_for_run]

    openrouter_models_for_run.sort()
    vllm_models_for_run.sort()
    hf_models_for_run.sort()

    # Build vllm_model_path (use base model name for path resolution so
    # quantization-suffixed variants map to the same physical checkpoint)
    from audit_llm.file_io import get_base_model_name

    vllm_model_path: dict[str, str]
    if model_cache_path is None:
        base_names = list({get_base_model_name(mn) for mn in vllm_models_for_run})
        base_paths = vllm_model_names_to_path(base_names)
        vllm_model_path = {mn: base_paths[get_base_model_name(mn)] for mn in vllm_models_for_run}
    else:
        if models_configs.get("local_cache", False):
            vllm_model_path = {}
            for mn in vllm_models_for_run:
                base_mn = get_base_model_name(mn)
                if models_configs.get("filename", None) is not None:
                    base_names_only = [get_base_model_name(m) for m in vllm_models_for_run]
                    assert len(set(base_names_only)) == 1, (
                        "When using local_cache with filename, " "only one model should be specified."
                    )
                    model_path = os.path.join(base_mn, models_configs["filename"])
                else:
                    model_path = base_mn
                vllm_model_path[mn] = os.path.join("Local_LLMs", model_path)
        else:
            vllm_model_path = {mn: os.path.join(model_cache_path, get_base_model_name(mn)) for mn in vllm_models_for_run}

    logger.debug("vllm_model_path: %s", vllm_model_path)

    run_config: RunConfigDict = {
        "run_name": run_name,
        "scrapping_rule": scrapping_rule,
        "min_seq_length": Inf_config["min_seq_length"],
        "dyn_checking_batch_size": Inf_config["dyn_checking_batch_size"],
        "TOKEN_PAIRS_SET": Inf_config["TOKEN_PAIRS_SET"],
        "MAX_TOKENS": Inf_config["MAX_TOKENS"],
        "vllm_model_config": vllm_model_config,
        "hf_model_config": hf_model_config,
        "openrouter_model_config": open_router_config,
        "Dataset_relative_path": Dataset_relative_path,
        "hours_delay": hours_delay,
        "outlines": outlines,
        "model_path_JZ": model_cache_path,  # legacy schema key — kept for released run_config compatibility
        "created_at": datetime.now().strftime("%m/%d/%Y %H:%M"),
        "Initial_checkpoint": True,
        "vllm_models": {mn: False for mn in vllm_models_for_run},
        "hf_models": {mn: False for mn in hf_models_for_run},
        "openrouter_models": {mn: False for mn in openrouter_models_for_run},
        "vllm_model_path": vllm_model_path,
        "hf_model_path": {
            mn: (None if model_cache_path is None else os.path.join(model_cache_path, mn)) for mn in hf_models_for_run
        },
        "openrouter_model_path": {mn: None for mn in openrouter_models_for_run},
        "vllm_models_progression": f"0/{len(vllm_models_for_run)}",
        "hf_models_progression": f"0/{len(hf_models_for_run)}",
        "openrouter_models_progression": f"0/{len(openrouter_models_for_run)}",
        "environment": _collect_env_metadata(),
        "quantization_map": quantization_map,
    }

    # Merge with former run_config if checkpoint exists (JSON-first, pickle fallback)
    run_config_json_path = os.path.join(run_path, "run_config.json")
    run_config_pickle_path = os.path.join(run_path, "run_config.pickle")

    former_run_config: Optional[dict[str, Any]] = None
    if os.path.exists(run_config_json_path):
        former_run_config = load_json(path=Path(run_config_json_path))
    elif os.path.exists(run_config_pickle_path):
        former_run_config = open_pickle_file(run_config_pickle_path)

    if former_run_config is not None:
        run_config["Initial_checkpoint"] = former_run_config["Initial_checkpoint"]

        # Previous models that are done won't be rerun
        run_config["vllm_models"].update(former_run_config["vllm_models"])
        run_config["hf_models"].update(former_run_config["hf_models"])
        run_config["openrouter_models"].update(former_run_config["openrouter_models"])
        run_config["vllm_model_path"].update(former_run_config["vllm_model_path"])
        run_config["hf_model_path"].update(former_run_config["hf_model_path"])
        run_config["openrouter_model_path"].update(former_run_config["openrouter_model_path"])
        if "quantization_map" in former_run_config:
            run_config["quantization_map"].update(former_run_config["quantization_map"])

        run_config["vllm_models_progression"] = f"0/{len(run_config['vllm_models'])}"
        run_config["hf_models_progression"] = f"0/{len(run_config['hf_models'])}"
        run_config["openrouter_models_progression"] = f"0/{len(run_config['openrouter_models'])}"

    # Save the unified run_config as JSON
    write_json(run_config, Path(run_config_json_path))

    # Log run metadata in a CSV
    log_file_path = os.path.join(productions_path, "run_metadata.csv")
    log_entry = {
        "run_name": run_name,
        "created_at": run_config["created_at"],
        "Dataset_relative_path": Dataset_relative_path,
        "vllm_models_progression": run_config["vllm_models_progression"],
        "hf_models_progression": run_config["hf_models_progression"],
        "openrouter_models_progression": run_config["openrouter_models_progression"],
    }

    write_header = not os.path.exists(log_file_path)
    with open(log_file_path, mode="a", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=log_entry.keys())
        if write_header:
            csv_writer.writeheader()
        csv_writer.writerow(log_entry)

    return run_name


# ------------------------------------------------------------------
# update_run_config
# ------------------------------------------------------------------


def update_run_config(run_config: RunConfigDict, run_name: str, Productions_path: str) -> None:
    """Persist the updated run_config and refresh progress in metadata CSV."""
    run_config["Initial_checkpoint"] = False

    run_path = os.path.join(Productions_path, run_name)
    run_config_path = os.path.join(run_path, "run_config.json")
    write_json(run_config, Path(run_config_path))

    # Update run_metadata with progression details
    log_file_path = os.path.join(Productions_path, "run_metadata.csv")

    vllm_done = sum(done for done in run_config["vllm_models"].values())
    hf_done = sum(done for done in run_config["hf_models"].values())
    openrouter_done = sum(done for done in run_config["openrouter_models"].values())

    vllm_total = len(run_config["vllm_models"])
    hf_total = len(run_config["hf_models"])
    openrouter_total = len(run_config["openrouter_models"])
    run_config["vllm_models_progression"] = f"{vllm_done}/{vllm_total}"
    run_config["hf_models_progression"] = f"{hf_done}/{hf_total}"
    run_config["openrouter_models_progression"] = f"{openrouter_done}/{openrouter_total}"

    updated_rows: list[dict[str, str]] = []
    with open(log_file_path, mode="r", newline="", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if row["run_name"] == run_name:
                row["vllm_models_progression"] = run_config["vllm_models_progression"]
                row["hf_models_progression"] = run_config["hf_models_progression"]
                row["openrouter_models_progression"] = run_config["openrouter_models_progression"]
            updated_rows.append(row)

    with open(log_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=updated_rows[0].keys())
        csv_writer.writeheader()
        csv_writer.writerows(updated_rows)
