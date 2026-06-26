# SPDX-FileCopyrightText: 2024 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Directory creation, folder discovery, and path helper utilities."""

import logging
import os

logger = logging.getLogger(__name__)


def make_dir(path: str) -> None:
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def make_exp_dir(
    Experiments_path: str,
    global_exp_name: str = "",
    exp_name: str | None = None,
) -> str:
    """Create auto-incrementing experiment directory with standard subdirs."""
    if exp_name is None:
        counter = 0
        Exp_path = Experiments_path + f"Exp{counter}/"
        while os.path.exists(Exp_path):
            counter += 1
            Exp_path = Experiments_path + f"Exp{counter}/"
        make_dir(Exp_path)
    else:
        Exp_path = Experiments_path + global_exp_name + exp_name + "/"

    make_dir(Exp_path + "DecisionTreesCSV")
    make_dir(Exp_path + "Figures")
    make_dir(Exp_path + "Exp_Raw_Data")

    return Exp_path


def make_path_and_create_folder(save_path: str, model_name: str) -> str:
    """Create a model folder under save_path, handling 'org/model' names."""
    splitted = model_name.split("/")
    folder_path = os.path.join(save_path, model_name)
    if len(splitted) == 2:
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    if len(splitted) > 2:
        raise ValueError(" The name of the model contains more than one '/' ")

    return folder_path


def proper_answer_read(path: str) -> list[str]:
    """Read proper_answers.txt and return comma-separated values."""
    path += "/proper_answers.txt"
    with open(path, mode="r", encoding="utf-8") as file:
        proper_answers = file.read().split(",")

    return proper_answers


def check_storage(folder_path: str, whole_only: bool = True) -> None:
    """Print the storage usage of a folder or its subfolders."""
    if not os.path.exists(folder_path):
        logger.error("The provided path does not exist: %s", folder_path)
        return

    if whole_only:
        folder_size = get_folder_size(folder_path)
        logger.info("Total size of the folder '%s': %.2f MB", folder_path, folder_size)
    else:
        for folder_name in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, folder_name)
            if os.path.isdir(subfolder_path):
                folder_size = get_folder_size(subfolder_path)
                logger.info("%s: %.2f MB", folder_name, folder_size)


def get_folder_size(folder_path: str) -> float:
    """Return total folder size in megabytes."""
    total_size = 0
    for dirpath, _dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
    return total_size / (1024 * 1024)


def get_folder_paths(dir_path: str) -> list[str]:
    """Return paths of every subfolder in dir_path."""
    contents = os.listdir(dir_path)
    folder_paths = [os.path.join(dir_path, item) for item in contents if os.path.isdir(os.path.join(dir_path, item))]
    return folder_paths


def get_csv_file_path(folder_path: str) -> str:
    """Return path to the single CSV file in a folder."""
    csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]

    if len(csv_files) == 1:
        return os.path.join(folder_path, csv_files[0])
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV file found in the folder: {folder_path}")
    raise ValueError("More than one CSV file found in the folder.")


def get_prompt_config(folder_path: str) -> str:
    """Return path to prompt_dataset_config.pickle in a folder."""
    config_files = [file for file in os.listdir(folder_path) if file == "prompt_dataset_config.pickle"]

    if len(config_files) == 1:
        return os.path.join(folder_path, config_files[0])
    if len(config_files) == 0:
        raise FileNotFoundError("No prompt_dataset_config file found in the folder.")
    raise ValueError("More than one prompt_dataset_config file found in the folder.")


def get_repository_level_path() -> str:
    """Walk up from this file until the .git directory is found."""
    repository_level = os.path.abspath(__file__)

    while not os.path.isdir(os.path.join(repository_level, ".git")) and repository_level != os.path.dirname(
        repository_level
    ):
        repository_level = os.path.abspath(os.path.join(repository_level, "../"))

    return repository_level


def _is_cluster_mirror_path(path: str) -> bool:
    """Return True if the path is an HPC-local mirror path (contains 'lustre')."""
    return "lustre" in path


# ------------------------------------------------------------------
# Per-model storage path helpers
# ------------------------------------------------------------------

from pathlib import Path


def get_answers_dir(base_path: str) -> Path:
    """Return the answers directory for per-model Parquet files."""
    return Path(base_path) / "answers"


def get_token_ids_dir(base_path: str) -> Path:
    """Return the token_ids directory for per-model Parquet files."""
    return Path(base_path) / "token_ids"


def get_logprobs_dir(base_path: str) -> Path:
    """Return the logprobs directory for per-model Parquet files."""
    return Path(base_path) / "logprobs"
