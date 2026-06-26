"""Post-inference generation parsing, merging, and analysis dispatch.

Functions
---------
parsing_generations                  — Format generations into Answers.csv.
recursive_models_generations_search  — Walk nested generation directories.
"""

import contextlib
import logging
import os
import shutil
import sys

logger = logging.getLogger(__name__)
from collections import defaultdict
from pathlib import Path
from typing import Optional

# REMOVED: Audit ions_analysis import for lazy loading inside parsing_generations()
# to avoid loading GPU packages at module import time
from audit_llm.file_io import load_json, open_pickle_file, write_json
from audit_llm.system_utils import send_mail


# ------------------------------------------------------------------
# recursive_models_generations_search
# ------------------------------------------------------------------


def recursive_models_generations_search(
    model_map: dict[str, list[str]],
    families_paths: list[str],
    Generations_merged_sub_runs_path: str,
) -> None:
    """Walk nested generation directories to build *model_map* and *families_paths*."""
    for model_generation_family in os.listdir(Generations_merged_sub_runs_path):
        family_path = os.path.join(Generations_merged_sub_runs_path, model_generation_family)
        if not os.path.isdir(family_path):
            continue
        files_in_family_path = os.listdir(family_path)
        if "run_config.json" in files_in_family_path or "run_config.pickle" in files_in_family_path:
            families_paths.append(family_path)
            for model_generation in os.listdir(family_path):
                model_map[model_generation_family].append(model_generation)
        else:
            recursive_models_generations_search(model_map, families_paths, family_path)


# ------------------------------------------------------------------
# parsing_generations
# ------------------------------------------------------------------


def parsing_generations(  # noqa: C901  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    run_name: str,
    Productions_path: str,
    mail_destinataire: str = "",
    mail: bool = False,
    merge_sub_runs: bool = False,
    use_cluster_mirror: bool = False,
    remove_generations: bool = True,
) -> None:
    """Format generations into an all-in-one Answers.csv.

    Runs ``AuditionsAnalysis.model_analysis()`` on the production run.
    Optionally merges sub-runs first.
    """
    # CRITICAL: Import AuditionsAnalysis here (lazy loading) to avoid loading GPU packages
    # at module import time. This ensures that optional dependencies like transformers
    # are only loaded when actually needed.
    from audit_llm.Analysis_Classes import AuditionsAnalysis
    
    run_path = os.path.join(Productions_path, run_name)

    if merge_sub_runs:
        run_path = _merge_sub_runs(run_path, remove_generations)

    output_dir_logs = os.path.join(run_path, "Output_logs", "parsing")
    os.makedirs(output_dir_logs, exist_ok=True)

    stdout_path = os.path.join(output_dir_logs, "output.log")
    stderr_path = os.path.join(output_dir_logs, "errors.log")

    def run() -> None:
        audition_analysis = AuditionsAnalysis(run_path=run_path)
        audition_analysis.use_cluster_mirror = use_cluster_mirror
        audition_analysis.model_analysis()

    if not mail:
        run()
        return

    with (
        open(stdout_path, "w", encoding="utf-8") as stdout_file,
        open(stderr_path, "w", encoding="utf-8") as stderr_file,
    ):
        with (
            contextlib.redirect_stdout(stdout_file),
            contextlib.redirect_stderr(stderr_file),
        ):
            try:
                run()
            except Exception as exc:  # pylint: disable=broad-except
                error_message = f"An error occurred: {exc}"
                logger.error("%s", error_message)
                mail_message = (
                    f"Hello,\n\n"
                    f"The generations for run {run_path} went well!\n"
                    f"But an error occurred for the parsing.\n\n"
                    f"{error_message}"
                )
                send_mail(mail_destinataire, mail_message)
                sys.exit()

        logger.info("Parsing generations succeeded.")

    mail_message = f"Hello,\n\n" f"The generations and parsing for run {run_path} went well!"
    send_mail(mail_destinataire, mail_message)


# ------------------------------------------------------------------
# _merge_sub_runs (private helper)
# ------------------------------------------------------------------


def _merge_sub_runs(run_path: str, remove_generations: bool) -> str:
    """Merge sub-run directories into a single merged path.

    Returns the *merged_sub_runs_path* to use as the new run_path.
    """
    merged_sub_runs_path = os.path.join(run_path, "merged_sub_runs")
    gens_merged_path = os.path.join(merged_sub_runs_path, "Generations")

    os.makedirs(gens_merged_path, exist_ok=True)

    # Build a map of already-merged generations
    model_map: dict[str, list[str]] = defaultdict(list)
    for family in os.listdir(gens_merged_path):
        fam_path = os.path.join(gens_merged_path, family)
        if not os.path.isdir(fam_path):
            continue
        for gen in os.listdir(fam_path):
            model_map[family].append(gen)

    # Load any existing merged_run_config (JSON-first, pickle fallback)
    merged_run_config_json_path = os.path.join(merged_sub_runs_path, "run_config.json")
    merged_run_config_pickle_path = os.path.join(merged_sub_runs_path, "run_config.pickle")

    merged_run_config: Optional[dict] = None
    if os.path.isfile(merged_run_config_json_path):
        merged_run_config = load_json(path=Path(merged_run_config_json_path))
    elif os.path.isfile(merged_run_config_pickle_path):
        merged_run_config = open_pickle_file(merged_run_config_pickle_path)

    # Identify sub-run folders
    excluded = {"merged_sub_runs", "Output_logs", ".ipynb_checkpoints"}
    sub_runs = [d for d in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, d)) and d not in excluded]

    # Process each sub-run
    for sub in sub_runs:
        sub_path = os.path.join(run_path, sub)
        sub_gens_path = os.path.join(sub_path, "Generations")

        for family in os.listdir(sub_gens_path):
            src_family = os.path.join(sub_gens_path, family)
            dest_family = os.path.join(gens_merged_path, family)
            os.makedirs(dest_family, exist_ok=True)

            for gen in os.listdir(src_family):
                if gen in model_map[family]:
                    raise ValueError(f"{gen} already merged in '{family}'")
                shutil.move(os.path.join(src_family, gen), dest_family)

        # Merge run_config (JSON-first, pickle fallback)
        sub_cfg_json_path = os.path.join(sub_path, "run_config.json")
        sub_cfg_pickle_path = os.path.join(sub_path, "run_config.pickle")
        if os.path.exists(sub_cfg_json_path):
            sub_cfg: dict = load_json(path=Path(sub_cfg_json_path))
        elif os.path.exists(sub_cfg_pickle_path):
            sub_cfg = open_pickle_file(sub_cfg_pickle_path)
        else:
            raise FileNotFoundError(f"No run_config found in {sub_path}")

        if merged_run_config is None:
            merged_run_config = sub_cfg
        else:
            for key in (
                "vllm_models",
                "hf_models",
                "openrouter_models",
                "vllm_model_path",
                "hf_model_path",
                "openrouter_model_path",
                "quantization_map",
            ):
                if key in sub_cfg:
                    merged_run_config[key].update(sub_cfg[key])

            merged_run_config["vllm_models_progression"] = f"0/{len(merged_run_config['vllm_models'])}"
            merged_run_config["hf_models_progression"] = f"0/{len(merged_run_config['hf_models'])}"
            merged_run_config["openrouter_models_progression"] = f"0/{len(merged_run_config['openrouter_models'])}"

        # Delete the empty sub-run folder
        shutil.rmtree(sub_path)

    # Save the updated merged_run_config as JSON
    write_json(merged_run_config, Path(merged_run_config_json_path))

    # Remove generation files to save storage space
    if remove_generations:
        for family in os.listdir(gens_merged_path):
            family_path = os.path.join(gens_merged_path, family)
            if not os.path.isdir(family_path):
                continue
            for gen in os.listdir(family_path):
                gen_path = os.path.join(family_path, gen)
                if os.path.isfile(gen_path):
                    os.remove(gen_path)

    logger.info("All sub-runs merged. Remaining directory tree: %s", os.listdir(run_path))

    return merged_sub_runs_path
