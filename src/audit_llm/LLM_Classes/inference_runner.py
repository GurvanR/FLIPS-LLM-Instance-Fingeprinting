"""Inference orchestration — model loading, dispatch, GPU cleanup.

Functions
---------
run_inferences        — Top-level wrapper with stdout/stderr redirection.
multi_model_infer     — Iterate through models and run inference.
vllm_supported_models — Identify which models are supported by vLLM.
"""

import logging
logger = logging.getLogger(__name__)

import contextlib
import gc
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from audit_llm.file_io import load_json, open_pickle_file
from audit_llm.LLM_Classes.run_config import RunConfigDict, update_run_config
from audit_llm.LLM_Classes.vLLM_Classes import vLLM
from audit_llm.path_utils import get_repository_level_path
from audit_llm.system_utils import vllm_version_import_manager
from audit_llm.system_utils import send_mail
from audit_llm.Tokens_analysis.vocab_io import clean_tokenizers_vocab, save_tokenizer_vocab_from_model


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

BANNED_KEY_WORDS: list[str] = [
    "clip",
    "CLIP",
    "transformers",
    "Hate",
    "hate",
    "detector",
    "aimagelab",
    "whisper",
    "detect",
    "-bert-",
    "johngiorgi",
]

HF_MODEL_KW: list[str] = ["t5", "gemma", "camembert", "intfloat"]


# ------------------------------------------------------------------
# vllm_supported_models
# ------------------------------------------------------------------


def vllm_supported_models(model_names: list[str]) -> list[str]:
    """Return the subset of *model_names* supported by the installed vLLM."""
    vllm_models: dict[str, tuple[str, ...]] = vllm_version_import_manager("VLLM_MODELS")

    # Corrections for known model families
    vllm_models["MistralForCausalLM"] = ("mistralai", "MistralForCausalLM")
    vllm_models["command-r_correct"] = ("command-r", "CohereForCausalLM")

    supported: list[str] = [tpl[0] for tpl in vllm_models.values()]
    return list(
        {
            model_name
            for model_name in model_names
            for vllm_class in supported
            if (model_name.split("/")[0].lower() in vllm_class.lower() or vllm_class.lower() in model_name.lower())
        }
    )


# ------------------------------------------------------------------
# run_inferences
# ------------------------------------------------------------------


def run_inferences(
    run_name: str,
    Productions_path: str,
    run: Callable[[], None],
    mail_destinataire: Optional[str] = "",
    mail: bool = False,
) -> None:
    """Execute inference with optional stdout/stderr logging and email alerts."""
    if not mail or not mail_destinataire:
        run()
        return

    run_path = os.path.join(Productions_path, run_name)
    output_dir_logs = os.path.join(run_path, "Output_logs", "generations")
    os.makedirs(output_dir_logs, exist_ok=True)

    stdout_path = os.path.join(output_dir_logs, "output.log")
    stderr_path = os.path.join(output_dir_logs, "errors.log")

    with (
        open(stdout_path, "w", encoding="utf-8") as stdout_file,
        open(stderr_path, "w", encoding="utf-8") as stderr_file,
    ):
        with (
            contextlib.redirect_stdout(stdout_file),
            contextlib.redirect_stderr(stderr_file),
        ):
            try:
                logger.info("Starting inference process for run: %s", run_path)
                run()
                logger.info("Inference process completed successfully.")
            except Exception as exc:
                error_message = f"An error occurred during inference: {exc}"
                logger.error("%s", error_message)
                mail_message = (
                    f"Hello,\n\n"
                    f"An error occurred for Run {run_path}. "
                    f"Details below:\n\n"
                    f"{error_message}\n\n"
                    f"Please investigate the issue."
                )
                send_mail(mail_destinataire, mail_message)
                sys.exit(1)


# ------------------------------------------------------------------
# multi_model_infer
# ------------------------------------------------------------------


def multi_model_infer(run_name: str, Productions_path: str) -> None:
    """Iterate through configured models and run vLLM inference."""
    import torch  # pylint: disable=import-outside-toplevel
    destroy_model_parallel: Callable[[], None] = vllm_version_import_manager("destroy_model_parallel")

    run_path = os.path.join(Productions_path, run_name)

    # JSON-first, pickle fallback
    run_config_json_path = os.path.join(run_path, "run_config.json")
    run_config_pickle_path = os.path.join(run_path, "run_config.pickle")

    if os.path.exists(run_config_json_path):
        run_config: RunConfigDict = load_json(path=Path(run_config_json_path))
    elif os.path.exists(run_config_pickle_path):
        run_config = open_pickle_file(run_config_pickle_path)
    else:
        raise FileNotFoundError(f"No run_config found in {run_path}")

    if not run_config["Initial_checkpoint"]:
        logger.info("Advanced checkpoint found.")

    logger.info("VLLM models:")
    for vllm_model, is_done in run_config.get("vllm_models", {}).items():
        logger.info("%s, is done: %s", vllm_model, is_done)

    logger.info("HF models:")
    for hf_model, is_done in run_config.get("hf_models", {}).items():
        logger.info("%s, is done: %s", hf_model, is_done)

    logger.info("OpenRouter models:")
    for openrouter_model, is_done in run_config.get("openrouter_models", {}).items():
        logger.info("%s, is done: %s", openrouter_model, is_done)

    if run_config["hours_delay"] > 0:
        logger.info("Delaying run by %s hours.", run_config['hours_delay'])
        time.sleep(run_config["hours_delay"] * 3600)

    dataset_path = Path(get_repository_level_path()) / run_config["Dataset_relative_path"]

    # Process VLLM models
    logger.info("DEALING WITH VLLM MODELS: %s", run_config["vllm_models"])
    for model_name, is_done in run_config["vllm_models"].items():
        if not is_done:
            model_path = run_config["vllm_model_path"][model_name]
            logger.info("CURRENT MODEL (VLLM): %s", model_name)
            logger.info("MODEL PATH: %s", model_path)
            from audit_llm.file_io import get_base_model_name
            save_tokenizer_vocab_from_model(get_base_model_name(model_name), model_path)

            audited_llm = vLLM(model_name, run_config, run_path, model_path)

            audited_llm.multi_dataset_infer(dataset_path)
            run_config["vllm_models"][model_name] = True
            update_run_config(run_config, run_name, Productions_path)

            destroy_model_parallel()
            del audited_llm
            gc.collect()
            torch.cuda.empty_cache()

    logger.info("Grouping and cleaning stored tokenizers.")
    clean_tokenizers_vocab()
