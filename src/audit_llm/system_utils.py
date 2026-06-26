# SPDX-FileCopyrightText: 2024 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""System utilities: vLLM compat, argparse, email, cache, and debug/print."""

import argparse
import contextlib
import logging
import os

logger = logging.getLogger(__name__)
import shutil
import smtplib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from packaging import version

from audit_llm.config import _ROOT_PATH


def print_list_diff(list1: list, list2: list) -> None:
    """Log side-by-side diff of two lists."""
    max_len = max(len(list1), len(list2))
    logger.warning("List difference detected:")
    logger.warning("%-7s | %-20s | %-20s", "Index", "new_model_index", "self.model_index")
    logger.warning("-" * 55)

    for i in range(max_len):
        val1 = list1[i] if i < len(list1) else "<missing>"
        val2 = list2[i] if i < len(list2) else "<missing>"
        if val1 != val2:
            logger.warning("%-7d | %-20s | %-20s", i, str(val1), str(val2))


def get_openrouter_key() -> str:
    """Return the OpenRouter API key from the OPENROUTER_API_KEY environment variable."""
    try:
        return os.environ["OPENROUTER_API_KEY"]
    except KeyError:
        print(
            "OPENROUTER_API_KEY is not set. "
            "Copy .env.example to .env and fill in your key."
        )
        raise


def _import_vllm_components() -> dict[str, Any]:
    """Import vLLM components for the installed version."""
    try:
        import vllm  # type: ignore[import-not-found]  # pylint: disable=import-error
    except ImportError as exc:
        raise ImportError(
            "vllm is required for this operation but is not installed. "
            "Install the generation extras: poetry install --extras generation"
        ) from exc

    vllm_version = version.parse(vllm.__version__)

    if vllm_version <= version.parse("0.4.0"):
        return _import_vllm_pre_04()
    if vllm_version < version.parse("0.7.0"):
        return _import_vllm_04_to_07()
    return _import_vllm_07_plus()


def _import_vllm_pre_04() -> dict[str, Any]:
    """Import vLLM components for version <= 0.4.0."""
    try:
        from vllm.model_executor.models import _MODELS as VLLM_MODELS  # type: ignore[attr-defined]
    except ImportError as e:
        raise ImportError(f"Failed to import vLLM models: {e}") from e
    try:
        from vllm.model_executor.parallel_utils.parallel_state import (  # type: ignore[import-not-found]
            destroy_model_parallel,
        )
    except ImportError as e:
        raise ImportError(f"Failed to import destroy_model_parallel: {e}") from e
    return {
        "max_context": "max_context_len_to_capture",
        "VLLM_MODELS": VLLM_MODELS,
        "destroy_model_parallel": destroy_model_parallel,
    }


def _import_vllm_04_to_07() -> dict[str, Any]:
    """Import vLLM components for version 0.4.x to 0.6.x."""
    try:
        from vllm.model_executor.models import _GENERATION_MODELS as VLLM_MODELS  # type: ignore[attr-defined]
    except ImportError as e:
        raise ImportError(f"Failed to import vLLM models: {e}") from e
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel  # type: ignore[attr-defined]
    except ImportError as e:
        raise ImportError(f"Failed to import destroy_model_parallel: {e}") from e
    return {
        "max_context": "max_seq_len_to_capture",
        "VLLM_MODELS": VLLM_MODELS,
        "destroy_model_parallel": destroy_model_parallel,
    }


def _import_vllm_07_plus() -> dict[str, Any]:
    """Import vLLM components for version >= 0.7.0."""
    try:
        from vllm.model_executor.models.registry import (  # type: ignore[import-not-found]  # noqa: E501
            _TEXT_GENERATION_MODELS as VLLM_MODELS,
        )
    except ImportError as e:
        raise ImportError(f"Failed to import vLLM models: {e}") from e
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel  # type: ignore[attr-defined]
    except ImportError as e:
        raise ImportError(f"Failed to import destroy_model_parallel: {e}") from e
    return {
        "max_context": "max_seq_len_to_capture",
        "VLLM_MODELS": VLLM_MODELS,
        "destroy_model_parallel": destroy_model_parallel,
    }


def vllm_version_import_manager(item: str) -> Any:
    """Dynamically import vLLM components based on version compatibility."""
    components = _import_vllm_components()

    if item not in components:
        raise KeyError(f"Invalid item requested: '{item}'. " f"Available keys: {list(components.keys())}")

    return components[item]


def argparsing(args: dict[str, dict]) -> dict:
    """Build an argparse parser from a dict specification and parse sys.argv."""
    parser = argparse.ArgumentParser()

    for name, properties in args.items():
        arg_action = properties["action"]
        arg_default = properties.get("default", None)
        arg_type = properties.get("type", None)

        if arg_type is bool:
            if arg_default is True:
                parser.add_argument(
                    arg_action,
                    action="store_false",
                    dest=name,
                    help=f"Disable {name}",
                )
            else:
                parser.add_argument(
                    arg_action,
                    action="store_true",
                    dest=name,
                    help=f"Enable {name}",
                )
        elif arg_type is not None:
            parser.add_argument(arg_action, type=arg_type, default=arg_default)
        else:
            parser.add_argument(arg_action, default=arg_default)

    parsed_args = parser.parse_args()
    return vars(parsed_args)


def fprint(**kwargs: Any) -> None:
    """Log each kwarg as 'name = value' at DEBUG level."""
    for var_name, var_value in kwargs.items():
        logger.debug("%s = %s", var_name, var_value)


def model_eraser(model_path: str | Path) -> None:
    """Delete a model directory."""
    try:
        shutil.rmtree(model_path)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Error: %s", exc)


def clear_cache_of_models(
    model_dir_path: str | None = None,
) -> None:
    """Erase every loaded model from the cache directory."""
    if model_dir_path is None:
        model_dir_path = str(
            Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
        )
    for directory in os.listdir(model_dir_path):
        if directory.startswith("models"):
            model_eraser(os.path.join(model_dir_path, directory))


# --- Email utilities ---

mail_destinataire = os.environ.get("FLIPS_ALERT_EMAIL", "")


def send_mail(mail_dest: str, message: str) -> None:
    """Send an email via localhost SMTP. No-op if *mail_dest* is empty."""
    if not mail_dest:
        return
    smtp_obj = smtplib.SMTP("localhost")
    smtp_obj.sendmail(os.environ.get("FLIPS_SMTP_SENDER", "noreply@example.com"), mail_dest, message.encode("utf8"))


def run_with_mail(save_path: str, run: Callable[[], None], SEND: bool = True) -> None:
    """Execute a callable, log stdout, and email on completion or error."""
    output_dir_logs = save_path + "Output_logs/"
    os.makedirs(output_dir_logs, exist_ok=True)

    stdout_path = os.path.join(output_dir_logs, "output.log")

    with open(stdout_path, "w", encoding="utf-8") as stdout_file:
        with contextlib.redirect_stdout(stdout_file):
            try:
                run()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                error_message = f"An error occurred: {exc}"
                logger.error("%s", error_message)
                mail_message = f" Hello, \n an error for Run {save_path} " f"has occured, sorry. \n {error_message}"
                if SEND:
                    send_mail(mail_destinataire, mail_message)
                sys.exit()

            logger.info("Run succeeded.")
    mail_message = f" Hello, \n The run {save_path} went well ! "
    if SEND:
        send_mail(mail_destinataire, mail_message)
