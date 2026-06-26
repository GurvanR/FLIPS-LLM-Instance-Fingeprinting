# SPDX-FileCopyrightText: 2024 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

import logging
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PACKAGE_PATH = Path(__file__).resolve().parent
SRC_PATH = PACKAGE_PATH.parent
_ROOT_PATH = PACKAGE_PATH.parent.parent  # Accessible from clone of the project, not from package


class Config(BaseSettings):
    verbose: bool = Field(False, description="Enable / disable verbose logging")
    systemd_logging: bool = Field(
        True,
        description="Enable / disable logging with systemd.journal.JournalHandler",
    )

    model_config = SettingsConfigDict(
        env_file=Path(".env"),
        json_schema_extra={"description": "Handle the environment variable for the app configuration"},
    )


env_config = Config()  # type: ignore[call-arg]  # pydantic BaseSettings


class PathConfig(BaseSettings):
    """Centralized path constants, overridable via environment variables."""

    datasets_dir: Path = Field(
        default=_ROOT_PATH / "datasets",
        description="Base directory for datasets",
    )
    xp_configs_dir: Path = Field(
        default=_ROOT_PATH / "XP_configs" / "XP_config_libs",
        description="Directory for experiment configuration presets",
    )
    inference_configs_path: Path = Field(
        default=_ROOT_PATH / "scripts" / "Inference_configs.yaml",
        description="Path to inference configuration YAML",
    )
    hf_cache_dir: Path = Field(
        default=Path.home() / ".cache" / "huggingface" / "hub",
        description="HuggingFace model cache directory (hub subdirectory)",
    )
    model_cache_dir: Path | None = Field(
        None,
        description="HPC model cache root; None = use HF default (~/.cache/huggingface/hub)",
    )
    model_specs_path: Path = Field(
        default=_ROOT_PATH / "datasets" / "model_specifications_df" / "model_specifications_df_1.csv",
        description="Path to model specifications CSV",
    )

    model_config = SettingsConfigDict(
        env_prefix="AUDIT_LLM_",
        env_file=Path(".env"),
    )

    @model_validator(mode="after")
    def _resolve_cache_dir(self) -> "PathConfig":
        if self.model_cache_dir is not None:
            self.hf_cache_dir = self.model_cache_dir
        return self


path_config = PathConfig()  # type: ignore[call-arg]  # pydantic BaseSettings


def use_systemd_config() -> None:
    """Configure root logger to use systemd journal handler."""
    if not env_config.systemd_logging:
        return

    from systemd import (  # type: ignore[import-not-found]  # pylint: disable=import-outside-toplevel,import-error  # noqa: E501
        journal,
    )

    # remove the default handler, if already initialized
    existing_handlers = logging.getLogger().handlers
    for handler in existing_handlers:
        logging.getLogger().removeHandler(handler)
    # Sending logs to systemd-journal when run via systemd.
    logging.getLogger().addHandler(journal.JournalHandler())
