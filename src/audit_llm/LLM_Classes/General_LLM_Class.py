"""Base class for LLM auditing backends and the LLM_Generation data class."""

import logging
logger = logging.getLogger(__name__)

import abc
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import polars as pl

from audit_llm.Bits_Generation.Bits_Dataset_Making.Prompts import (
    prompt_idx_to_actual_prompt,
)
from audit_llm.Bits_Generation.parsing_bits_tools import (
    token_pair_string_to_list,
)
from audit_llm.Bits_Generation.Token_Pairs_Sets import (
    TOKEN_PAIRS_SETS_DICT,
)
from audit_llm.LLM_Classes.fallback_templates import (
    format_with_fallback,
)
from audit_llm.LLM_Classes.model_tokenizer_map import (
    model_tokenizer_map,
)
from audit_llm.LLM_Classes.run_config import RunConfigDict
from audit_llm.file_io import load_json, open_pickle_file, save_pickle_file
from audit_llm.path_utils import make_path_and_create_folder


# Best-effort import for jinja2 template errors
try:
    from jinja2.exceptions import TemplateError
except Exception:  # pylint: disable=broad-except
    TemplateError = Exception  # type: ignore[misc,assignment]


class LLM_for_audit(abc.ABC):
    """Base class implementing the Template Method pattern for LLM inference."""

    def __init__(
        self,
        model_name: str,
        run_path: str,
        run_config: RunConfigDict,
        model_path: Optional[str] = None,
    ) -> None:
        """Initialise base fields from the run configuration.

        Args:
            model_name: HuggingFace model identifier.
            run_path: Production directory for this run.
            run_config: Unified run configuration dictionary.
            model_path: Optional local or remote model path override.
        """
        self.max_tokens: str = "Default"

        self.dataframe: pd.DataFrame = pd.DataFrame()
        self.model_name: str = model_name
        self.model: str = model_path if model_path is not None else model_name

        self.Dataset_path: str = ""
        self.proper_answers: list[str] = []

        # Data saving
        if run_path:
            self.generations_path: str = make_path_and_create_folder(os.path.join(run_path, "Generations"), model_name)
            logger.debug("model_name=%s, generations_path=%s", model_name, self.generations_path)
            self.sampling_key_tuple: str = "NO DATASET"
            self.execution_time: float = 1.0

        self.run_path: str = run_path
        self.model_type: str = "DEFAULT"

        self.min_seq_length: int = run_config["min_seq_length"]
        self.dyn_checking_batch_size: int = run_config["dyn_checking_batch_size"]
        self.token_pairs_set: list[str] = TOKEN_PAIRS_SETS_DICT[run_config["TOKEN_PAIRS_SET"]]
        self.max_inference_checkpoint_batch_size: int = run_config.get("max_inference_checkpoint_batch_size", 250)
        self.max_gen_counter: int = 5

        self.sampling_config_completed: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Abstract method — must be overridden by each backend
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _generate(self, sampling_key: dict[str, float]) -> None:
        """Run model inference for one sampling-key group."""

    # ------------------------------------------------------------------
    # Config merging helper
    # ------------------------------------------------------------------

    @staticmethod
    def _build_config(defaults: dict[str, Any], overrides: dict[str, Any], **extra: Any) -> dict[str, Any]:
        """Merge default config with overrides and extra key-value pairs."""
        return {**defaults, **overrides, **extra}

    # ------------------------------------------------------------------
    # Multi-dataset / single-dataset inference
    # ------------------------------------------------------------------

    def multi_dataset_infer(self, Dataset_path: Path) -> None:
        """Run inference on every sampling-key group in the dataset CSV."""
        sampling_params_of_df = ["temperature", "frequency_penalty"]

        self.dataframe = pl.read_csv(Dataset_path)  # type: ignore[assignment]

        missing_cols = [c for c in ["Index", *sampling_params_of_df] if c not in self.dataframe.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in dataset: {missing_cols}")

        sampling_keys = self.dataframe.select(["temperature", "frequency_penalty"]).unique().to_dicts()

        generations_done = os.listdir(self.generations_path)
        self._initialize_tokeniser()
        logger.info("Tokenizer initialized")

        for sampling_key in sampling_keys:
            self.sampling_key_str = self.sampling_key_to_string(sampling_key)

            sampling_key_done = any(
                self.sampling_key_str in gen_file
                and "generations_terminated" in gen_file
                and (gen_file.endswith(".parquet") or gen_file.endswith(".pickle"))
                for gen_file in generations_done
            )

            if not sampling_key_done:
                logger.debug("Sampling Key %s : NOT DONE", self.sampling_key_str)
                self.infer(sampling_key)
            else:
                logger.debug("Sampling Key %s : DONE", self.sampling_key_str)

    def infer(self, sampling_key: dict[str, float]) -> None:
        """Run inference for a single sampling-key group."""
        self._load_generation(sampling_key)

        logger.info("Begins inferences for dataset %s.", self.sampling_key_tuple)
        start_time = time.perf_counter()

        self._generate(sampling_key)

        end_time = time.perf_counter()
        self.execution_time = end_time - start_time

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def _does_template_have_system(self) -> bool:
        """Check whether the tokenizer's chat template supports a system role."""
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template is None:
            return False
        return "system" in chat_template

    def _format_prompts_for_model(
        self, sampling_key: dict[str, float], verbose: bool = False
    ) -> list["LLM_Generation"]:
        """Format user prompts for the model, with fallback chain.

        Returns:
            List of LLM_Generation instances with formatted prompts.
        """
        if self.model_type == "openrouter":
            raise NotImplementedError(
                f"Model {self.model_name} is using openrouter. " "Openrouter formatting not implemented yet."
            )

        system_prompt_list: list[str] = load_json("system_prompts")  # type: ignore[assignment]
        formatted: list[LLM_Generation] = []

        has_apply = hasattr(self.tokenizer, "apply_chat_template")
        has_system_template = self._does_template_have_system()

        if not has_apply:
            logger.warning(
                "Tokenizer %s has no apply_chat_template. Using fallback formatting. Model: %s",
                self.tokenizer.__class__.__name__,
                self.model_name,
            )
        elif not has_system_template:
            logger.warning(
                "Tokenizer %s chat template does not support 'system' role. "
                "Will try merging system into user where needed. Model: %s",
                self.tokenizer.__class__.__name__,
                self.model_name,
            )

        # Filter by sampling params
        sampling_key_df = self.dataframe.filter(
            (pl.col("temperature") == sampling_key["temperature"])  # type: ignore[arg-type]
            & (pl.col("frequency_penalty") == sampling_key["frequency_penalty"])
        )

        for row in sampling_key_df.iter_rows(named=True):  # type: ignore[call-overload]
            dataset_idx: int = row["Index"]
            raw_prompt_idx = row["prompt_idx"]
            system_prompt_idx: int = row["system_prompt_idx"] if "system_prompt_idx" in self.dataframe.columns else -1
            system_prompt: Optional[str] = system_prompt_list[system_prompt_idx] if system_prompt_idx != -1 else None

            for token_pair in self.token_pairs_set:
                raw_prompt = prompt_idx_to_actual_prompt(raw_prompt_idx, token_pair)

                if has_apply:
                    text = self._apply_chat_template(
                        raw_prompt,
                        system_prompt,
                        has_system_template,
                        verbose,
                    )
                else:
                    text = format_with_fallback(self.model_name, raw_prompt, system_prompt, verbose)

                # Ensure BOS token is present if tokenizer defines one
                if self.tokenizer.bos_token is not None and not text.startswith(self.tokenizer.bos_token):
                    text = self.tokenizer.bos_token + text

                formatted.append(
                    LLM_Generation(
                        formatted_prompt=text,
                        token_pair=token_pair,
                        dataset_idx=dataset_idx,
                    )
                )

        return formatted

    def _apply_chat_template(
        self,
        raw_prompt: str,
        system_prompt: Optional[str],
        has_system_template: bool,
        verbose: bool,
    ) -> str:
        """Try apply_chat_template with fallback chain."""
        messages: list[dict[str, str]] = []
        if has_system_template and system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": raw_prompt})

        text: Optional[str] = None
        try:
            try:
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except TypeError:
                text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception as e:  # pylint: disable=broad-except
            msg = str(e) or ""
            is_template_error = (
                isinstance(e, TemplateError)
                or "system role not supported" in msg.lower()
                or ("system" in msg.lower() and "not supported" in msg.lower())
            )
            if is_template_error and system_prompt:
                merged_user = f"{system_prompt}\n\n{raw_prompt}"
                retry_messages = [{"role": "user", "content": merged_user}]
                try:
                    try:
                        text = self.tokenizer.apply_chat_template(
                            retry_messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    except TypeError:
                        text = self.tokenizer.apply_chat_template(retry_messages, tokenize=False)
                except Exception as e2:  # pylint: disable=broad-except
                    if verbose:
                        logger.warning("apply_chat_template retry failed for model %s: %s. Falling back.", self.model_name, e2)
                    text = format_with_fallback(self.model_name, raw_prompt, system_prompt, verbose)
            else:
                if verbose:
                    logger.warning("apply_chat_template failed for model %s: %s. Falling back.", self.model_name, e)
                text = format_with_fallback(self.model_name, raw_prompt, system_prompt, verbose)

        assert text is not None
        return text

    # ------------------------------------------------------------------
    # Checkpoint / persistence
    # ------------------------------------------------------------------

    def _save_generations(self, dataset_terminated: bool = False) -> None:
        """Save generations as a Parquet checkpoint."""
        if not os.path.exists(self.generations_path):
            os.makedirs(self.generations_path)

        gen_dir = Path(self.generations_path)
        parquet_checkpoint = gen_dir / f"{self.sampling_key_str}_generations.parquet"
        pickle_checkpoint = gen_dir / f"{self.sampling_key_str}_generations.pickle"

        if dataset_terminated:
            # Remove checkpoint files (both formats)
            for p in (parquet_checkpoint, pickle_checkpoint):
                if p.exists():
                    p.unlink()
            save_path = gen_dir / f"{self.sampling_key_str}_generations_terminated.parquet"
            # Also remove old pickle terminated file if it exists
            old_pickle_terminated = gen_dir / f"{self.sampling_key_str}_generations_terminated.pickle"
            if old_pickle_terminated.exists():
                old_pickle_terminated.unlink()
        else:
            save_path = parquet_checkpoint

        rows = [gen.to_row_dict() for gen in self.generations]
        df = pd.DataFrame(rows)
        logger.info("Saving generations to %s", save_path)
        df.to_parquet(save_path, index=False)

        execution_time_path = gen_dir / f"{self.sampling_key_str}_execution_time.txt"
        with open(execution_time_path, "w", encoding="utf-8") as f:
            f.write(str(self.execution_time))

    def _load_generation(self, sampling_key: dict[str, float]) -> None:
        """Load existing generations from checkpoint or format new prompts.

        Tries Parquet first, falls back to pickle for backward compatibility.
        """
        gen_dir = Path(self.generations_path)
        parquet_path = gen_dir / f"{self.sampling_key_str}_generations.parquet"
        pickle_path = gen_dir / f"{self.sampling_key_str}_generations.pickle"

        if parquet_path.exists():
            logger.info("Loading existing generations from %s", parquet_path)
            df = pd.read_parquet(parquet_path)
            self.generations: list[LLM_Generation] = [
                LLM_Generation.from_row_dict(row) for row in df.to_dict(orient="records")
            ]
        elif pickle_path.exists():
            logger.info("Loading existing generations from %s (legacy)", pickle_path)
            self.generations = open_pickle_file(pickle_path)
        else:
            logger.info("No existing generations found for %s. Formatting prompts.", self.sampling_key_str)
            self.generations = self._format_prompts_for_model(sampling_key)

    # ------------------------------------------------------------------
    # Tokeniser
    # ------------------------------------------------------------------

    def _initialize_tokeniser(self) -> None:
        """Lazy-initialise the tokenizer from the model."""
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            return
        from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

        self.tokenizer = model_tokenizer_map.get(self.model_name, AutoTokenizer).from_pretrained(
            self.model,
            padding_side="left",
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,
        )
        if self.tokenizer.eos_token is None:
            raise ValueError("NO EOS TOKEN FOR:", self.model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def sampling_key_to_string(self, sampling_key: dict[str, float]) -> str:
        """Convert a sampling-key dict to a filename-safe string."""
        return "-".join([str(val) for val in sampling_key.values()])

    def store_completion_elements(self, LLM_gen: "LLM_Generation", output: Any) -> None:
        """Extract output text, token_ids, and logprobs from a completion."""
        completion = output.outputs[0]
        LLM_gen.output_text = completion.text
        LLM_gen.output_token_ids = completion.token_ids
        if self.model_type == "vllm" and self.sampling_config_completed["logprobs"] is not None:
            LLM_gen.output_logprobs = self.extract_logprobs(completion, LLM_gen.token_pair)
        else:
            LLM_gen.output_logprobs = None

        LLM_gen.formatted_prompt = ""  # free memory

    def extract_logprobs(self, completion: Any, token_pair_str: str) -> list[dict[str, Any]]:
        """Extract per-step log probabilities for a target token pair.

        If a target token is in the top-k, its exact logprob is recorded.
        Otherwise, the minimum logprob in the returned set is used as a floor.
        """
        assert self.model_type == "vllm", "Logprob extraction only implemented for vllm model type."

        token_pair_list = token_pair_string_to_list(token_pair_str)
        target_map = {
            token_pair_list[0]: "tA",
            token_pair_list[1]: "tB",
        }
        target_count = len(target_map)

        logprobs_data: list[dict[str, Any]] = []

        for step_lp_dict in completion.logprobs:
            lp_values = list(step_lp_dict.values())
            floor_via_index = lp_values[-1].logprob

            found_data: dict[str, Any] = {}
            for lp in lp_values:
                if lp.decoded_token in target_map:
                    label = target_map[lp.decoded_token]
                    found_data[label] = lp
                    if len(found_data) == target_count:
                        break

            row: dict[str, Any] = {}
            for label in ["tA", "tB"]:
                if label in found_data:
                    row[f"logprob_{label}"] = found_data[label].logprob
                    row[f"rank_{label}"] = found_data[label].rank
                    row[f"floored_{label}"] = False
                else:
                    row[f"logprob_{label}"] = floor_via_index
                    row[f"rank_{label}"] = self.sampling_config_completed["logprobs"]
                    row[f"floored_{label}"] = True

            logprobs_data.append(row)

        return logprobs_data


# ======================================================================
# LLM_Generation dataclass
# ======================================================================


@dataclass
class LLM_Generation:
    """Single prompt/generation unit for LLM auditing.

    Note: NO ``__slots__`` so that old pickle files (created before the
    dataclass conversion) can still be unpickled safely.
    """

    # Constructor arguments (always provided at creation time)
    formatted_prompt: str
    token_pair: str
    dataset_idx: int

    # Fields populated after generation
    output: Optional[Any] = field(default=None, repr=False)
    output_text: str = ""
    scrapped_output: str = ""
    output_token_ids: Optional[list[int]] = None
    output_logprobs: Optional[list[dict[str, Any]]] = None
    fail: bool = False
    success: bool = False
    gen_counter: int = 0

    def increase_gen_counter(self, max_counter: int, is_valid: bool) -> None:
        """Increment counter and mark as failed if max retries exceeded."""
        self.gen_counter += 1
        if self.gen_counter >= max_counter and not is_valid:
            self.fail = True
            self.formatted_prompt = ""  # freeing memory

    # ------------------------------------------------------------------
    # Parquet serialization
    # ------------------------------------------------------------------

    def to_row_dict(self) -> dict[str, Any]:
        """Convert to a flat dict suitable for a Parquet row."""
        return {
            "formatted_prompt": self.formatted_prompt,
            "token_pair": self.token_pair,
            "dataset_idx": self.dataset_idx,
            "output_text": self.output_text,
            "scrapped_output": self.scrapped_output,
            "output_token_ids": json.dumps(self.output_token_ids) if self.output_token_ids is not None else None,
            "output_logprobs": json.dumps(self.output_logprobs) if self.output_logprobs is not None else None,
            "fail": self.fail,
            "success": self.success,
            "gen_counter": self.gen_counter,
        }

    @classmethod
    def from_row_dict(cls, row: dict[str, Any]) -> "LLM_Generation":
        """Reconstruct from a Parquet row dict."""
        token_ids_raw = row.get("output_token_ids")
        logprobs_raw = row.get("output_logprobs")
        return cls(
            formatted_prompt=row["formatted_prompt"],
            token_pair=row["token_pair"],
            dataset_idx=row["dataset_idx"],
            output_text=row.get("output_text", ""),
            scrapped_output=row.get("scrapped_output", ""),
            output_token_ids=json.loads(token_ids_raw) if token_ids_raw is not None else None,
            output_logprobs=json.loads(logprobs_raw) if logprobs_raw is not None else None,
            fail=row.get("fail", False),
            success=row.get("success", False),
            gen_counter=row.get("gen_counter", 0),
        )

    def __str__(self) -> str:
        return f"({self.dataset_idx}, {self.token_pair})"
