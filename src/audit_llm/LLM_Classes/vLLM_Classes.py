"""vLLM backend for LLM auditing — the only active inference backend."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import multiprocessing

from overrides import overrides

from audit_llm.Bits_Generation.parsing_bits_tools import (
    validate_seq,
)
from audit_llm.LLM_Classes.General_LLM_Class import (
    LLM_for_audit,
    LLM_Generation,
)
from audit_llm.LLM_Classes.run_config import RunConfigDict
from audit_llm.system_utils import vllm_version_import_manager


# Force spawn method (redundant with env var but safer)
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

QUANTIZATION_VLLM_PARAMS: dict[str, dict] = {
    "no_quantized": {},
    "fp8": {"quantization": "fp8"},
    # vLLM bitsandbytes support is NF4 (4-bit) only; load_in_4bit/load_in_8bit are HF-only params
    "bitsandbytes_int4": {"quantization": "bitsandbytes"},
}


class vLLM(LLM_for_audit):
    """vLLM inference backend with dynamic batching and retry logic."""

    def __init__(
        self,
        model_name: str,
        run_config: RunConfigDict,
        run_path: str,
        model_path: str | None = None,
    ) -> None:
        super().__init__(model_name, run_path, run_config, model_path)

        model_config = run_config["vllm_model_config"]
        model_only_config = dict(model_config["model_only_config"])
        sampling_config = model_config["sampling_config"]

        # Apply per-model quantization params
        quant_key = run_config.get("quantization_map", {}).get(model_name, "no_quantized")
        quant_params = QUANTIZATION_VLLM_PARAMS.get(quant_key, {})
        if quant_params:
            logger.info("Applying quantization '%s': %s", quant_key, quant_params)
            model_only_config.update(quant_params)

        from vllm import LLM  # pylint: disable=import-outside-toplevel

        # Version-dependent key name for max context
        max_context = vllm_version_import_manager("max_context")
        default_model_config = {
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.5,
            "trust_remote_code": False,
            max_context: 2048,
            "max_model_len": 1024,
            "tensor_parallel_size": 1,
        }

        default_sampling_config = {
            "max_tokens": 16,
            "temperature": [0],
            "top_k": 1,
            "logprobs": None,
        }

        model_config_completed = self._build_config(default_model_config, model_only_config, model=self.model)
        self.sampling_config_completed = self._build_config(default_sampling_config, sampling_config)

        self.max_tokens = self.sampling_config_completed["max_tokens"]

        logprobs = self.sampling_config_completed["logprobs"]
        self.llm = LLM(
            **model_config_completed,
            max_logprobs=0 if logprobs is None else logprobs,
        )

        self.model_type = "vllm"

    @overrides
    def _generate(self, sampling_key: dict) -> None:  # type: ignore[override]
        """Generate outputs using vLLM with dynamic batching and retries."""
        from vllm import SamplingParams  # pylint: disable=import-outside-toplevel
        if not isinstance(self.generations[0], LLM_Generation):
            raise NotImplementedError("Not implemented in vLLM class.")

        sampling_params = {
            **self.sampling_config_completed,
            **sampling_key,
        }
        gen_remaining = [gen for gen in self.generations if (not gen.success and not gen.fail)]
        total_gen_remaining = len(gen_remaining)
        current_index = 0
        completed_generations = 0

        buffer: list[LLM_Generation] = []

        # --- MAIN LOOP ---
        while current_index < total_gen_remaining or len(buffer) > 0:
            # Fill buffer up to dyn_checking_batch_size
            while len(buffer) < self.dyn_checking_batch_size and current_index < total_gen_remaining:
                buffer.append(gen_remaining[current_index])
                current_index += 1
            logger.debug("before inferences: len(buffer)=%d", len(buffer))

            if not buffer:
                break

            formatted_prompts_of_batch: list[str] = [llm_gen.formatted_prompt for llm_gen in buffer]

            batch_outputs = self.llm.generate(formatted_prompts_of_batch, SamplingParams(**sampling_params))

            for output, llm_gen in zip(batch_outputs, buffer):
                is_valid = validate_seq(
                    llm_gen,
                    output,
                    min_seq_length=self.min_seq_length,
                    mode="vllm",
                )
                llm_gen.increase_gen_counter(self.max_gen_counter, is_valid)
                if is_valid:
                    self.store_completion_elements(llm_gen, output)
                    llm_gen.success = True
                    completed_generations += 1
                elif llm_gen.fail:
                    # Keep last failed output in case we want to lower
                    # min_seq_length later.
                    self.store_completion_elements(llm_gen, output)

            # Keep only generations that haven't finished
            buffer = [p for p in buffer if (not p.success and not p.fail)]
            logger.debug("after inferences: len(buffer)=%d", len(buffer))
            if completed_generations >= self.max_inference_checkpoint_batch_size:
                self._save_generations()
                completed_generations = 0

        # Save final terminated dataset
        self._save_generations(dataset_terminated=True)
