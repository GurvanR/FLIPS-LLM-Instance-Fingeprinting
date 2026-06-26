"""LLM_Classes — LLM auditing backends, config, parsing, and analysis.

Public API re-exports for convenience.
Lazy-loaded to avoid importing GPU-dependent packages (torch, transformers,
vllm) at package-load time.  This allows the post-processing environment
to ``import audit_llm`` without those packages installed.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from audit_llm.LLM_Classes.General_LLM_Class import (
        LLM_for_audit,
        LLM_Generation,
    )
    from audit_llm.LLM_Classes.generation_parser import (
        parsing_generations,
    )
    from audit_llm.LLM_Classes.inference_runner import (
        multi_model_infer,
    )
    from audit_llm.LLM_Classes.run_config import make_run_config, RunConfigDict
    from audit_llm.LLM_Classes.vLLM_Classes import vLLM


__all__ = [
    "LLM_Generation",
    "LLM_for_audit",
    "make_run_config",
    "multi_model_infer",
    "parsing_generations",
    "RunConfigDict",
    "vLLM",
]

# Map of public names to (module_path, attribute_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "LLM_Generation": ("audit_llm.LLM_Classes.General_LLM_Class", "LLM_Generation"),
    "LLM_for_audit": ("audit_llm.LLM_Classes.General_LLM_Class", "LLM_for_audit"),
    "make_run_config": ("audit_llm.LLM_Classes.run_config", "make_run_config"),
    "multi_model_infer": ("audit_llm.LLM_Classes.inference_runner", "multi_model_infer"),
    "parsing_generations": ("audit_llm.LLM_Classes.generation_parser", "parsing_generations"),
    "RunConfigDict": ("audit_llm.LLM_Classes.run_config", "RunConfigDict"),
    "vLLM": ("audit_llm.LLM_Classes.vLLM_Classes", "vLLM"),
}


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
