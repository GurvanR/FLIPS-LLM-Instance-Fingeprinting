"""Static lookup for model-name → tokenizer-class mapping.

The transformers tokenizer classes are imported lazily to avoid requiring
the ``transformers`` package when only post-processing modules are used.
If ``transformers`` is not installed, the map is empty and callers should
fall back to their own import.
"""

from audit_llm.models_management.model_names import GPT, T5, LLama

try:
    from transformers import GPT2Tokenizer, LlamaTokenizer, T5Tokenizer  # pylint: disable=import-error

    _llama_tokenizer_map = {model_name: LlamaTokenizer for model_name in LLama}
    _t5_tokenizer_map = {model_name: T5Tokenizer for model_name in T5}
    _gpt2_tokenizer_map = {model_name: GPT2Tokenizer for model_name in GPT}

    model_tokenizer_map: dict[str, type] = {
        **_t5_tokenizer_map,
        **_llama_tokenizer_map,
        **_gpt2_tokenizer_map,
    }
except ImportError:
    # transformers not installed — map is empty; callers use AutoTokenizer fallback
    model_tokenizer_map = {}
