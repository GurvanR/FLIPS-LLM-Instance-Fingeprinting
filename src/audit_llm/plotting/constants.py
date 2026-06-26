"""Plotting constants — color palettes and model shortlists."""

COLORBLIND_COLORS: list[str] = [
    "#377eb8", "#ff7f00", "#4daf4a", "#f781bf",
    "#a65628", "#984ea3", "#999999", "#e41a1c", "#dede00",
]

COLOR_DELTA_TP_MAP: dict[str, str] = {
    tp_group: COLORBLIND_COLORS[k]
    for k, tp_group in enumerate(["Monochar", "0-1"])
} | {"FLiPS": "red"}

SHORTLIST_OF_LLMS: list[str] = [
    "CohereForAI/c4ai-command-r-plus",
    "CohereLabs/aya-23-35B",
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    "Qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "google/gemma-2-27b-it",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "microsoft/Phi-3-medium-128k-instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2-7B-Instruct",
    "abacusai/Smaug-Llama-3-70B-Instruct",
]

TINYLIST_OF_LLMS: list[str] = [
    "CohereForAI/c4ai-command-r-plus",
    "google/gemma-2-27b-it",
    "Qwen/Qwen2-7B-Instruct",
]
