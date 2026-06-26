"""Substring composition analysis and cross-tokenizer statistics."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

from audit_llm.path_utils import get_repository_level_path
from audit_llm.Tokens_analysis.token_decoding import token_decoder
from audit_llm.Tokens_analysis.vocab_io import (
    get_models_by_tokenizer,
    load_tokenizer_vocabs,
)
from audit_llm.Tokens_analysis.vocab_viz import save_tokenizer_token_boxes


def can_help_compose_substring(
    complementary_substrings: List[str],
    necessary_substrings: List[str],
    s: str,
) -> bool:
    """Check if string s can help build one of the necessary_substrings.

    Tests:
    >>> necessary_substrings = ['moth', 'quiz']
    >>> complementary_substrings = [' ', ',', '\\n']
    >>> can_help_compose_substring(complementary_substrings, necessary_substrings, 'th, ')
    True
    >>> can_help_compose_substring(complementary_substrings, necessary_substrings, 't, ')
    False
    """
    comp_chars = set(ch for comp in complementary_substrings for ch in comp)

    # Strip complementary characters from the start
    i = 0
    while i < len(s) and s[i] in comp_chars:
        i += 1
    # Strip complementary characters from the end
    j = len(s)
    while j > i and s[j - 1] in comp_chars:
        j -= 1

    core = s[i:j]
    if not core or any(ch in comp_chars for ch in core):
        return False

    has_leading = i > 0
    has_trailing = j < len(s)

    for needed in necessary_substrings:
        # No comps at all: any substring match
        if not has_leading and not has_trailing:
            if core in needed:
                return True
        # Only trailing comps: must be a suffix (i.e., end part)
        elif not has_leading and has_trailing:
            if needed.endswith(core):
                return True
        # Only leading comps: must be a prefix (i.e., beginning part)
        elif has_leading and not has_trailing:
            if needed.startswith(core):
                return True
        # Both leading and trailing comps: must match the whole word
        else:
            if core == needed:
                return True

    return False


def can_be_decomposed_from_substring(
    complementary_substrings: list, necessary_substrings: list, string: str
) -> bool:
    """Return True iff string contains a necessary_substring and can be fully segmented from all substrings.

    Examples:
        complementary_substrings = ['0', '1'], necessary_substrings = ['1']
        '1001' -> True, '0000' -> False

        complementary_substrings = ['a', 'b', 'ab'], necessary_substrings = ['ab']
        'abab' -> True, 'aa' -> False, 'abc' -> False
    """
    all_substrings = list(set(complementary_substrings + necessary_substrings))
    if any([substr in string for substr in necessary_substrings]):
        # Sort by length so longer tokens get tried before shorter ones
        parts = sorted(all_substrings, key=len, reverse=True)
        pat = re.compile(r"^(?:" + "|".join(map(re.escape, parts)) + r")+$")
        return bool(pat.fullmatch(string))
    else:
        return False


def get_token_stats(
    tokenizer_to_models: Dict[str, List[str]],
    tokenizer_vocabs: Dict[str, Dict[str, int]],
    complementary_substrings: List[str],
    necessary_substrings: List[str],
) -> Dict[str, Any]:
    """Compute cross-tokenizer statistics for substring-matching tokens."""
    tokenizer_tokens: Dict[str, Any] = {}
    for name, models in tokenizer_to_models.items():
        vocab = tokenizer_vocabs[name]

        valid_tokens = [
            tok
            for tok in vocab.keys()
            if tok
            and can_be_decomposed_from_substring(complementary_substrings, necessary_substrings, token_decoder(tok))
        ]

        for nec_substring in necessary_substrings:
            if len(nec_substring) > 1 and nec_substring != "\n":
                valid_tokens += [
                    tok
                    for tok in vocab.keys()
                    if tok and can_help_compose_substring(complementary_substrings, [nec_substring], token_decoder(tok))
                ]

        tokenizer_tokens[name] = {"models": models, "matches": valid_tokens}

    # Prepare for cross computation
    set_map = {name: set(info["matches"]) for name, info in tokenizer_tokens.items()}
    names = list(set_map.keys())
    n = len(names)

    if n < 2:
        raise ValueError("At least two tokenizers required to compute stats")

    union_set = list(set().union(*set_map.values()))

    return {"tokenizer_tokens": tokenizer_tokens, "union": union_set}


def analyze_and_save_vocab_substrings(
    complementary_substrings: List[str],
    necessary_substrings: List[str],
    saving_path: Path,
    grouped_vocab_path: Path | None = None,
) -> Dict[str, Any]:
    """Analyze grouped tokenizer vocabularies to find tokens containing specified substrings."""
    repo_path = get_repository_level_path()
    grouped_root = (
        Path(grouped_vocab_path) if grouped_vocab_path else Path(repo_path) / "Productions" / "Grouped_Tokenizer_Vocabs"
    )
    assert grouped_root.exists(), f"Grouped vocab root {grouped_root} does not exist"

    tokenizer_to_models = get_models_by_tokenizer(grouped_vocab_path)
    logger.debug("tokenizer_to_models=%s", tokenizer_to_models)
    tokenizer_vocabs = load_tokenizer_vocabs(grouped_vocab_path)

    token_stats_dict = get_token_stats(
        tokenizer_to_models, tokenizer_vocabs, complementary_substrings, necessary_substrings
    )

    Path(saving_path).mkdir(parents=True, exist_ok=True)

    with open(Path(saving_path) / "token_stats.json", "w") as f:
        json.dump(token_stats_dict, f, indent=2)

    save_tokenizer_token_boxes(token_stats_dict, saving_path)

    return token_stats_dict
