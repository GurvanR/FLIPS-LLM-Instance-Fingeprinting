"""Token intersection computation, random sampling from cross-tokenizer intersections."""

import json
import random
import string
from pathlib import Path
from typing import Dict, List, Optional, Set

from audit_llm.path_utils import get_repository_level_path
from audit_llm.Tokens_analysis.token_decoding import (
    token_decoder,
    universalize_token_vocab,
)
from audit_llm.Tokens_analysis.vocab_io import load_tokenizer_vocabs


def save_intersection_of_tokenizers(save_path: Optional[str] = None) -> set:
    """Compute and save the intersection of decoded tokens across all tokenizer vocabs."""
    model_to_vocab = load_tokenizer_vocabs()

    # universalize vocabs
    for model, vocab in model_to_vocab.items():
        model_to_vocab[model] = universalize_token_vocab(vocab)

    # get the intersection of decoded tokens over all vocabs
    vocab_sets: Dict[str, Set[str]] = {model: set(vocab.keys()) for model, vocab in model_to_vocab.items()}

    all_models = list(vocab_sets.values())
    if not all_models:
        intersection: set = set()
    else:
        intersection = set.intersection(*all_models)

    # save to file
    repo_path = get_repository_level_path()
    intersection_vocab_path = Path(repo_path) / "Productions" / "Intersection_vocab"
    intersection_vocab_path.mkdir(parents=True, exist_ok=True)
    intersection_vocab_file_path = intersection_vocab_path / f"Intersection_of_{len(model_to_vocab)}_models.json"

    with open(intersection_vocab_file_path, "w", encoding="utf-8") as f:
        json.dump(sorted(intersection), f, indent=2, ensure_ascii=False)

    return intersection


def get_random_token_pairs_from_intersection(
    nb_of_uplets: int, uplet_size: int = 2, monochar: bool = False, seed: int = 42
) -> list:
    """Generate nb_of_uplets random token tuples from the intersection file."""
    random.seed(seed)
    random_instruction_prompts = [
        get_random_token_from_intersection(nb_of_token=uplet_size, monochar=monochar) for k in range(nb_of_uplets)
    ]
    return random_instruction_prompts


def get_random_token_from_intersection(
    nb_of_token: int,
    allowed_char_types: List[str] | None = None,
    monochar: bool = False,
    max_length: int | None = None,
) -> List[str]:
    """Return a random sample of tokens from the intersection file, with filtering."""
    if allowed_char_types is None:
        allowed_char_types = ["alpha", "digit"]

    repo_path = get_repository_level_path()
    intersection_vocab_path = Path(repo_path) / "Productions" / "Intersection_vocab"

    files = list(intersection_vocab_path.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {intersection_vocab_path}")
    chosen_file = files[0]

    with open(chosen_file, "r", encoding="utf-8") as f:
        intersection = json.load(f)

    char_sets = {
        "alpha": set(string.ascii_letters),
        "lower": set(string.ascii_lowercase),
        "upper": set(string.ascii_uppercase),
        "digit": set(string.digits),
        "alnum": set(string.ascii_letters + string.digits),
        "punct": set(string.punctuation),
        "whitespace": set(string.whitespace),
    }

    allowed_chars: set = set()
    for ctype in allowed_char_types:
        if ctype not in char_sets:
            raise ValueError(f"Unsupported character type '{ctype}'. " f"Supported types: {list(char_sets.keys())}")
        allowed_chars |= char_sets[ctype]

    def valid_whitespace_rule(token: str) -> bool:
        if token.startswith(" "):
            return len(token) > 1 and token[1] != " " and token.count(" ") == 1 and token[0] == " "
        return " " not in token

    def valid_length_rule(token: str) -> bool:
        if monochar:
            return len(token) == 1
        return len(token) >= 1

    def valid_max_length_rule(token: str) -> bool:
        return max_length is None or len(token) <= max_length

    filtered = [
        token
        for token in intersection
        if all(ch in allowed_chars for ch in token)
        and valid_whitespace_rule(token)
        and valid_length_rule(token)
        and valid_max_length_rule(token)
    ]

    if nb_of_token > len(filtered):
        raise ValueError(f"Requested {nb_of_token} tokens, but only {len(filtered)} available after filtering.")

    return random.sample(filtered, nb_of_token)
