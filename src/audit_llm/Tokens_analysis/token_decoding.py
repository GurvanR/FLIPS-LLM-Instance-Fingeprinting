"""Token string decoding across tokenizer families and vocab universalization."""

from typing import Dict


def token_decoder(token: str) -> str:
    """Convert a raw tokenizer token to its text representation.

    Handles special characters from GPT, LLaMA, BERT, T5, Qwen, BLOOM,
    RoBERTa, XLM-RoBERTa, CodeGen, TikToken, SentencePiece, and others.

    Examples:
        'Ġlight' -> ' light'
        'Ċ' -> '\\n'
        'Ĥ' -> '\\t'
        '▁word' -> ' word'
        '##ing' -> 'ing'
    """
    if not token or not isinstance(token, str):
        return ""

    # Handle special vocabulary tokens
    special_tokens = {
        "<s>": "",  # Start of sequence
        "</s>": "",  # End of sequence
        "<pad>": "",  # Padding
        "<unk>": "[UNK]",  # Unknown token
        "<mask>": "[MASK]",  # Mask token
        "<|endoftext|>": "",  # GPT document separator
        "<|im_start|>": "",  # Llama message start
        "<|im_end|>": "",  # Llama message end
        "<eos>": "",  # End of string
        "<bos>": "",  # Beginning of string
    }

    if token in special_tokens:
        return special_tokens[token]

    # 1. Transformer spaces/whitespace characters
    # GPT/LLaMA/Qwen/BLOOM style: Ġ (G with dot)
    if token.startswith("Ġ"):
        return " " + token[1:]

    # SentencePiece style: ▁ (underscore symbol)
    elif token.startswith("▁"):
        return " " + token[1:]

    # T5 style: _ (literal underscore)
    elif token.startswith("_") and token != "_":
        return " " + token[1:]

    # 2. Newline characters
    # Common newline: Ċ (C with dot)
    elif token.startswith("Ċ"):
        return "\n" + token[1:] if len(token) > 1 else "\n"

    # Alternative newline: Ń (N with acute accent)
    elif token.startswith("Ń"):
        return "\n" + token[1:] if len(token) > 1 else "\n"

    # 3. Tab characters
    # Common tab: Ĥ (H with circumflex)
    elif token.startswith("Ĥ"):
        return "\t" + token[1:] if len(token) > 1 else "\t"

    # 4. BERT-style subword tokens
    elif token.startswith("##"):
        return token[2:]

    # 5. Multiple whitespace markers
    elif token == "ĊĊ":
        return "\n\n"
    elif token == "ĠĠ":
        return "  "  # Double space
    elif token == "ĤĤ":
        return "\t\t"  # Double tab

    # 6. Escaped characters in some tokenizers
    elif token.startswith("\\"):
        escape_chars = {"\\n": "\n", "\\t": "\t", "\\r": "\r", '\\"': '"', "\\'": "'", "\\\\": "\\"}
        if token in escape_chars:
            return escape_chars[token]

    # 7. Byte tokens from byte-level BPE (for non-UTF8 characters)
    elif token.startswith("<0x") and token.endswith(">"):
        try:
            hex_val = token[3:-1]
            return chr(int(hex_val, 16))
        except (ValueError, OverflowError):
            pass  # If conversion fails, just return the token as is

    # 8. HTML/XML entity encoding used by some tokenizers
    elif token.startswith("&") and token.endswith(";"):
        html_entities = {"&lt;": "<", "&gt;": ">", "&amp;": "&", "&quot;": '"', "&apos;": "'", "&nbsp;": " "}
        if token in html_entities:
            return html_entities[token]

    # 9. Chinese-specific tokenizers (like ByT5)
    elif token == "□":
        return "[UNK]"

    # 10. CodeGen and code-specific tokenizers
    elif token == "<EOL>":
        return "\n"

    # Default: return the token as is for normal tokens
    return token


def universalize_token_vocab(vocab: Dict[str, int]) -> Dict[str, int]:
    """Apply token_decoder to all vocab token keys."""
    return {token_decoder(token): token_id for token, token_id in vocab.items()}
