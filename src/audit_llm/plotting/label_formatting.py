"""Variation-label formatting for heatmaps and LaTeX tables."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal

import numpy as np

from audit_llm.xp_tools.label_formatting import SP_IDX_TO_ICML_SP_IDX

QUANTIZATION_DISPLAY_MAP: dict[str, str] = {
    "bitsandbytes_int4": "int4",
    "bitsandbytes_int8": "int8",
    "fp8": "fp8",
}


# ---------------------------------------------------------------------------
# Shared parser
# ---------------------------------------------------------------------------

def _parse_variation_label(name: str) -> dict:
    """Parse a variation name into structured parts.

    Returns a dict with key ``'type'`` (one of ``'ablit'``, ``'temp_sp'``,
    ``'fallback'``) and, for ``'temp_sp'``, additional keys ``'temp_val'``,
    ``'sp_val'``, and ``'quant_prefix'`` (str or None).
    """
    if name == "ablit":
        return {"type": "ablit"}
    # Extract quantization prefix if present (e.g., "fp8_temp-..." or "bitsandbytes_int4_temp-...")
    quant_prefix = None
    core_name = name
    temp_split = name.split("_temp-", 1)
    if len(temp_split) == 2 and temp_split[0]:
        quant_prefix = temp_split[0]
        core_name = "temp-" + temp_split[1]
    match = re.search(r"temp-([\d.]+)_sp-(-?\d+)", core_name)
    if match:
        return {
            "type": "temp_sp",
            "temp_val": match.group(1),
            "sp_val": match.group(2),
            "quant_prefix": quant_prefix,
        }
    return {"type": "fallback"}


# ---------------------------------------------------------------------------
# Matplotlib-oriented formatter (newline separators)
# ---------------------------------------------------------------------------

def format_label_multiline(name: str) -> str:
    """Format a variation name for matplotlib (newline separators)."""
    parsed = _parse_variation_label(name)
    if parsed["type"] == "ablit":
        return "abli-\nterated"
    if parsed["type"] == "temp_sp":
        quant_prefix = parsed.get("quant_prefix")
        quant_line = QUANTIZATION_DISPLAY_MAP.get(quant_prefix, quant_prefix) + "\n" if quant_prefix else ""
        x_val, y_val = parsed["temp_val"], parsed["sp_val"]
        if y_val == "-1":
            return f"{quant_line}temp {x_val}\n(no sp)"
        return f"{quant_line}sp {SP_IDX_TO_ICML_SP_IDX.get(y_val, y_val)}\n(temp {x_val})"
    return name


# ---------------------------------------------------------------------------
# LaTeX-oriented formatter (tabular line breaks)
# ---------------------------------------------------------------------------

def format_two_line_header(name: str) -> str:
    """Format a variation name as a two-line LaTeX tabular block."""
    parsed = _parse_variation_label(name)
    if parsed["type"] == "ablit":
        content = r"abli- \\ terated"
    elif parsed["type"] == "temp_sp":
        quant_prefix = parsed.get("quant_prefix")
        quant_line = QUANTIZATION_DISPLAY_MAP.get(quant_prefix, quant_prefix) + r" \\ " if quant_prefix else ""
        x_val, y_val = parsed["temp_val"], parsed["sp_val"]
        if y_val == "-1":
            content = f"{quant_line}temp {x_val} \\\\ (no sp)"
        else:
            content = f"{quant_line}sp {y_val} \\\\ (temp {x_val})"
    else:
        content = name
    return f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{content}\\end{{tabular}}"


# ---------------------------------------------------------------------------
# Numeric formatting
# ---------------------------------------------------------------------------

def format_value(value: float, compact: bool = True) -> str:
    """Format a float as a percentage string."""
    if np.isnan(value):
        return "NaN"
    d = Decimal(str(value * 100)).quantize(
        Decimal("1") if compact else Decimal("1.00"),
        rounding=ROUND_HALF_UP,
    )
    return str(d)
