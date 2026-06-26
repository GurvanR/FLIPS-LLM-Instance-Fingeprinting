# SPDX-FileCopyrightText: 2024 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Dictionary, list, and DataFrame transformation utilities."""

import re
from collections.abc import Callable
from typing import Any, Optional, Union

import numpy as np
import polars as pl


def sort_by_key(L: dict[int, Any]) -> list[Any]:
    """Sort the values of a dictionary by its keys."""
    return [value for _key, value in sorted(L.items())]


def str_tuple_to_int_list(s: str) -> list[int]:
    """Extract all integers from a string via regex."""
    numbers = re.findall(r"\d+", s)
    return [int(num) for num in numbers]


def preprocess_token_ids_col_of_answers(df: pl.DataFrame) -> pl.DataFrame:
    """Convert Token_IDs column to List[Int32] from string or List[Int64]."""
    token_dtype = df.schema["Token_IDs"]

    # CSV case -> string
    if token_dtype == pl.Utf8:
        return df.with_columns(
            pl.col("Token_IDs").map_elements(str_tuple_to_int_list, return_dtype=pl.List(pl.Int32)).alias("Token_IDs")
        )

    # Parquet case -> already a list
    if isinstance(token_dtype, pl.List):
        return df.with_columns(pl.col("Token_IDs").cast(pl.List(pl.Int32)).alias("Token_IDs"))

    raise TypeError(f"Unsupported Token_IDs dtype: {token_dtype}")


def mean_of_second_elements(
    sublist: list[tuple[Any, float]],
) -> Union[float, np.floating[Any]]:
    """Return the mean of second elements in a list of tuples."""
    return np.mean([score for _, score in sublist])


def revert_dictionary(my_dict: dict) -> dict:
    """Swap keys and values of a dictionary."""
    return {value: key for key, value in my_dict.items()}


def merge_dict_list(d1: dict[Any, list[Any]], d2: dict[Any, list[Any]]) -> dict:
    """Merge two dicts by extending list values for each key."""
    merged_dict = {}
    for key in set(d1.keys()).union(d2.keys()):
        merged_dict[key] = d1.get(key, []) + d2.get(key, [])
    return merged_dict


def convert_keys_at_depth(
    obj: Any,
    target_depth: int,
    key_type: type,
    current_depth: int = 0,
) -> Any:
    """Recursively convert dict keys at a specific nesting depth."""
    if not isinstance(obj, dict):
        return obj

    new_obj: dict = {}
    for k, v in obj.items():
        try:
            new_key = key_type(k) if current_depth == target_depth else k
        except (ValueError, TypeError):
            new_key = k

        if isinstance(v, dict):
            v = convert_keys_at_depth(v, target_depth, key_type, current_depth + 1)

        new_obj[new_key] = v

    return new_obj


def nested_loop(
    data: dict,
    main_action: Callable[[dict], None],
    keys: Optional[list[str]] = None,
    current: Optional[dict] = None,
) -> None:
    """Generate cartesian product of dict values, calling main_action for each."""
    if keys is None:
        keys = list(data.keys())
    if current is None:
        current = {}

    if not keys:
        main_action(current)
        return

    key = keys[0]
    for value in data[key]:
        nested_loop(data, main_action, keys[1:], {**current, key: value})
