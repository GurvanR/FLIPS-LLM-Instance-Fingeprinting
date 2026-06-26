"""Token box visualization: multi-panel matplotlib figure showing decoded tokens per tokenizer group."""

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt

from audit_llm.Tokens_analysis.token_decoding import token_decoder


def save_tokenizer_token_boxes(
    result: Dict[str, Any],
    save_fig_path,
    top_n: int = None,  # type: ignore
    sort_by: str = "cross_tokenizer_freq",
) -> None:
    """Create a visualization with independent boxes for each tokenizer group.

    Parameters:
    - result: The result dictionary from get_token_stats
    - top_n: Number of top tokens to display per tokenizer. If None, shows all.
    - sort_by: Sorting method ('cross_tokenizer_freq' or 'token_length')
    """
    import colorsys

    import matplotlib.colors as mcolors
    import matplotlib.patches as patches
    import numpy as np
    import pandas as pd
    from matplotlib import cm

    # Extract tokenizer matches
    tokenizer_tokens = result["tokenizer_tokens"]

    # Decode all tokens for each tokenizer and group by decoded tokens
    decoded_tokenizer_data = {}

    for name, info in tokenizer_tokens.items():
        decoded_tokens = set()
        for token in info["matches"]:
            try:
                decoded_token = token_decoder(token)
                if decoded_token != "":
                    decoded_tokens.add(decoded_token)
            except:
                if token != "":
                    decoded_tokens.add(token)

        decoded_tokenizer_data[name] = {"decoded_tokens": decoded_tokens, "models": info["models"]}

    # Group tokenizers with identical sets of decoded tokens
    tokenizer_groups = {}
    processed_tokenizers = set()

    for name, data in decoded_tokenizer_data.items():
        if name in processed_tokenizers:
            continue

        group_tokenizers = [name]
        group_models = data["models"].copy()

        for other_name, other_data in decoded_tokenizer_data.items():
            if (
                other_name != name
                and other_name not in processed_tokenizers
                and data["decoded_tokens"] == other_data["decoded_tokens"]
            ):
                group_tokenizers.append(other_name)
                group_models.extend(other_data["models"])
                processed_tokenizers.add(other_name)

        if len(group_tokenizers) == 1:
            group_name = name
        else:
            group_name = " + ".join(sorted(group_tokenizers))

        tokenizer_groups[group_name] = {
            "decoded_tokens": data["decoded_tokens"],
            "models": list(set(group_models)),
            "tokenizers": group_tokenizers,
        }
        processed_tokenizers.add(name)

    # Compute token cross-tokenizer frequencies based on decoded tokens
    all_decoded_tokens = set().union(*[info["decoded_tokens"] for info in tokenizer_groups.values()])

    token_data = []
    for decoded_token in all_decoded_tokens:
        groups_with_token = [name for name, info in tokenizer_groups.items() if decoded_token in info["decoded_tokens"]]
        cross_freq = len(groups_with_token)

        token_data.append(
            {
                "token": decoded_token,
                "cross_tokenizer_freq": cross_freq,
                "token_length": len(decoded_token),
                "is_empty": decoded_token == "",
            }
        )

    df = pd.DataFrame(token_data)
    df = df[~df["is_empty"]]

    if sort_by == "cross_tokenizer_freq":
        df_sorted = df.sort_values("cross_tokenizer_freq", ascending=False)
    elif sort_by == "token_length":
        df_sorted = df.sort_values("token_length", ascending=False)
    else:
        raise ValueError("sort_by must be 'cross_tokenizer_freq' or 'token_length'")

    # Create color map for token frequencies
    max_freq = df["cross_tokenizer_freq"].max()
    min_freq = df["cross_tokenizer_freq"].min()
    unique_frequencies = sorted(df["cross_tokenizer_freq"].unique())

    standard_colors = ["#FF6B6B", "#FFA500", "#32CD32", "#9370DB", "#20B2AA", "#FF69B4"]

    freq_to_color = {}
    for i, freq in enumerate(unique_frequencies):
        freq_to_color[freq] = standard_colors[i % len(standard_colors)]

    def get_token_color(frequency):
        return freq_to_color[frequency]

    # Calculate adaptive subplot dimensions
    num_groups = len(tokenizer_groups)

    group_info_list = []
    for group_name, group_info in tokenizer_groups.items():
        tokens_in_group = group_info["decoded_tokens"]
        tokens_df = df[df["token"].isin(tokens_in_group)]

        if sort_by == "cross_tokenizer_freq":
            tokens_sorted = tokens_df.sort_values("cross_tokenizer_freq", ascending=False)
        else:
            tokens_sorted = tokens_df.sort_values("token_length", ascending=False)

        tokens_to_display = tokens_sorted if top_n is None else tokens_sorted.head(top_n)
        n_tokens = len(tokens_to_display)

        avg_token_length = tokens_to_display["token_length"].mean() if n_tokens > 0 else 1
        max_token_length = tokens_to_display["token_length"].max() if n_tokens > 0 else 1

        group_info_list.append(
            {
                "name": group_name,
                "info": group_info,
                "n_tokens": max(1, n_tokens),
                "avg_token_length": avg_token_length,
                "max_token_length": max_token_length,
            }
        )

    # Determine layout
    cols = min(3, num_groups)
    rows = (num_groups + cols - 1) // cols

    # Calculate width and height ratios
    width_ratios = []
    height_ratios = []

    for c in range(cols):
        max_width_factor = 1
        for r in range(rows):
            idx = r * cols + c
            if idx < len(group_info_list):
                group = group_info_list[idx]
                width_factor = 1 + (group["avg_token_length"] * 0.05) + (group["max_token_length"] * 0.02)
                max_width_factor = max(max_width_factor, width_factor)
        width_ratios.append(max_width_factor)

    for r in range(rows):
        max_height_factor = 1
        for c in range(cols):
            idx = r * cols + c
            if idx < len(group_info_list):
                group = group_info_list[idx]
                height_factor = max(1, group["n_tokens"] / 10)
                max_height_factor = max(max_height_factor, height_factor)
        height_ratios.append(max_height_factor)

    # Calculate figure size based on content
    total_tokens = sum([g["n_tokens"] for g in group_info_list])
    avg_tokens_per_group = total_tokens / num_groups if num_groups > 0 else 1
    avg_token_length = sum([g["avg_token_length"] for g in group_info_list]) / num_groups if num_groups > 0 else 1

    base_width_per_col = max(4, min(10, 3 + avg_token_length * 0.3))
    base_height_per_row = max(3, min(12, 2 + avg_tokens_per_group * 0.15))

    fig_width = base_width_per_col * cols * max(width_ratios) / len(width_ratios)
    fig_height = base_height_per_row * rows * max(height_ratios) / len(height_ratios) + 1.5

    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = fig.add_gridspec(rows, cols, height_ratios=height_ratios, width_ratios=width_ratios, hspace=0.3, wspace=0.3)

    # Plot each tokenizer group's tokens
    for idx, group_data in enumerate(group_info_list):
        row = idx // cols
        col = idx % cols
        ax = fig.add_subplot(gs[row, col])

        group_name = group_data["name"]
        group_info = group_data["info"]

        tokens_in_group = group_info["decoded_tokens"]
        tokens_df = df[df["token"].isin(tokens_in_group)]

        if sort_by == "cross_tokenizer_freq":
            tokens_sorted = tokens_df.sort_values("cross_tokenizer_freq", ascending=False)
        else:
            tokens_sorted = tokens_df.sort_values("token_length", ascending=False)

        tokens_to_display = tokens_sorted if top_n is None else tokens_sorted.head(top_n)
        n_tokens = len(tokens_to_display)
        n_models = len(group_info["models"])

        ax.set_facecolor("white")

        border = patches.Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            fill=False,
            edgecolor="darkgray",
            linewidth=2,
            linestyle="-",
            zorder=10,
        )
        ax.add_patch(border)

        ax.text(
            0.5,
            0.95,
            f"Group {idx+1}: {n_models} models",
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=10,
            fontweight="bold",
            transform=ax.transAxes,
        )

        ax.set_axis_off()

        if n_tokens == 0:
            ax.text(
                0.5,
                0.5,
                "No matching tokens",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=10,
                fontstyle="italic",
                transform=ax.transAxes,
            )
            continue

        # Adaptive column layout
        avg_token_length = group_data["avg_token_length"]

        if n_tokens <= 8:
            columns = 1
        elif n_tokens <= 24 or avg_token_length > 8:
            columns = 2
        elif n_tokens <= 60 or avg_token_length > 15:
            columns = 3
        else:
            columns = 4

        tokens_per_col = (n_tokens + columns - 1) // columns

        usable_height = 0.85
        start_position_y = 0.88
        token_spacing = usable_height / max(tokens_per_col + 1, 8)

        for i, (_, token_row) in enumerate(tokens_to_display.iterrows()):
            col_idx = i // tokens_per_col
            pos_in_col = i % tokens_per_col

            token = token_row["token"]
            cross_freq = token_row["cross_tokenizer_freq"]

            token_color = get_token_color(cross_freq)

            col_width = 0.95 / columns
            x_position = 0.025 + (col_idx * col_width)

            y_position = start_position_y - ((pos_in_col + 1) * token_spacing)

            if avg_token_length > 12 or n_tokens > 80:
                font_size = 6
            elif n_tokens <= 15 and avg_token_length <= 6:
                font_size = 10
            elif n_tokens <= 30:
                font_size = 8
            else:
                font_size = 7

            ax.text(
                x_position,
                y_position,
                token,
                fontsize=font_size,
                color="black",
                fontweight="normal",
                bbox=dict(
                    facecolor=token_color,
                    alpha=0.7,
                    edgecolor="none",
                    boxstyle="round,pad=0.2",
                    mutation_scale=0.8,
                ),
                transform=ax.transAxes,
            )

    # Create discrete legend for frequencies
    fig.text(
        0.5,
        0.02,
        "Cross-tokenizer frequency shows how many tokenizer groups contain this decoded token",
        ha="center",
        fontsize=9,
        style="italic",
    )

    n_freqs = len(unique_frequencies)
    label_spacing = min(0.08, 0.4 / n_freqs)
    total_width = label_spacing * (n_freqs - 1)
    x_start = 0.5 - total_width / 2
    legend_y = 0.05

    for i, freq in enumerate(unique_frequencies):
        color = freq_to_color[freq]
        x_pos = x_start + i * label_spacing
        fig.text(
            x_pos,
            legend_y,
            str(freq),
            ha="center",
            fontsize=12,
            fontweight="bold",
            color=color,
            bbox=dict(facecolor=color, alpha=0.3, boxstyle="round,pad=0.3"),
        )

    plt.tight_layout(rect=[0, 0.08, 1, 1])  # type:ignore

    token_boxes_path = Path(save_fig_path)
    token_boxes_path.mkdir(parents=True, exist_ok=True)
    token_boxes_image_path = Path(token_boxes_path) / "vocab_token_boxes.pdf"
    plt.savefig(token_boxes_image_path, bbox_inches="tight")
    plt.close()
    box_info = {tuple(sorted(info["decoded_tokens"])): info["models"] for info in tokenizer_groups.values()}

    json_save_path = token_boxes_path / "vocab_token_boxes.json"
    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in box_info.items()}, f, indent=2)
