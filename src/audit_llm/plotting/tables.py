"""Table generators — LaTeX and Markdown accuracy tables."""

from __future__ import annotations

from typing import Any

import numpy as np

from audit_llm.plotting.label_formatting import format_two_line_header, format_value
from audit_llm.xp_tools.model_filtering import truncate_model_name


# ---------------------------------------------------------------------------
# 1×n tables (one row per model, one column per batch size)
# ---------------------------------------------------------------------------

def make_accuracy_table_latex(
    bs_accuracies: dict,
    models_idx: dict,
    averages: dict,
    caption: str = "",
    label: str = "",
    bs_accuracies_extra: dict | None = None,
    reverted_models_idx_extra: dict | None = None,
    averages_extra: dict | None = None,
) -> str:
    """Generate a LaTeX table of per-class accuracies across batch sizes."""
    batch_sizes = sorted(bs_accuracies.keys())
    has_extra = bs_accuracies_extra is not None

    if has_extra:
        col_spec = "l" + "".join(["r@{\\hspace{4pt}}l@{\\hspace{8pt}}r@{\\hspace{4pt}}l" for _ in batch_sizes])
    else:
        col_spec = "l" + "".join(["r@{\\hspace{4pt}}l" for _ in batch_sizes])

    lines = [r"\begin{table*}[t]", r"\centering", f"\\begin{{tabular}}{{{col_spec}}}", r"\toprule"]

    # Header Logic
    header = "LLM"
    for bs in batch_sizes:
        bs_label = "Single Query" if bs == 1 else f"{bs}-queries"
        header += f" & \\multicolumn{{{4 if has_extra else 2}}}{{c}}{{{bs_label}}}"
    header += r" \\"

    lines.append(header)
    if has_extra:
        subheader = ""
        for bs in batch_sizes:
            subheader += " & \\multicolumn{2}{c}{Main} & \\multicolumn{2}{c}{Extra}"
        lines.append(subheader + r" \\")

    lines.append(r"\midrule")

    # Per-class rows
    for class_idx, class_label in models_idx.items():
        row = class_label
        for bs in batch_sizes:
            # Main
            m, s = bs_accuracies[bs]["means"][class_idx], bs_accuracies[bs]["CI"][class_idx]
            if np.isnan(m):
                row += " & NaN & "
            else:
                row += f" & \\textbf{{{m*100:.2f}\\%}} & {{\\color{{gray}}($\\pm{100*s:.2f}\\%$)}}"

            # Extra
            if has_extra:
                if bs in bs_accuracies_extra:
                    idx_e = int(reverted_models_idx_extra[class_label]) if reverted_models_idx_extra else class_idx
                    me, se = bs_accuracies_extra[bs]["means"][idx_e], bs_accuracies_extra[bs]["CI"][idx_e]
                    if np.isnan(me):
                        row += " & NaN & "
                    else:
                        row += f" & \\textbf{{{me*100:.2f}\\%}} & {{\\color{{gray}}($\\pm{100*se:.2f}\\%$)}}"
                else:
                    row += " & - & "
        lines.append(row + r" \\")

    # Average row using pre-calculated values
    lines.append(r"\midrule")
    avg_row = "Average"
    for bs in batch_sizes:
        # Main Average
        am, asig = averages[bs]["mean"], averages[bs]["std"]
        avg_row += f" & \\textbf{{{am*100:.2f}\\%}} & {{\\color{{gray}}($\\pm{asig*100:.2f}\\%$)}}"

        # Extra Average
        if has_extra:
            if averages_extra and bs in averages_extra:
                aem, aesig = averages_extra[bs]["mean"], averages_extra[bs]["std"]
                avg_row += f" & \\textbf{{{aem*100:.2f}\\%}} & {{\\color{{gray}}($\\pm{aesig*100:.2f}\\%$)}}"
            else:
                avg_row += " & - & "

    lines.append(avg_row + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])

    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def make_accuracy_table_markdown(
    bs_accuracies: dict,
    models_idx: dict,
    averages: dict,
    bs_accuracies_extra: dict | None = None,
    reverted_models_idx_extra: dict | None = None,
    averages_extra: dict | None = None,
) -> str:
    """Generate a Markdown table of per-class accuracies across batch sizes."""
    batch_sizes = sorted(bs_accuracies.keys())
    has_extra = bs_accuracies_extra is not None
    lines = []

    # 1. Header Row
    header_cells = ["LLM"]
    for bs in batch_sizes:
        bs_label = "Single Query" if bs == 1 else f"{bs}-queries"
        if has_extra:
            header_cells.extend([f"{bs_label} (Main)", f"{bs_label} (Extra)"])
        else:
            header_cells.append(bs_label)
    lines.append("| " + " | ".join(header_cells) + " |")

    # 2. Separator Row
    align_cells = [":---"] + [":---:"] * (len(header_cells) - 1)
    lines.append("| " + " | ".join(align_cells) + " |")

    # 3. Data Rows (Per-model)
    for class_idx, class_label in models_idx.items():
        row_cells = [class_label]
        for bs in batch_sizes:
            # Main data
            m, s = bs_accuracies[bs]["means"][class_idx], bs_accuracies[bs]["CI"][class_idx]
            row_cells.append(f"**{m*100:.2f}%** (±{s*100:.2f}%)" if not np.isnan(m) else "NaN")

            # Extra data
            if has_extra:
                if bs in bs_accuracies_extra:
                    idx_e = int(reverted_models_idx_extra[class_label]) if reverted_models_idx_extra else class_idx
                    me, se = bs_accuracies_extra[bs]["means"][idx_e], bs_accuracies_extra[bs]["CI"][idx_e]
                    row_cells.append(f"**{me*100:.2f}%** (±{se*100:.2f}%)" if not np.isnan(me) else "NaN")
                else:
                    row_cells.append("-")
        lines.append("| " + " | ".join(row_cells) + " |")

    # 4. Average Row (Using pre-calculated values)
    avg_cells = ["**Average**"]
    for bs in batch_sizes:
        # Main Average
        am, asig = averages[bs]["mean"], averages[bs]["std"]
        avg_cells.append(f"**{am*100:.2f}%** (±{asig*100:.2f}%)" if not np.isnan(am) else "NaN")

        # Extra Average
        if has_extra:
            if averages_extra and bs in averages_extra:
                aem, aesig = averages_extra[bs]["mean"], averages_extra[bs]["std"]
                avg_cells.append(f"**{aem*100:.2f}%** (±{aesig*100:.2f}%)" if not np.isnan(aem) else "NaN")
            else:
                avg_cells.append("-")

    lines.append("| " + " | ".join(avg_cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# n×m table (rows = original models, columns = variations)
# ---------------------------------------------------------------------------

def make_accuracy_table_nxm_latex(
    stats: dict,
    bs: Any,
    caption: str = "",
    label: str = "",
    show_ci: bool = False,
    compact: bool = True,
    truncate_orig_names: bool = True,
) -> str:
    """Generate a LaTeX table in n×m format using pre-calculated stats."""
    if not stats:
        return ""

    # Detect if we have an "Unseen" scenario
    has_unseen = stats["orig_labels"] and "Unseen" in stats["orig_labels"]

    # Filter out "Unseen" from display if present
    orig_labels_to_display = [o for o in stats["orig_labels"] if o != "Unseen"]

    # Get Unseen accuracy for caption if applicable
    unseen_avg = None
    if has_unseen:
        unseen_avg = stats["row_avgs"].get("Unseen")
        # removing Unseen from headers (work on a copy to avoid mutating caller's data)
        stats = {**stats, "var_names": [v for v in stats["var_names"] if v != "Unseen"]}

    # 1. Setup Headers
    var_names_display = [format_two_line_header(v) for v in stats["var_names"]]

    if show_ci:
        col_spec = "l" + "".join(["r@{\\hspace{4pt}}l" for _ in stats["var_names"]]) + "r@{\\hspace{4pt}}l"
    else:
        col_spec = "l" + "".join(["r" for _ in stats["var_names"]]) + "r"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small" if compact else "",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    # 2. Header Row
    header = "Original Model" if show_ci else "Model"
    for var_display in var_names_display:
        header += f" & \\multicolumn{{2}}{{c}}{{{var_display}}}" if show_ci else f" & {var_display}"
    header += " & \\multicolumn{2}{c}{\\textbf{Avg}}" if show_ci else " & \\textbf{Avg}"
    lines.extend([header + r" \\", r"\midrule"])

    # 3. Body Rows (only for visible original models)
    for orig in orig_labels_to_display:
        display_label = truncate_model_name(orig) if truncate_orig_names else orig
        row = display_label

        # Cells
        for var in stats["var_names"]:
            val = stats["matrix"].get((orig, var))
            m, s = val["mean"], val["std"]
            if np.isnan(m):
                row += " & NaN & " if show_ci else " & -"
            else:
                m_str = f"\\textbf{{{format_value(m)}\\%}}"
                row += f" & {m_str} & {{\\color{{gray}}($\\pm{format_value(s)}\\%$)}}" if show_ci else f" & {m_str}"

        # Row Average
        avg = stats["row_avgs"].get(orig)
        am, asig = avg["mean"], avg["std"]
        if np.isnan(am):
            row += " & -" if not show_ci else " & - & "
        else:
            avg_str = f"\\textbf{{{format_value(am)}\\%}}"
            row += f" & {avg_str} & {{\\color{{gray}}($\\pm{format_value(asig)}\\%$)}}" if show_ci else f" & {avg_str}"

        lines.append(row + r" \\")

    lines.append(r"\midrule")

    # 4. Average Footer Row
    avg_row = "\\textbf{Average}"
    for var in stats["var_names"]:
        avg = stats["col_avgs"].get(var)
        cm, cs = avg["mean"], avg["std"]
        if np.isnan(cm):
            avg_row += " & -" if not show_ci else " & - & "
        else:
            m_str = f"\\textbf{{{format_value(cm)}\\%}}"
            avg_row += f" & {m_str} & {{\\color{{gray}}($\\pm{format_value(cs)}\\%$)}}" if show_ci else f" & {m_str}"

    # Grand Total
    gm, gs = stats["grand_avg"]["mean"], stats["grand_avg"]["std"]
    if np.isnan(gm):
        avg_row += " & -"
    else:
        avg_row += f" & \\textbf{{{format_value(gm)}\\%}}"
        if show_ci:
            avg_row += f" & {{\\color{{gray}}($\\pm{format_value(gs)}\\%$)}}"

    lines.append(avg_row + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])

    # 5. Caption with Unseen info if applicable
    if caption:
        if has_unseen and unseen_avg and not np.isnan(unseen_avg["mean"]):
            unseen_rate = format_value(unseen_avg["mean"])
            unseen_caption = f"{caption} At an Unseen rate of {unseen_rate}\\%, given half of the time the model to classify is unseen."
            lines.append(f"\\caption{{{unseen_caption}}}")
        else:
            lines.append(f"\\caption{{{caption}}}")

    if label:
        lines.append(f"\\label{{{label}}}")

    lines.append(r"\end{table*}")

    return "\n".join(lines)
