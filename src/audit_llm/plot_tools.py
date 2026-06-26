# SPDX-FileCopyrightText: 2024 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Deprecated module — plotting functions have been split into focused submodules.

Use the canonical submodules directly:
  - plotting.constants        : COLORBLIND_COLORS, COLOR_DELTA_DS_MAP, TINYLIST_OF_LLMS, ...
  - plotting.figure_io        : save_fig_and_show, clip_std
  - plotting.label_formatting : format_label_multiline, format_two_line_header, format_value
  - plotting.tables           : make_accuracy_table_latex, make_accuracy_table_markdown, ...
  - plotting.heatmaps         : make_accuracy_heatmap
  - plotting.perf_curves      : generate_personalized_figures, plot_performance_vs_axis_iterator
  - plotting.threshold_plots  : plot_alpha_tradeoff, plot_thresholds_distribution, extract_thresholds
  - plotting.figure_renderer  : FigureRenderer, PerformancePlotRenderer
  - xp_tools.model_filtering  : truncate_model_name, full_var_model_name_to_original_model_name, ...
  - xp_tools.label_formatting : format_tp_group_name_label, put_uppercase_first, ...
  - data_transforms           : revert_dictionary, nested_loop
"""

# Backward-compat re-export for off-limits consumers (Fingerprinting_methods)
from audit_llm.xp_tools.model_filtering import (  # noqa: F401
    full_var_model_name_to_full_safe_var_model_name_mapper,
)
