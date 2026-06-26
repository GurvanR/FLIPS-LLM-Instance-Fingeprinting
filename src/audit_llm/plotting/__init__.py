"""plotting — Visualization package for audit_llm.

Re-exports all public symbols for backward compatibility.
For new code, prefer importing from the specific submodules.
"""

# --- Constants ---
from audit_llm.plotting.constants import (
    COLORBLIND_COLORS,
    COLOR_DELTA_TP_MAP,
    SHORTLIST_OF_LLMS,
    TINYLIST_OF_LLMS,
)

# --- Figure I/O ---
from audit_llm.plotting.figure_io import clip_std, save_fig_and_show

# --- Label formatting ---
from audit_llm.plotting.label_formatting import (
    format_label_multiline,
    format_two_line_header,
    format_value,
)

# --- Tables ---
from audit_llm.plotting.tables import (
    make_accuracy_table_latex,
    make_accuracy_table_markdown,
    make_accuracy_table_nxm_latex,
)

# --- Heatmaps ---
from audit_llm.plotting.heatmaps import make_accuracy_heatmap

# --- Performance curves ---
from audit_llm.plotting.perf_curves import (
    generate_personalized_figures,
    plot_performance_vs_axis_iterator,
)

# --- Threshold plots ---
from audit_llm.plotting.threshold_plots import (
    extract_thresholds,
    plot_alpha_roc_curves,
    plot_alpha_tradeoff,
    plot_openset_roc_curves,
    plot_roc_curves_overlay,
    plot_thresholds_distribution,
    plot_unseen_and_global_pr_vs_alpha,
    plot_unseen_and_global_pr_vs_confidence,
    save_openset_metrics_table,
)

# --- FigureRenderer ---
from audit_llm.plotting.figure_renderer import (
    FigureRenderer,
    PerformancePlotRenderer,
    get_renderer,
    register_renderer,
    render_figures,
)

__all__ = [
    # constants
    "COLORBLIND_COLORS",
    "COLOR_DELTA_TP_MAP",
    "SHORTLIST_OF_LLMS",
    "TINYLIST_OF_LLMS",
    # figure_io
    "save_fig_and_show",
    "clip_std",
    # label_formatting
    "format_label_multiline",
    "format_two_line_header",
    "format_value",
    # tables
    "make_accuracy_table_latex",
    "make_accuracy_table_markdown",
    "make_accuracy_table_nxm_latex",
    # heatmaps
    "make_accuracy_heatmap",
    # perf_curves
    "generate_personalized_figures",
    "plot_performance_vs_axis_iterator",
    # threshold_plots
    "plot_alpha_tradeoff",
    "plot_alpha_roc_curves",
    "plot_roc_curves_overlay",
    "plot_thresholds_distribution",
    "extract_thresholds",
    "plot_openset_roc_curves",
    "plot_unseen_and_global_pr_vs_alpha",
    "plot_unseen_and_global_pr_vs_confidence",
    "save_openset_metrics_table",
    # figure_renderer
    "FigureRenderer",
    "PerformancePlotRenderer",
    "register_renderer",
    "get_renderer",
    "render_figures",
]
