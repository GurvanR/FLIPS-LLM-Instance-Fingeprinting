from typing import TypedDict, Callable, Literal
import numpy as np

# --- TypedDicts for matplotlib configs ---

class MatplotlibFigureConfig(TypedDict, total=False):
    figsize: tuple[float, float]
    dpi: float
    facecolor: str
    edgecolor: str
    frameon: bool
    clear: bool

class LegendConfig(TypedDict, total=False):
    fontsize: int | float | str
    title_fontsize: int | float | str
    frameon: bool
    #loc: str # Added in code

class TickLabelConfig(TypedDict, total=False):
    labelsize: int | float

class LabelConfig(TypedDict, total=False):
    fontsize: int | float

class TicksMetricConfig(TypedDict, total=False):
    ticks: np.ndarray

from typing import TypedDict, Literal

class GridConfig(TypedDict, total=False):
    visible: bool
    which: Literal["major", "minor", "both"]
    axis: Literal["both", "x", "y"]
    color: str
    linestyle: str
    linewidth: float
    alpha: float


def get_mpl_configs(
    multiplier: float = 1.0, 
    col_type: Literal["single_col", "double_col"] = "single_col",
    orientation: str = 'horizontal'
):
    """
    Generates a suite of Matplotlib configurations based on a layout 
    type and a scale multiplier.
    """
    
    # Define Base Dimensions (for multiplier = 1.0)
    # Single: (3.5, 2.5), Double: (7, 3.5)
    if orientation == 'horizontal':
        base_fig_size = (4, 2.5) if col_type == "single_col" else (7, 3.5)
    else:
        assert col_type == "single_col"
        base_fig_size = (2.5, 3.5) 
    base_font_small = 8
    base_font_large = 9
    
    # Scale everything
    fig_width = base_fig_size[0] * multiplier
    fig_height = base_fig_size[1] * multiplier
    
    f_small = base_font_small * multiplier
    f_large = base_font_large * multiplier

    return {
        "fig_config": MatplotlibFigureConfig(figsize=(fig_width, fig_height)),
        "legend_config": LegendConfig(
            fontsize=f_small, 
            title_fontsize=f_large, 
            frameon=True
        ),
        "xticks_config": TickLabelConfig(labelsize=f_small),
        "yticks_config": TickLabelConfig(labelsize=f_small),
        "xlabel_config": LabelConfig(fontsize=f_large),
        "ylabel_config": LabelConfig(fontsize=f_large),
        "grid_config": GridConfig(
            visible=True, 
            which="both",
            linestyle='--',
            alpha=0.5, 
            linewidth=0.4 * multiplier # Scaling line width feels better too
        )
    }

# --- Predefined configs ---
FIG_CONFIG_SIMPLE_COL_BIG: MatplotlibFigureConfig = {"figsize": (10.5, 7.5)}
FIG_CONFIG_SIMPLE_COL: MatplotlibFigureConfig = {"figsize": (3.5, 2.5)} #  3.5 * 3  =10.5 2.5 * 3 = 7.5
FIG_CONFIG_DOUBLE_COL: MatplotlibFigureConfig = {"figsize": (7, 3.5)}


FIG_CONFIG_TEMP_TR_SIZE: MatplotlibFigureConfig = {"figsize": (7, 5)}

LEGEND_CONFIG_BIG: LegendConfig = {"fontsize": 21, "title_fontsize": 24}
LEGEND_CONFIG: LegendConfig = {"fontsize": 7, "title_fontsize": 8}

XTICKS_CONFIG_BIG: TickLabelConfig = {"labelsize": 21}
XTICKS_CONFIG: TickLabelConfig = {"labelsize": 7}
YTICKS_CONFIG_BIG: TickLabelConfig = {"labelsize": 21}
YTICKS_CONFIG: TickLabelConfig = {"labelsize": 7}

XLABEL_CONFIG_BIG: LabelConfig = {"fontsize": 24}
XLABEL_CONFIG: LabelConfig = {"fontsize": 12}
YLABEL_CONFIG_BIG: LabelConfig = {"fontsize": 24}
YLABEL_CONFIG: LabelConfig = {"fontsize": 12}

# --- Actual configs ---
xticks_metric_ticks: dict[str, np.ndarray] = {
    "accuracy": np.arange(0.40, 1.10, 0.1),
}

yticks_metric_ticks: dict[str, np.ndarray] = {
    "accuracy_0-1": np.arange(0.40, 1.05, 0.05),
    "accuracy_FLiPS_vs_LLMmap": np.arange(0.00, 1.1, 0.1),
    "accuracy_FLiPS": np.arange(0.70, 1.00, 0.05),
    "accuracy_Monochar": np.arange(0.60, 1.05, 0.05),
}


YTICKS_METRIC_CONFIG: Callable[[str], TicksMetricConfig] = (
    lambda ftype_name,: {"ticks": yticks_metric_ticks.get(ftype_name, yticks_metric_ticks["accuracy_0-1"])}
)
XTICKS_HISTF4_CONFIG: Callable[[str], TicksMetricConfig] = (
    lambda metric: {"ticks": xticks_metric_ticks.get(metric, xticks_metric_ticks["accuracy"])}
)

GRID_CONFIG: GridConfig = {
    "visible": True,
    "which": "both",  # type-checked
    "linestyle": "--",
    "alpha": 0.5,
}
