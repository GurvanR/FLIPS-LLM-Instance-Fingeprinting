"""FigureRenderer Protocol and registry for declarative figure dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class FigureRenderer(Protocol):
    """Interface for declarative (YAML-driven) figure renderers."""

    figure_type: str

    def render(
        self,
        data: dict[str, Any],
        figure_config: dict[str, Any],
        xp_config: dict[str, Any],
        save_path: str | Path | None,
        **kwargs: Any,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_RENDERER_REGISTRY: dict[str, FigureRenderer] = {}


def register_renderer(renderer: FigureRenderer) -> FigureRenderer:
    """Register a renderer instance for its ``figure_type``."""
    _RENDERER_REGISTRY[renderer.figure_type] = renderer
    return renderer


def get_renderer(figure_type: str) -> FigureRenderer:
    """Look up a registered renderer by figure type."""
    if figure_type not in _RENDERER_REGISTRY:
        raise KeyError(
            f"No renderer registered for figure_type={figure_type!r}. "
            f"Available: {list(_RENDERER_REGISTRY)}"
        )
    return _RENDERER_REGISTRY[figure_type]


def render_figures(
    xp_config: dict[str, Any],
    data: dict[str, Any],
    save_path: str | Path | None,
    **kwargs: Any,
) -> None:
    """Dispatch each figure in ``xp_config['figures']`` to its registered renderer."""
    for figure_idx, figure_config in xp_config.get("figures", {}).items():
        figure_type = figure_config.get("type", "lineplot")
        renderer = get_renderer(figure_type)
        renderer.render(
            data=data,
            figure_config=figure_config,
            xp_config=xp_config,
            save_path=save_path,
            figure_idx=figure_idx,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Concrete renderer: PerformancePlotRenderer
# ---------------------------------------------------------------------------

class PerformancePlotRenderer:
    """Renderer for ``barplot`` and ``lineplot`` figure types.

    Wraps :func:`~audit_llm.plotting.perf_curves.generate_personalized_figures`.
    Register one instance per figure type.
    """

    def __init__(self, figure_type: str) -> None:
        self.figure_type = figure_type

    def render(
        self,
        data: dict[str, Any],
        figure_config: dict[str, Any],
        xp_config: dict[str, Any],
        save_path: str | Path | None,
        **kwargs: Any,
    ) -> Any:
        from audit_llm.plotting.perf_curves import generate_personalized_figures

        generate_personalized_figures(
            xp_config=xp_config,
            calculations_config=kwargs.get("calculations_config", xp_config.get("calculations", {})),
            calculations_iter_lists=kwargs.get("calculations_iter_lists", {}),
            new_model_idx=kwargs.get("new_model_idx", {}),
            save_fig_path=save_path,
            data_dict=data,
            pipe_summary_mode=kwargs.get("pipe_summary_mode", False),
            tp_group_names=kwargs.get("tp_group_names", None),
        )


# --- Auto-register built-in renderers ---
register_renderer(PerformancePlotRenderer("barplot"))
register_renderer(PerformancePlotRenderer("lineplot"))
