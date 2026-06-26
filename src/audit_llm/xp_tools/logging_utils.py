"""Logging utilities for experiment output management."""

import logging
from pathlib import Path


def setup_experiment_logging(save_fig_path: Path, *, level: int = logging.DEBUG) -> None:
    """Configure file + console logging for one experiment run.

    Creates two log files in *save_fig_path*:
      - ``summary.log``  (INFO)  — high-level milestones, easy to monitor.
      - ``detailed.log`` (DEBUG) — full diagnostics, shapes, per-split info.

    Also attaches a console handler at INFO level so ``screen`` sessions
    still show progress.  Previous handlers are cleared to avoid duplication
    when called more than once in the same process.
    """
    pkg_logger = logging.getLogger("audit_llm")
    pkg_logger.setLevel(level)
    # Clear any prior handlers (prevents duplication in nested calls)
    pkg_logger.handlers.clear()

    fmt_summary = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    fmt_detail = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d — %(message)s",
        datefmt="%H:%M:%S",
    )

    save_fig_path = Path(save_fig_path)
    save_fig_path.mkdir(parents=True, exist_ok=True)

    summary_handler = logging.FileHandler(
        save_fig_path / "summary.log", mode="a", encoding="utf-8",
    )
    summary_handler.setLevel(logging.INFO)
    summary_handler.setFormatter(fmt_summary)

    detail_handler = logging.FileHandler(
        save_fig_path / "detailed.log", mode="a", encoding="utf-8",
    )
    detail_handler.setLevel(logging.DEBUG)
    detail_handler.setFormatter(fmt_detail)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt_summary)

    pkg_logger.addHandler(summary_handler)
    pkg_logger.addHandler(detail_handler)
    pkg_logger.addHandler(console_handler)


def teardown_experiment_logging() -> None:
    """Remove all handlers from the package logger."""
    pkg_logger = logging.getLogger("audit_llm")
    for h in pkg_logger.handlers[:]:
        h.close()
        pkg_logger.removeHandler(h)


def setup_output_logging(save_fig_path, dataset, xp_config):
    """Set up output folder and configure logging.

    Backward-compatible wrapper that replaces the old builtins.print override.
    Returns the dataset-specific output folder path.
    """
    temp_save_fig_path = save_fig_path / (dataset + "_" + xp_config["mode"])
    temp_save_fig_path.mkdir(parents=True, exist_ok=True)
    setup_experiment_logging(temp_save_fig_path)
    return temp_save_fig_path
