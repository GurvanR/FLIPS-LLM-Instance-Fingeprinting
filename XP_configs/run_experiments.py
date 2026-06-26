#!/usr/bin/env python3
"""Python dispatcher for running multiple experiments.

Replaces the legacy ``run_multi_xp.sh`` script with a validated,
reproducible Python CLI. Pre‑validates every YAML config with Pydantic
before dispatching.

Outputs and Errors are combined and logged to:
    xp_logs/<run_name>/<xp-suffix>/xp_{i}_log_{timestamp}.log

Usage examples
--------------
# Inline mode — run all YAMLs in a folder sequentially:
python run_experiments.py --config-dir Final_FLiPS_ICML \\
    --run-name FLiPS_ICML_run --xp-suffix default --mode inline

# Dry‑run — validate all configs and print what would be executed:
python run_experiments.py --config-dir Final_FLiPS_ICML \\
    --run-name FLiPS_ICML_run --mode dry-run
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

# ---------------------------------------------------------------------------
# Resolve project paths 
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent  # XP_configs/
REPO_ROOT = SCRIPT_DIR.parent
LOG_DIR = REPO_ROOT / "xp_logs"


def _find_yaml_configs(config_dir: Path, recursive: bool) -> list[Path]:
    """Collect *.yaml files under *config_dir*, optionally recursively."""
    pattern = "**/*.yaml" if recursive else "*.yaml"
    configs = sorted(config_dir.glob(pattern))
    # Exclude XP_config_libs and Old_configs by convention
    exclude_dirs = {"XP_config_libs", "Old_configs"}
    configs = [
        c for c in configs
        if not any(part in exclude_dirs for part in c.parts)
    ]
    return configs


def _validate_configs(config_paths: list[Path]) -> list[Path]:
    """Validate all configs with Pydantic, returning those that pass."""
    # Lazy import to avoid import-time dependency on the whole package
    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from audit_llm.experiment_config_schema import load_experiment_config
    except ImportError:
        click.secho("Warning: Could not import validation schema. Skipping validation.", fg="yellow")
        return config_paths

    valid: list[Path] = []
    for path in config_paths:
        try:
            load_experiment_config(path)
            click.echo(f"  ✓ {path.name}")
            valid.append(path)
        except Exception as exc:
            click.secho(f"  ✗ {path.name}: {exc}", fg="red")
    return valid


def _build_command(config_path: Path, run_name: str, xp_suffix: str) -> list[str]:
    """Build the command line for a single experiment run."""
    return [
        sys.executable,
        str(SCRIPT_DIR / "XP_script_global.py"),
        "--run_name", run_name,
        "--xp_suffix", xp_suffix,
        "--xp_config_path", str(config_path.relative_to(REPO_ROOT)),
    ]


def _get_log_path(run_name: str, xp_suffix: str, index: int) -> Path:
    """Generate path: xp_logs/<run_name>/<xp-suffix>/xp_{i}_log_{timestamp}.log"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / run_name / xp_suffix / f"xp_{index}_log_{timestamp}.log"


def _ensure_log_dir(log_path: Path) -> None:
    """Create the parent directory for the log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)


def _run_inline(
    items: list[tuple[Path, list[str]]], 
    run_name: str, 
    xp_suffix: str,
    sleep_hours: float
) -> None:
    """Run all commands sequentially in the current process, combining output/error."""
    if sleep_hours > 0:
        click.echo(f"Sleeping for {sleep_hours}h before start…")
        time.sleep(sleep_hours * 3600)

    for i, (cfg_path, cmd) in enumerate(items, 1):
        log_path = _get_log_path(run_name, xp_suffix, i)
        _ensure_log_dir(log_path)
        
        click.echo(f"\n[{i}/{len(items)}] {shlex.join(cmd)}")
        click.echo(f"  Log: {log_path.relative_to(REPO_ROOT)}")
        
        with open(log_path, "w") as f_log:
            result = subprocess.run(
                cmd, 
                cwd=str(REPO_ROOT), 
                stdout=f_log, 
                stderr=subprocess.STDOUT  # Combine stderr into stdout
            )
        
        if result.returncode != 0:
            click.secho(f"Command failed with exit code {result.returncode}. See log.", fg="red")
        else:
            click.secho(f"  XP done successfully.", fg="green")


def _run_slurm(
    items: list[tuple[Path, list[str]]],
    run_name: str,
    xp_suffix: str,
    slurm_time: str,
    slurm_gpus: int,
    slurm_cpus: int,
    sleep_hours: float,
) -> None:
    """Submit each command as a SLURM job, combining logs."""
    for i, (cfg_path, cmd) in enumerate(items, 1):
        log_path = _get_log_path(run_name, xp_suffix, i)
        _ensure_log_dir(log_path)
        
        # Paths for SLURM should be relative to CWD or absolute
        rel_log = log_path.relative_to(REPO_ROOT)

        sbatch_cmd = [
            "sbatch",
            f"--time={slurm_time}",
            f"--gres=gpu:{slurm_gpus}",
            f"--cpus-per-task={slurm_cpus}",
            f"--output={rel_log}",  # SLURM writes both stdout/stderr here if --error is omitted
            "--wrap", shlex.join(cmd),
        ]
        if sleep_hours > 0:
            sbatch_cmd.extend(["--begin", f"now+{int(sleep_hours * 60)}minutes"])
        
        click.echo(f"Submitting {cfg_path.name} -> {rel_log}")
        subprocess.run(sbatch_cmd, check=True, cwd=str(REPO_ROOT))


def _run_screen(
    items: list[tuple[Path, list[str]]],
    run_name: str,
    xp_suffix: str,
    screen_name: Optional[str],
    separate: bool,
    sleep_hours: float,
) -> None:
    """Run commands in GNU screen session(s) with shell redirection (2>&1)."""

    def _wrap_redirect(cmd_list: list[str], log_p: Path, prefix: str = "") -> str:
        """Helper to append redirection to the shell command string.

        If *prefix* is given, wraps prefix + cmd in a subshell so both share
        the same log redirection.
        """
        rel_log = log_p.relative_to(REPO_ROOT)
        # > log 2>&1 combines stdout and stderr
        if prefix:
            return f"({prefix}{shlex.join(cmd_list)}) > {rel_log} 2>&1"
        return f"{shlex.join(cmd_list)} > {rel_log} 2>&1"

    sleep_prefix = ""
    if sleep_hours > 0:
        sleep_seconds = int(sleep_hours * 3600)
        sleep_prefix = (
            f"echo \"[$(date '+%Y-%m-%d %H:%M:%S')] Sleeping for {sleep_hours}h before start…\" && "
            f"sleep {sleep_seconds} && "
            f"echo \"[$(date '+%Y-%m-%d %H:%M:%S')] Sleep done, launching experiment.\" && "
        )

    if separate:
        for i, (cfg_path, cmd) in enumerate(items, 1):
            log_path = _get_log_path(run_name, xp_suffix, i)
            _ensure_log_dir(log_path)

            # Screen Name: 1_experiment_name
            name = f"{i}_{cfg_path.stem}"
            full_cmd_str = _wrap_redirect(cmd, log_path, prefix=sleep_prefix)

            subprocess.run(
                ["screen", "-dmS", name, "bash", "-c", full_cmd_str],
                cwd=str(REPO_ROOT),
            )
            click.echo(f"Started screen '{name}' -> {log_path.relative_to(REPO_ROOT)}")
    else:
        # Single screen mode: default to xp_suffix if screen_name not provided
        name = screen_name if screen_name else xp_suffix

        # Build chain of commands; sleep notice goes to the first log only.
        chained_cmds_list = []
        for i, (cfg_path, cmd) in enumerate(items, 1):
            log_path = _get_log_path(run_name, xp_suffix, i)
            _ensure_log_dir(log_path)
            prefix = sleep_prefix if i == 1 else ""
            chained_cmds_list.append(_wrap_redirect(cmd, log_path, prefix=prefix))

        chained_cmds_str = " && ".join(chained_cmds_list)

        subprocess.run(
            ["screen", "-dmS", name, "bash", "-c", chained_cmds_str],
            cwd=str(REPO_ROOT),
        )
        click.echo(f"Started screen '{name}' with {len(items)} commands.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--config-dir", required=True,
    help="Directory containing YAML experiment configs (relative to XP_configs/).",
)
@click.option("--run-name", required=True, help="Name of the production run.")
@click.option("--xp-suffix", default="default", help="Experiment suffix.")
@click.option("--recursive", is_flag=True, help="Search YAMLs recursively.")
@click.option(
    "--mode",
    type=click.Choice(["inline", "slurm", "single-screen", "separate-screens", "dry-run"]),
    default="inline",
    help="Execution mode (default: inline).",
)
@click.option("--sleep", default=0.0, type=float, help="Hours to sleep before starting.")
@click.option("--slurm-time", default="24:00:00", help="SLURM --time value.")
@click.option("--slurm-gpus", default=1, type=int, help="SLURM GPUs per task.")
@click.option("--slurm-cpus", default=4, type=int, help="SLURM CPUs per task.")
@click.option(
    "--screen-name", 
    default=None, 
    help="GNU screen session name (defaults to xp_suffix for single-screen)."
)
def main(
    config_dir: str,
    run_name: str,
    xp_suffix: str,
    recursive: bool,
    mode: str,
    sleep: float,
    slurm_time: str,
    slurm_gpus: int,
    slurm_cpus: int,
    screen_name: Optional[str],
) -> None:
    """Run multiple experiment configs with pre‑validation.

    Replaces run_multi_xp.sh with Pydantic validation and Python dispatch.
    """
    config_root = SCRIPT_DIR / config_dir
    if not config_root.is_dir():
        click.secho(f"Config directory not found: {config_root}", fg="red")
        raise SystemExit(1)

    # 1. Discover configs
    configs = _find_yaml_configs(config_root, recursive)
    if not configs:
        click.secho(f"No YAML configs found in {config_root}", fg="yellow")
        raise SystemExit(1)

    click.echo(f"Found {len(configs)} config(s) in {config_root}")

    # 2. Validate all configs
    click.echo("\nValidating configs…")
    valid_configs = _validate_configs(configs)

    if len(valid_configs) < len(configs):
        click.secho(
            f"\n{len(configs) - len(valid_configs)} config(s) failed validation.",
            fg="red",
        )
        if not valid_configs:
            raise SystemExit(1)
        if not click.confirm("Continue with valid configs only?"):
            raise SystemExit(1)

    # 3. Build commands
    items = [(c, _build_command(c, run_name, xp_suffix)) for c in valid_configs]

    click.echo(f"\n{len(items)} experiment(s) to run (mode={mode})")

    # 4. Dispatch
    if mode == "dry-run":
        click.echo("\nDry-run — commands that would be executed:")
        for i, (cfg, cmd) in enumerate(items, 1):
            log_path = _get_log_path(run_name, xp_suffix, i)
            click.echo(f"  [Config: {cfg.name}] -> {shlex.join(cmd)}")
            click.echo(f"   L> Log: {log_path.relative_to(REPO_ROOT)}")
    elif mode == "inline":
        _run_inline(items, run_name, xp_suffix, sleep)
    elif mode == "slurm":
        _run_slurm(items, run_name, xp_suffix, slurm_time, slurm_gpus, slurm_cpus, sleep)
    elif mode in ("single-screen", "separate-screens"):
        _run_screen(
            items,
            run_name=run_name,
            xp_suffix=xp_suffix,
            screen_name=screen_name,
            separate=(mode == "separate-screens"),
            sleep_hours=sleep,
        )


if __name__ == "__main__":
    main()