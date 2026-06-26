"""Pydantic schema for experiment YAML configuration files.

Provides strict validation of experiment configs with:
- Duplicate YAML key detection
- experiment_fun validation against EXPERIMENT_FUNCTION_MAP
- Typed sub-models for sampling_parameters, figures, and classification_config

Usage
-----
>>> from audit_llm.experiment_config_schema import load_experiment_config
>>> config = load_experiment_config("path/to/config.yaml")
>>> xp_config = config.model_dump()  # back to dict for backward compat
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


# ---------------------------------------------------------------------------
# Custom YAML loader that rejects duplicate keys
# ---------------------------------------------------------------------------


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that raises on duplicate mapping keys."""

    pass


def _unique_key_constructor(loader: yaml.Loader, node: yaml.MappingNode, deep: bool = False) -> dict:
    """Construct a mapping, raising ValueError on duplicate keys."""
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(
                f"Duplicate YAML key '{key}' found at line {key_node.start_mark.line + 1}"
            )
        value = loader.construct_object(value_node, deep=deep)
        mapping[key] = value
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _unique_key_constructor,
)


# ---------------------------------------------------------------------------
# Pydantic sub-models
# ---------------------------------------------------------------------------


class SamplingParameters(BaseModel):
    """Sampling parameter specification — each key maps to a list of allowed values."""

    model_config = ConfigDict(extra="allow")

    system_prompt_idx: List[int] = []
    temperature: List[float] = []
    frequency_penalty: List[float] = [0.0]


class FigureConfig(BaseModel):
    """Configuration for a single figure in the experiment."""

    model_config = ConfigDict(extra="allow")

    name: Optional[str] = None
    type: str = "lineplot"
    x_axis: str = ""  # mapped from "x-axis" via alias
    x_mode: str = "categorical"
    y_axis: str = "metric"  # mapped from "y-axis" via alias
    metrics: List[str] = []
    group_by: str = "none"
    aggregation: str = "mean"
    error_bar: str = "std"
    repeat_for_each: str = "none"
    layout: Optional[str] = None
    grid_columns: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def _remap_hyphenated_keys(cls, data: Any) -> Any:
        """Accept YAML keys with hyphens (x-axis, y-axis) and remap them."""
        if isinstance(data, dict):
            if "x-axis" in data:
                data["x_axis"] = data.pop("x-axis")
            if "y-axis" in data:
                data["y_axis"] = data.pop("y-axis")
        return data


class ClassificationConfig(BaseModel):
    """Classification pipeline configuration."""

    model_config = ConfigDict(extra="allow")

    classifiers: List[str] = ["XGBoost"]
    splitter_type: str = "StratifiedShuffleSplit"
    n_splits: int = 2
    test_size: Union[float, int] = 0.5
    force_class_size: Union[int, Literal["auto"], None] = "auto"
    classifier_metrics: List[str] = ["accuracy"]
    batch_prediction_sizes: Optional[List[int]] = None
    batch_types: Optional[List[str]] = None
    openset: bool = False
    m_test_size: Union[float, int] = 5
    openset_m_splits: int = 10
    default_normalization: str = "auto"
    normalization_methods: Dict[str, str] = {"seq_length": "none"}
    is_closed: Optional[bool] = None
    train_size_dict_2_checkpoint_path: Optional[str] = None
    train_size_dict_map: Optional[Dict[str, str]] = None
    alpha_quantile_threshold: Optional[float] = None
    alpha_trade_off_show: Optional[bool] = None
    unique_tp_in_mix: Optional[Union[int, List[int], str, List[str]]] = ["max"]
    max_nb_of_uplet: Optional[int] = None
    store_prediction_probas: bool = True
    compute_micro_pr_curve: bool = True
    compute_confusion_matrices: bool = False
    openset_fig_cache: bool = False
    micro_pr_curve_cache: bool = True

    @model_validator(mode="after")
    def _validate_force_class_size(self) -> "ClassificationConfig":
        """Guard the unstated ``force_class_size > test_size`` invariant at load time.

        ``StratifiedShuffleSplit`` is invoked with an *absolute* ``test_size * n_classes``
        over a pool capped at ``force_class_size * n_classes`` rows, so the split is only
        feasible when ``force_class_size > test_size`` and ``test_size`` is a positive
        integer per-class hold-out count. When ``force_class_size`` is an explicit int we
        can check this now and fail with a clear message instead of crashing deep inside
        sklearn. ``"auto"`` (the default) and ``None`` are data-dependent / no-op and are
        validated at resolution time inside the classifier.
        """
        fcs = self.force_class_size
        if isinstance(fcs, bool):  # YAML true/false must not slip through as 1/0
            raise ValueError("force_class_size must be a positive int or 'auto', not a bool.")
        if isinstance(fcs, int):
            ts = self.test_size
            if isinstance(ts, bool) or not isinstance(ts, int):
                raise ValueError(
                    f"force_class_size={fcs} requires test_size to be a positive integer "
                    f"(per-class hold-out count); got test_size={ts!r}. A fractional "
                    "test_size is unsupported when force_class_size is set."
                )
            if ts <= 0:
                raise ValueError(
                    f"test_size must be a positive integer when force_class_size is set; got {ts}."
                )
            if fcs <= ts:
                raise ValueError(
                    f"force_class_size ({fcs}) must be strictly greater than test_size ({ts}); "
                    "otherwise StratifiedShuffleSplit cannot form a valid test fold (it is "
                    "invoked with test_size * n_classes over a pool of force_class_size * "
                    "n_classes rows)."
                )
        return self

    @model_validator(mode="after")
    def _derive_batch_types(self) -> "ClassificationConfig":
        """Derive ``batch_types`` from ``batch_prediction_sizes`` when unset.

        Batch-size convention: a batch of 1 can only be classified token-pair-wise
        (``tp_wise``); batches > 1 mix token pairs at prediction (``mix_tp_at_pred``).
        So an omitted ``batch_types`` is inferred from the requested batch sizes —
        ``tp_wise`` if size 1 is present, ``mix_tp_at_pred`` if any size > 1.
        Setting ``batch_types`` explicitly overrides this inference.
        """
        if self.batch_types is None and self.batch_prediction_sizes:
            derived: List[str] = []
            if any(bs == 1 for bs in self.batch_prediction_sizes):
                derived.append("tp_wise")
            if any(bs > 1 for bs in self.batch_prediction_sizes):
                derived.append("mix_tp_at_pred")
            self.batch_types = derived or None
        return self


# ---------------------------------------------------------------------------
# Top-level experiment config
# ---------------------------------------------------------------------------

# Valid experiment function names — must match EXPERIMENT_FUNCTION_MAP keys
# in experiment_runner.py.  Kept as a module-level set so it can be imported
# independently of heavy classification imports.
VALID_EXPERIMENT_FUNCTIONS = frozenset({
    "Nist_perf_chart",
    "Seq_Length_visualization",
    "Valid_count_chart",
    "classify",
    "classify_cross_token_pairs",
    "batch_classification_across_token_pairs",
    "LLMmap_classification",
    "Save_pv_in_parquet",
    "feature_space_visualization",
    # Legacy names still accepted
    "Classification",
    "Batch_Classification_across_token_pairs",
})


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration loaded from a YAML file.

    All fields use ``extra="allow"`` so that future or ad-hoc keys
    (e.g. ``cuda_visible_devices``, ``save_dca_showcase_data``) pass through
    without breaking validation.
    """

    model_config = ConfigDict(extra="allow")

    # --- Required ---
    experiment_fun: str

    # --- Common optional fields ---
    alpha: float = 0.05
    features: Optional[str] = "SmallNist"  # canonical NIST feature set; every shipped config uses it
    min_seq_length: int = 100
    set_constant_seq_length: Optional[int] = None  # None = no truncation, int = truncate all sequences to this length before NIST computation
    PRNGs: Optional[Union[List[Any], str]] = None  # None, [], or "None"

    models: List[str] = []
    token_pairs: List[str] = []
    models_to_remove: Optional[List[str]] = None
    include_quantized: Optional[List[str]] = None  # None → exclude all @@-suffixed quantized variants, [] → include all, ["fp8"] → only fp8

    sampling_parameters: SamplingParameters = SamplingParameters()

    # Declarative scenario file (path to config/scenarios/*.yaml). When set, the
    # analysis layer sources model × variation selection from the scenario via
    # build_instances; the legacy model_variations / quantized_model_variations /
    # abliterated_models keys below act as a fallback when it is absent.
    scenario: Optional[str] = None

    # model_variations can be a dict, list of dicts, or a string (e.g. "temperature")
    model_variations: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]] = None
    # quantized_model_variations: separate variation grid for @@-suffixed quantized models
    # None → quantized models use the same model_variations as base models
    quantized_model_variations: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]] = None

    abliterated_models: Optional[List[str]] = None  # None → skip, [] → all, [names] → specific

    # Model groups for group-based classification (N-class = N groups instead of N models)
    # None → standard per-model classification
    # Dict with "group_by" key → parameter-based grouping (e.g., {"group_by": "temperature"})
    # Dict with group_name → list of model names → hardcoded grouping
    # Dict with group_name → {models: [...], temperature: [...]} → hardcoded with param filters
    model_groups: Optional[Dict[str, Any]] = None

    @field_validator("abliterated_models", mode="before")
    @classmethod
    def _normalize_abliterated_models(cls, v: Any) -> Optional[List[str]]:
        """None/'None' → None (skip), [] → [] (all), [names] → specific models."""
        if v is None or v == "None":
            return None
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            return [v]  # single model name as string
        return v

    # Calculations: readable keys mapping iterator-name → iterator-name
    calculations: Optional[Dict[str, str]] = None

    # Figures: named figure configs
    figures: Optional[Dict[str, FigureConfig]] = None

    train_sizes: Optional[List[int]] = None
    train_size_dict_path: Optional[Dict[str, str]] = None

    classification_config: Optional[ClassificationConfig] = None

    # Reproducibility / dispatch fields
    dataset_cross_split_seed: Optional[int] = None
    cuda_visible_devices: Optional[str] = None
    stop_computing_splits: Optional[bool] = None
    train_only: Optional[bool] = None
    save_dca_showcase_data: Optional[bool] = None

    # These are added at runtime by XP_script_global.py
    xp_name: Optional[str] = None
    save: Optional[bool] = None

    @field_validator("experiment_fun")
    @classmethod
    def _validate_experiment_fun(cls, v: str) -> str:
        if v not in VALID_EXPERIMENT_FUNCTIONS:
            raise ValueError(
                f"Unknown experiment_fun '{v}'. "
                f"Valid options: {sorted(VALID_EXPERIMENT_FUNCTIONS)}"
            )
        return v


# ---------------------------------------------------------------------------
# Public loading function
# ---------------------------------------------------------------------------


def load_experiment_config(path: Union[str, Path]) -> dict:
    """Load and validate an experiment YAML config.

    Parameters
    ----------
    path : str or Path
        Path to the YAML config file.

    Returns
    -------
    dict
        Validated configuration as a plain dictionary (for backward
        compatibility with code that expects ``xp_config`` as a dict).

    Raises
    ------
    ValueError
        If the YAML contains duplicate keys.
    pydantic.ValidationError
        If the config fails schema validation.
    """
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.load(f, Loader=_UniqueKeyLoader)

    if raw is None:
        raise ValueError(f"Empty or unparseable YAML file: {path}")

    # Validate through Pydantic
    config = ExperimentConfig.model_validate(raw)

    # Return as dict for backward compatibility
    return config.model_dump(exclude_none=False)
