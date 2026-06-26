"""Multi-token-pair batch classification orchestration.

Top-level ``batch_classification_across_token_pairs()`` entry point that
manages train-size sweeps, batch-type routing, checkpoint persistence,
DCA analysis, and post-classification visualization.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

from audit_llm.Classification.Confusion_matrices_on_var import create_confusion_matrix_heatmaps
from audit_llm.Classification.openset_classification import OpenSetClassification, _alpha_str
from audit_llm.Classification.dca_analysis import (
    compute_dca_showcase_data,
    plot_dca_showcase_by_original_model,
)
from audit_llm.Classification.multi_classification import MultiTokenPairClassification
from audit_llm.Classification.results_tables import (
    get_effective_batch_type_keys,
    make_model_wise_tables,
    plot_confusion_matrices_on_tr_size_dict,
)
from audit_llm.Classification.training_size_analysis import (
    plot_train_size_curves,
    plot_trainsize_wise_curves,
)
from audit_llm.file_io import load_json, write_json
from audit_llm.xp_tools import (
    dict_product_with_fix_item,
    get_calculation_item_name,
    get_iter_idx_from_calculations_config,
    prepare_dataset_features,
    prepare_experiment_context,
    remap_features_index,
    select_features_for_token_pair,
)
from audit_llm.xp_tools.model_filtering import (
    full_var_model_name_to_full_safe_var_model_name_mapper,
)

logger = logging.getLogger(__name__)


def _log_batch_summary(summary_results: Dict, train_size: int, batch_type: str, label: str = "") -> None:
    """Log accuracy averaged over token pairs per (clf, batch_size) to the summary log."""
    for bt, bt_summary in summary_results.items():
        for batch_size, bs_summary in bt_summary.items():
            if not isinstance(bs_summary, dict):
                continue
            acc_per_clf: Dict[str, list] = {}
            for ds_summary in bs_summary.values():
                if not isinstance(ds_summary, dict):
                    continue
                for clf_name, clf_s in ds_summary.items():
                    if not isinstance(clf_s, dict):
                        continue
                    acc_m = clf_s.get("accuracy_mean", float("nan"))
                    acc_per_clf.setdefault(clf_name, []).append(acc_m)
            for clf_name, accs in acc_per_clf.items():
                valid = [a for a in accs if not np.isnan(a)]
                if valid:
                    tag = label or bt
                    logger.info(
                        "[ts=%s, %s, bs=%s] %s — accuracy (avg over %d tps): %.4f ± %.4f",
                        train_size, tag, batch_size, clf_name, len(valid),
                        float(np.mean(valid)), float(np.std(valid)),
                    )


def _save_incremental_checkpoint(
    step_key: str,
    completed_steps: set,
    results_entry: Dict,
    checkpoint_path: Path,
    completed_steps_path: Path,
) -> None:
    """Save an incremental checkpoint after one batch-type step completes."""
    completed_steps.add(step_key)
    joblib.dump(results_entry, checkpoint_path, compress=3)
    joblib.dump(list(completed_steps), completed_steps_path)
    logger.info("Checkpoint saved for step %s at %s", step_key, checkpoint_path)


def _prepare_all_token_pair_features(
    token_pairs: List[str],
    calculation_item: Dict,
    ctx,
    Experiment_config: Dict[str, Any],
) -> tuple:
    """Load and remap features for every token pair in the experiment.

    Returns (X_s, remapped_features_index, current_new_var_models_idx).
    """
    X_s: Dict[str, np.ndarray] = {}
    current_new_var_models_idx = None

    for token_pair in token_pairs:
        selected_features_for_classification = select_features_for_token_pair(
            ctx.intra_samples_feature_index_dict, ctx.xp_config
        )
        logger.debug("sanity check that only _ts values and no _pv: %s", selected_features_for_classification)
        remapped_features_index = remap_features_index(selected_features_for_classification)

        X_s[token_pair], current_new_var_models_idx = prepare_dataset_features(
            token_pair,
            calculation_item,
            ctx.intra_samples_features_dict,
            ctx.intra_samples_feature_index_dict,
            selected_features_for_classification,
            ctx.new_var_models_idx,
            ctx.models_indices,
            ctx.xp_config,
            Experiment_config,
            ctx.MainDataset_df_iterators,
            ctx.Answers_df,
        )

    return X_s, remapped_features_index, current_new_var_models_idx


def _run_batch_type_step(
    X_s: Dict[str, np.ndarray],
    remapped_features_index: Dict[str, int],
    new_var_models_idx: Dict,
    effective_config: Dict[str, Any],
    batch_type: str,
    token_pairs: List[str],
    xp_config: Dict,
    save_dir: Path,
    train_size: int,
    token_pairs_banned_path: Path,
) -> Dict:
    """Run one batch-type classification step and return its raw summary results."""
    multi_classifier = MultiTokenPairClassification(
        X_s,
        remapped_features_index,
        new_var_models_idx,
        effective_config,
        batch_type,
        token_pairs,
        xp_config,
        save_dir,
        train_size=train_size,
        token_pairs_banned_path=token_pairs_banned_path,
        model_groups_config=xp_config.get("model_groups"),
    )
    multi_classifier.batch_classification()
    return multi_classifier.get_raw_summary_results(), getattr(multi_classifier, "openset_roc_data", None)


def _replot_batch_type_step(
    new_var_models_idx,
    effective_config: Dict[str, Any],
    batch_type: str,
    token_pairs: List[str],
    xp_config: Dict,
    save_dir: Path,
    train_size: int,
    token_pairs_banned_path: Path,
) -> None:
    """Re-run only the openset plot tail using cached on-disk results.

    Uses zero-row stub feature matrices since the plot path never touches
    `X_s`. Silently no-ops if the openset cache pickles are missing
    (legacy XPs).
    """
    X_s_stub = {tp: np.zeros((0, 1)) for tp in token_pairs}
    multi = MultiTokenPairClassification(
        X_s_stub,
        {},
        new_var_models_idx,
        effective_config,
        batch_type,
        token_pairs,
        xp_config,
        save_dir,
        train_size=train_size,
        token_pairs_banned_path=token_pairs_banned_path,
        model_groups_config=xp_config.get("model_groups"),
    )
    OpenSetClassification(multi).replot_from_cache()


def _load_or_run_train_size_item(
    ctx,
    Experiment_config: Dict[str, Any],
    classification_config: Dict[str, Any],
    train_size: int,
    calculation_item: Dict,
    calculations_config: Dict,
    token_pairs: List[str],
    batch_types: List[str],
    utp_values: List,
    train_size_dict: Dict,
    checkpoint_dir: Path,
    new_var_models_idx,
    new_var_models_idx_path: Path,
):
    """Load checkpoint or compute classification for one (train_size, calculation_item) pair.

    Returns updated new_var_models_idx.
    """
    calculation_item_name = get_calculation_item_name(calculations_config, calculation_item)
    train_size_dict[train_size][calculation_item_name] = {}

    # Path for global entries and metadata saved per calculation item
    calculation_item_name_checkpoint_dir = checkpoint_dir / calculation_item_name
    calculation_item_name_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_size_save_dir = calculation_item_name_checkpoint_dir / str(train_size)
    train_size_save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = train_size_save_dir / f"train_size{train_size}_{calculation_item_name}.pkl"
    completed_steps_path = train_size_save_dir / "completed_steps.pkl"

    # --- Checkpoint loading ---
    if checkpoint_path.exists():
        logger.info("Loading checkpoint for train_size=%s, item=%s", train_size, calculation_item_name)
        summary_results = joblib.load(checkpoint_path)
        logger.debug("Checkpoint loaded at %s", checkpoint_path)
        train_size_dict[train_size][calculation_item_name] = summary_results

        if new_var_models_idx_path.exists():
            loaded_new_var_models_idx = load_json(path=new_var_models_idx_path, set_keys_as_int=True)
            if new_var_models_idx is None:
                new_var_models_idx = loaded_new_var_models_idx
            elif new_var_models_idx != loaded_new_var_models_idx:
                raise ValueError(
                    f"Inconsistent new_var_models_idx: existing differs from loaded for {calculation_item_name}"
                )

        # Load step tracking for incremental resumption
        if completed_steps_path.exists():
            completed_steps = set(joblib.load(completed_steps_path))
        else:
            # Old format checkpoint (no step tracking) — treat as fully complete
            completed_steps = None

        def _do_replot_for_all_steps() -> None:
            """Regenerate openset figures for every (batch_type, utp) combo from cache."""
            if not classification_config.get("openset", False):
                return
            for bt in batch_types:
                if bt in ("mix_tp_at_pred", "mix_tp_at_train"):
                    for uv in utp_values:
                        cfg_for_uv = {**classification_config, "unique_tp_in_mix": uv}
                        _replot_batch_type_step(
                            new_var_models_idx, cfg_for_uv, bt, token_pairs,
                            ctx.xp_config, train_size_save_dir, train_size,
                            ctx.token_pairs_banned_path,
                        )
                else:
                    _replot_batch_type_step(
                        new_var_models_idx, classification_config, bt, token_pairs,
                        ctx.xp_config, train_size_save_dir, train_size,
                        ctx.token_pairs_banned_path,
                    )

        if completed_steps is None:
            _do_replot_for_all_steps()
            return new_var_models_idx, None  # Old format, fully complete

        # Compute expected steps to check if all are done
        all_expected_steps = set()
        for bt in batch_types:
            if bt in ("mix_tp_at_pred", "mix_tp_at_train"):
                for uv in utp_values:
                    all_expected_steps.add(f"{bt}_utp{uv}")
            else:
                all_expected_steps.add(bt)

        if completed_steps >= all_expected_steps:
            _do_replot_for_all_steps()
            return new_var_models_idx, None  # All steps done

        logger.info(
            "Partial checkpoint for train_size=%s, item=%s: %d/%d steps done, resuming.",
            train_size, calculation_item_name, len(completed_steps), len(all_expected_steps),
        )
    else:
        completed_steps = set()
        logger.info(
            "No checkpoint found for train_size=%s, item=%s, computing pipeline.",
            train_size, calculation_item_name,
        )

    # --- Data preparation ---
    X_s, remapped_features_index, current_new_var_models_idx = _prepare_all_token_pair_features(
        token_pairs, calculation_item, ctx, Experiment_config,
    )

    if new_var_models_idx is None:
        new_var_models_idx = current_new_var_models_idx
    elif new_var_models_idx != current_new_var_models_idx:
        raise ValueError(
            f"Inconsistent new_var_models_idx: existing differs from computed for {calculation_item_name}"
        )

    logger.info("=" * 60)
    logger.info("Training size %s, calculation_item=%s", train_size, calculation_item_name)
    logger.info("=" * 60)

    # --- Batch-type routing ---
    openset_roc_data = None
    for batch_type in batch_types:
        if batch_type in ("mix_tp_at_pred", "mix_tp_at_train"):
            for utp_idx, utp_val in enumerate(utp_values):
                step_key = f"{batch_type}_utp{utp_val}"
                if step_key in completed_steps:
                    logger.info("Skipping %s, already completed.", step_key)
                    if classification_config.get("openset", False):
                        cfg_for_uv = {**classification_config, "unique_tp_in_mix": utp_val}
                        _replot_batch_type_step(
                            new_var_models_idx, cfg_for_uv, batch_type, token_pairs,
                            ctx.xp_config, train_size_save_dir, train_size,
                            ctx.token_pairs_banned_path,
                        )
                    continue

                config_for_utp = {**classification_config, "unique_tp_in_mix": utp_val}
                logger.info(
                    "Running batch_type=%s classification for train_size=%s, "
                    "unique_tp_in_mix=%s, calculation_item=%s",
                    batch_type, train_size, utp_val, calculation_item_name,
                )
                summary_results, roc_d = _run_batch_type_step(
                    X_s, remapped_features_index, new_var_models_idx,
                    config_for_utp, batch_type, token_pairs, ctx.xp_config,
                    train_size_save_dir, train_size, ctx.token_pairs_banned_path,
                )
                if roc_d is not None:
                    openset_roc_data = roc_d
                _log_batch_summary(summary_results, train_size, batch_type, label=step_key)

                # Store under utp-specific key
                utp_summary = {f"{batch_type}_utp{utp_val}": summary_results[batch_type]}
                train_size_dict[train_size][calculation_item_name].update(utp_summary)

                # Backward compat: store first value under batch_type key
                if utp_idx == 0:
                    train_size_dict[train_size][calculation_item_name].update(summary_results)

                _save_incremental_checkpoint(
                    step_key, completed_steps,
                    train_size_dict[train_size][calculation_item_name],
                    checkpoint_path, completed_steps_path,
                )
        else:
            step_key = batch_type
            if step_key in completed_steps:
                logger.info("Skipping %s, already completed.", step_key)
                if classification_config.get("openset", False):
                    _replot_batch_type_step(
                        new_var_models_idx, classification_config, batch_type, token_pairs,
                        ctx.xp_config, train_size_save_dir, train_size,
                        ctx.token_pairs_banned_path,
                    )
                continue

            logger.info(
                "Running batch_type=%s classification for train_size=%s, calculation_item=%s",
                batch_type, train_size, calculation_item_name,
            )
            summary_results, roc_d = _run_batch_type_step(
                X_s, remapped_features_index, new_var_models_idx,
                classification_config, batch_type, token_pairs, ctx.xp_config,
                train_size_save_dir, train_size, ctx.token_pairs_banned_path,
            )
            if roc_d is not None:
                openset_roc_data = roc_d
            _log_batch_summary(summary_results, train_size, batch_type)

            # Accumulate results across all batch_types
            train_size_dict[train_size][calculation_item_name].update(summary_results)

            _save_incremental_checkpoint(
                step_key, completed_steps,
                train_size_dict[train_size][calculation_item_name],
                checkpoint_path, completed_steps_path,
            )

    # --- DCA showcase ---
    if ctx.xp_config.get("save_dca_showcase_data", False):
        full_var_model_name_to_full_safe_var_model_name_map = (
            full_var_model_name_to_full_safe_var_model_name_mapper(new_var_models_idx)
        )
        logger.debug("full_var_model_name_to_full_safe_var_model_name_map = %s", full_var_model_name_to_full_safe_var_model_name_map)
        compute_dca_showcase_data(
            train_size_dict, classification_config, full_var_model_name_to_full_safe_var_model_name_map
        )
        logger.info(
            "DCA showcase data computed and added to train_size_dict for train_size=%s, calculation_item=%s",
            train_size, calculation_item_name,
        )

    write_json(new_var_models_idx, new_var_models_idx_path)

    return new_var_models_idx, openset_roc_data


def batch_classification_across_token_pairs(Experiment_config: Dict[str, Any]) -> None:
    """Multi-token-pair batch classification across training sizes."""
    ctx = prepare_experiment_context(Experiment_config)

    classification_config = ctx.xp_config["classification_config"]

    if classification_config.get("openset", False):
        save_fig_path = Path(ctx.save_fig_path) / "OpenSet"
    else:
        save_fig_path = Path(ctx.save_fig_path) / "ClosedSet"

    # Alpha-dependent outputs go under alpha_<value>/ subdirectory for openset
    if classification_config.get("openset", False):
        alpha_subdir = f"alpha_{_alpha_str(classification_config.get('alpha_quantile_threshold', 0.05))}"
        downstream_save_fig_path = save_fig_path / alpha_subdir
        downstream_save_fig_path.mkdir(parents=True, exist_ok=True)
    else:
        downstream_save_fig_path = save_fig_path

    checkpoint_dir = Path(downstream_save_fig_path) / "train_size_checkpoints"

    if checkpoint_dir.exists() and not classification_config.get("use_checkpoint", True):
        logger.info("Removing existing checkpoint directory: %s", checkpoint_dir)
        import shutil
        shutil.rmtree(checkpoint_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_size_dict: Dict = {}
    new_var_models_idx = None
    new_var_models_idx_path = ctx.checkpoint_dir_path / "new_var_models_idx.json"

    dataset_iter_idx = get_iter_idx_from_calculations_config(iterator_name="token_pairs", xp_config=ctx.xp_config)
    token_pairs = ctx.calculations_iter_lists[dataset_iter_idx]
    calculation_items_list = dict_product_with_fix_item(ctx.calculations_iter_lists, fix_iterator_idx=dataset_iter_idx)
    calculations_config = ctx.xp_config["calculations"]

    # Normalize unique_tp_in_mix to a list (needed early for checkpoint step tracking)
    utp_raw = classification_config.get("unique_tp_in_mix")
    if isinstance(utp_raw, int):
        utp_values = [utp_raw]
    elif isinstance(utp_raw, list):
        utp_values = utp_raw
    else:
        utp_values = [None]

    batch_types = classification_config.get("batch_types") or ["tp_wise"]

    # --- Main pipeline: iterate over train sizes and calculation items ---
    collected_openset_roc_data = None
    for train_size in ctx.xp_config["train_sizes"]:
        train_size_dict[train_size] = {}
        for calculation_item in calculation_items_list:
            new_var_models_idx, roc_d = _load_or_run_train_size_item(
                ctx, Experiment_config, classification_config,
                train_size, calculation_item, calculations_config,
                token_pairs, batch_types, utp_values,
                train_size_dict, checkpoint_dir,
                new_var_models_idx, new_var_models_idx_path,
            )
            if roc_d is not None:
                collected_openset_roc_data = roc_d

    # --- Post-processing and visualization ---
    logger.info("Model idx:")
    for model_idx, model_name in new_var_models_idx.items():  # type: ignore
        logger.debug("%s: %s", model_idx, model_name)

    if ctx.xp_config.get("save_dca_showcase_data", False):
        plot_dca_showcase_by_original_model(
            train_size_dict,
            classification_config,
            downstream_save_fig_path,
        )

    plot_trainsize_wise_curves(
        train_size_dict,
        downstream_save_fig_path,
        classification_config,
        batch_sizes=[bs for bs in classification_config["batch_prediction_sizes"] if bs <= 8],
        datasets=token_pairs,
        models_idx=new_var_models_idx,
    )

    # Merged-wrapper mode (train_size_dict_map with N sources): only F01 from
    # plot_trainsize_wise_curves is meaningful. Skip the rest of the pipeline (train-size
    # sweep, CM heatmaps, model-wise tables, etc.) — they iterate the main source with the
    # wrapper's token_pairs pool, which mismatches each source XP's saved uplet keys.
    if classification_config.get("train_size_dict_map"):
        logger.info("train_size_dict_map set: F01 produced; skipping downstream per-source plots.")
        return

    plot_train_size_curves(
        train_size_dict,
        downstream_save_fig_path,
        classification_config,
        [
            bs for bs in classification_config["batch_prediction_sizes"] if bs != 1
        ],  # batch_sizes=[1,2,3,5,8], # classification_config['batch_prediction_sizes'],
        token_pairs=token_pairs,
    )

    compute_cm = classification_config.get("compute_confusion_matrices", False)

    has_model_groups = ctx.xp_config.get("model_groups") is not None

    if compute_cm:
        # --- Confusion matrix heatmaps ---
        cm_batch_sizes = sorted(
            {1, max(classification_config["batch_prediction_sizes"])}
            & set(classification_config["batch_prediction_sizes"])
        )
        plot_confusion_matrices_on_tr_size_dict(
            train_size_dict,
            downstream_save_fig_path,
            ctx.xp_config,
            classification_config,
            datasets=token_pairs,
            models_idx=new_var_models_idx,
            batch_sizes=cm_batch_sizes,
        )

        # --- Variation heatmaps (only with model_variations and without model_groups) ---
        if ctx.xp_config.get("model_variations") and not has_model_groups:
            train_sizes = sorted(train_size_dict.keys())
            calc_items_names = sorted({t for ts in train_size_dict.values() for t in ts.keys()})
            for train_size in train_sizes:
                for calc_name in calc_items_names:
                    summary = train_size_dict[train_size][calc_name]
                    effective_keys = get_effective_batch_type_keys(summary, batch_types)
                    for clf in classification_config["classifiers"]:
                        for effective_key in effective_keys:
                            var_hm_path = downstream_save_fig_path / calc_name / clf / str(train_size) / "VariationHeatmaps" / effective_key
                            var_hm_path.mkdir(parents=True, exist_ok=True)
                            for tp_group in ["FLiPS", "0-1"]:
                                try:
                                    create_confusion_matrix_heatmaps(
                                        summary,
                                        new_var_models_idx,
                                        classification_config["batch_prediction_sizes"],
                                        clf,
                                        tp_group,
                                        var_hm_path,
                                        effective_key,
                                        openset=classification_config.get("openset", False),
                                        target_bs=max(cm_batch_sizes),
                                        token_pairs=token_pairs,
                                    )
                                except Exception as e:
                                    logger.warning(
                                        "Skipping variation heatmap for %s/%s/%s: %s",
                                        tp_group, clf, effective_key, e,
                                    )
    else:
        logger.info("Confusion matrix heatmaps skipped (compute_confusion_matrices=false)")

    # --- Per-class tables (always; skip NxM when model_groups) ---
    make_model_wise_tables(
        train_size_dict,
        downstream_save_fig_path,
        classification_config,
        datasets=token_pairs,
        models_idx=new_var_models_idx,
        batch_sizes=[bs for bs in classification_config["batch_prediction_sizes"] if bs <= 8],
        skip_nxm=has_model_groups,
        openset_roc_data=collected_openset_roc_data,
    )
