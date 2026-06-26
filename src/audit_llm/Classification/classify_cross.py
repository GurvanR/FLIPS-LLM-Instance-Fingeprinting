"""Cross-token-pair NxN classification orchestration.

Top-level ``classify_cross_token_pairs()`` entry point that trains on one
token pair and tests on another, producing NxN heatmaps of cross-generalization.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from audit_llm.Classification.results_plotting import (
    plot_cross_token_pairs_heatmaps,
    plot_cross_token_pairs_heatmaps_multi,
)
from audit_llm.Classification.single_classification import (
    SingleTokenPairClassification,
    persist_token_pair_ban,
)
from audit_llm.xp_tools import (
    aggregate_results_dict,
    dict_product_with_fix_item,
    get_iter_idx_from_calculations_config,
    get_token_pairs_of_group,
    load_classification_checkpoint,
    prepare_dataset_features,
    prepare_experiment_context,
    remap_features_index,
    save_classification_checkpoint,
    select_features_for_token_pair,
)

logger = logging.getLogger(__name__)


def _evaluate_cross_pair(
    i: int,
    j: int,
    pipeline: SingleTokenPairClassification,
    X_train: np.ndarray,
    cross_calculation_items_test: Dict,
    dataset_iter_idx: str,
    selected_features_for_classification: Dict,
    ctx,
    Experiment_config: Dict[str, Any],
    completed_pairs: set,
    results_dict: Dict,
    confusion_matrices_dict: Dict,
    item_checkpoint_path: Path,
) -> None:
    """Evaluate one (train, test) pair in cross-token-pair classification."""
    pair_id = (i, j)
    if pair_id in completed_pairs:
        logger.info("Skipping %s, already completed.", pair_id)
        return

    # Run evaluation
    if i == j:
        pipeline.fit_evaluate(X_train)
    else:
        dataset = cross_calculation_items_test[dataset_iter_idx]

        X_test, _ = prepare_dataset_features(
            dataset,
            cross_calculation_items_test,
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

        pipeline.fit_evaluate(X_train, X_test)

    if pipeline.dataset_banned:
        persist_token_pair_ban(pipeline.token_pair, ctx.token_pairs_banned_path)
        return

    summary = pipeline.get_raw_summary_results()
    confusion_matrices = pipeline.get_confusion_matrices()

    for clf_name, clf_summary in summary.items():
        acc_mean = clf_summary.get("accuracy_mean", float("nan"))
        logger.debug(
            "Cross pair (%d, %d) — %s — accuracy: %.4f",
            i, j, clf_name, acc_mean,
        )

    results_dict[pair_id] = summary
    confusion_matrices_dict[pair_id] = confusion_matrices

    completed_pairs.add(pair_id)
    save_classification_checkpoint(
        item_checkpoint_path, results_dict, confusion_matrices_dict, completed_pairs, pair_id
    )


def _plot_cross_figures(
    ctx,
    per_for_each_results_dict: Dict,
    per_for_each_confusion_matrices_dict: Dict,
) -> None:
    """Generate cross-token-pair heatmap figures (aggregated and individual)."""
    for figure_idx, figure_config in ctx.xp_config.get("figures", {}).items():
        logger.info("Plotting figure(s): %s", figure_idx)
        personalized_fig_save_path = Path(ctx.save_fig_path) / str(figure_idx)

        repeat_for_each_iterator = figure_config["repeat_for_each"]
        if repeat_for_each_iterator != "for_each":
            raise NotImplementedError()

        aggregation = figure_config.get("aggregation", "none")
        if aggregation == "none":
            repeat_for_each_list = ctx.calculations_iter_lists[repeat_for_each_iterator]
        else:
            placeholder_repeat_for_each_item = aggregation
            repeat_for_each_list = [placeholder_repeat_for_each_item]

            per_for_each_results_dict[placeholder_repeat_for_each_item] = aggregate_results_dict(
                per_for_each_results_dict, mode="results", aggregation=aggregation
            )
            per_for_each_confusion_matrices_dict[placeholder_repeat_for_each_item] = aggregate_results_dict(
                per_for_each_confusion_matrices_dict, mode="confusion_matrix", aggregation=aggregation
            )

            logger.debug("per_for_each_results_dict[%s] = %s", placeholder_repeat_for_each_item, per_for_each_results_dict[placeholder_repeat_for_each_item])

        if aggregation == "none":
            logger.debug("repeat_for_each_list = %s", repeat_for_each_list)
            plot_cross_token_pairs_heatmaps_multi(
                per_for_each_results_dict,
                repeat_for_each_list,
                repeat_for_each_iterator,
                ctx.calculations_iter_lists,
                ctx.xp_config,
                figure_config,
                save_fig_path=personalized_fig_save_path,
            )

        for repeat_for_each_item in repeat_for_each_list:
            logger.info("Processing %s: %s", repeat_for_each_iterator, repeat_for_each_item)
            classifier_heatmap_path = Path(personalized_fig_save_path) / repeat_for_each_iterator / repeat_for_each_item
            classifier_heatmap_path.mkdir(parents=True, exist_ok=True)
            results_dict = per_for_each_results_dict[repeat_for_each_item]
            cross_calculation_items_list = dict_product_with_fix_item(
                ctx.calculations_iter_lists, fix_iterator_idx=repeat_for_each_iterator, fix_item=repeat_for_each_item
            )

            plot_cross_token_pairs_heatmaps(
                results_dict,
                cross_calculation_items_list,
                ctx.xp_config,
                figure_config,
                save_fig_path=classifier_heatmap_path,
            )

            # TODO: Plot per-model transfer bar-plot
            # TODO: Plot per-class accuracy from confusion matrices


def classify_cross_token_pairs(Experiment_config: Dict[str, Any]) -> None:
    """Cross-token-pair NxN classification with per-pair checkpointing."""
    ctx = prepare_experiment_context(Experiment_config)

    dataset_iter_idx = get_iter_idx_from_calculations_config(iterator_name="token_pairs", xp_config=ctx.xp_config)

    # Taking N token pairs of Monochar and N of FLiPS
    max_nb_of_token_pairs_computed_per_group = ctx.xp_config.get("max_nb_of_token_pairs_computed_per_group", "none")
    if max_nb_of_token_pairs_computed_per_group == "none":
        max_nb_of_token_pairs_computed_per_group = 100_000
    datasets = ctx.calculations_iter_lists[dataset_iter_idx]
    logger.debug("Before: datasets = %s", datasets)
    Monochar_ds = get_token_pairs_of_group(group="Monochar", token_pairs=datasets)[
        :max_nb_of_token_pairs_computed_per_group
    ]
    FLiPS_ds = get_token_pairs_of_group(group="FLiPS", token_pairs=datasets)[:max_nb_of_token_pairs_computed_per_group]
    gr_01_ds = get_token_pairs_of_group(group="0-1", token_pairs=datasets)
    datasets_for_cross = Monochar_ds + FLiPS_ds + gr_01_ds
    ctx.calculations_iter_lists[dataset_iter_idx] = datasets_for_cross
    logger.debug("After: datasets_for_cross = %s", datasets_for_cross)

    for_each_list = ctx.calculations_iter_lists["for_each"]
    per_for_each_results_dict = {}
    per_for_each_confusion_matrices_dict = {}

    for each_item in for_each_list:
        item_checkpoint_path = Path(ctx.checkpoint_dir_path) / f"{str(each_item)}_checkpoint.pkl"
        results_dict, confusion_matrices_dict, completed_pairs = load_classification_checkpoint(item_checkpoint_path)  # type: ignore

        cross_calculation_items_list = dict_product_with_fix_item(
            ctx.calculations_iter_lists, fix_iterator_idx="for_each", fix_item=each_item
        )

        selected_features_for_classification = select_features_for_token_pair(
            ctx.intra_samples_feature_index_dict, ctx.xp_config
        )
        remapped_selected_features_for_classification_index = remap_features_index(selected_features_for_classification)

        for i, cross_calculation_items_train in enumerate(cross_calculation_items_list):
            if all((i, j) in completed_pairs for j in range(len(cross_calculation_items_list))):
                logger.info("Skipping all pairs with train idx %d, already completed.", i)
                continue

            dataset = cross_calculation_items_train[dataset_iter_idx]
            logger.debug("cross_calculation_items_train = %s", cross_calculation_items_train)
            X_train, _ = prepare_dataset_features(
                dataset, cross_calculation_items_train,
                ctx.intra_samples_features_dict, ctx.intra_samples_feature_index_dict,
                selected_features_for_classification, ctx.new_var_models_idx,
                ctx.models_indices, ctx.xp_config, Experiment_config,
                ctx.MainDataset_df_iterators, ctx.Answers_df,
            )

            pipeline = SingleTokenPairClassification(
                remapped_selected_features_for_classification_index,
                ctx.new_var_models_idx, ctx.xp_config["classification_config"],
                ctx.xp_config, dataset, ctx.token_pairs_banned_path,
                model_groups_config=ctx.xp_config.get("model_groups"),
            )
            pipeline.cross_classif = True

            for j, cross_calculation_items_test in enumerate(cross_calculation_items_list):
                _evaluate_cross_pair(
                    i, j, pipeline, X_train, cross_calculation_items_test,
                    dataset_iter_idx, selected_features_for_classification, ctx,
                    Experiment_config, completed_pairs, results_dict,
                    confusion_matrices_dict, item_checkpoint_path,
                )

        per_for_each_results_dict[each_item] = results_dict
        per_for_each_confusion_matrices_dict[each_item] = confusion_matrices_dict

    _plot_cross_figures(ctx, per_for_each_results_dict, per_for_each_confusion_matrices_dict)
