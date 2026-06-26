"""Single token-pair classification orchestration.

Top-level ``classify()`` entry point that wires experiment config to
``SingleTokenPairClassification`` pipelines, manages checkpoints, and
generates post-classification figures.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

from audit_llm.Classification.confusion_matrix_utils import plot_confusion_matrices
from audit_llm.Classification.single_classification import (
    SingleTokenPairClassification,
    persist_token_pair_ban,
)
from audit_llm.data_transforms import nested_loop
from audit_llm.plotting.perf_curves import generate_personalized_figures
from audit_llm.xp_tools import (
    get_calculation_item_name,
    get_iter_idx_from_calculations_config,
    load_results_checkpoint,
    prepare_dataset_features,
    prepare_experiment_context,
    remap_features_index,
    select_features_for_token_pair,
)

logger = logging.getLogger(__name__)


# TRAIN_SIZE_FOR_XP=[40, 80]
def classify(Experiment_config: Dict[str, Any]) -> None:
    # Get ExperimentContext (or FLiPSExperimentContext) from Experiment_config
    ctx = prepare_experiment_context(Experiment_config)


    calculations_config = ctx.xp_config["calculations"]

    results_dict = defaultdict(dict)

    new_var_models_idx = None
    new_var_models_idx_path = ctx.checkpoint_dir_path / "new_var_models_idx.json"

    def classify_under_loop(calculation_item):
        """
        calculation_item: Dict[iterator_idx: iterator_item]
        e.g.
        - if iterator_idx corresponds to token_pairs, iterator_item is a token_pair
        - if iterator_idx corresponds to temperature, iterator_item is a temperature
        """
        nonlocal new_var_models_idx

        # Checkpoint_path
        calculation_item_name = get_calculation_item_name(calculations_config, calculation_item)
        assert calculation_item_name != ""
        checkpoint_path = ctx.checkpoint_dir_path / f"results_dict_{calculation_item_name}.pkl"

        # Loading Checkpoint
        if checkpoint_path.exists():
            logger.info("Loading checkpoint for: %s", calculation_item_name)
            load_results_checkpoint(results_dict, checkpoint_path, calculation_item_name)

            if new_var_models_idx_path.exists():
                loaded_new_var_models_idx = joblib.load(new_var_models_idx_path)
                if new_var_models_idx is None:
                    new_var_models_idx = loaded_new_var_models_idx
                elif new_var_models_idx != loaded_new_var_models_idx:
                    raise ValueError(
                        f"Inconsistent new_var_models_idx: existing differs from loaded for {calculation_item_name}"
                    )
        else:
            logger.info("No checkpoint found for: %s, computing pipeline.", calculation_item_name)
            # Gathering data on which classify:
            dataset_iter_idx = get_iter_idx_from_calculations_config(iterator_name="token_pairs", xp_config=ctx.xp_config)
            dataset = calculation_item[dataset_iter_idx]

            selected_features_for_classification = select_features_for_token_pair(  # features_index depends on dataset (i.e. token pairs) because there are some features linked to token stats.
                ctx.intra_samples_feature_index_dict, ctx.xp_config
            )  # Dict[feat_name: feat_idx]
            logger.debug("sanity check that only _ts values and no _pv: %s", selected_features_for_classification)
            remapped_selected_features_for_classification_index = remap_features_index(
                selected_features_for_classification
            )

            X, current_new_var_models_idx = prepare_dataset_features(
                dataset,
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

            if new_var_models_idx is None:
                new_var_models_idx = current_new_var_models_idx
            elif new_var_models_idx != current_new_var_models_idx:
                raise ValueError(
                    f"Inconsistent new_var_models_idx: existing differs from computed for {calculation_item_name}"
                )

            pipeline = SingleTokenPairClassification(
                remapped_selected_features_for_classification_index,
                new_var_models_idx,
                ctx.xp_config["classification_config"],
                ctx.xp_config,
                dataset,
                ctx.token_pairs_banned_path,
                model_groups_config=ctx.xp_config.get("model_groups"),
            )

            results = pipeline.fit_evaluate(X)

            if pipeline.dataset_banned:
                persist_token_pair_ban(pipeline.token_pair, ctx.token_pairs_banned_path)

            # Save pipeline and new_var_models_idx for reuse
            summary = pipeline.get_raw_summary_results()
            confusion_matrices = pipeline.get_confusion_matrices()

            for clf_name, clf_summary in summary.items():
                acc_mean = clf_summary.get("accuracy_mean", float("nan"))
                acc_std = clf_summary.get("accuracy_std", float("nan"))
                logger.info(
                    "[%s] %s — accuracy: %.4f ± %.4f",
                    calculation_item_name, clf_name, acc_mean, acc_std,
                )

            results_dict[calculation_item_name]["summary"] = summary
            results_dict[calculation_item_name]["confusion_matrices"] = confusion_matrices

            joblib.dump(results_dict[calculation_item_name], checkpoint_path)
            joblib.dump(new_var_models_idx, new_var_models_idx_path)
            logger.info("Checkpoint saved at %s", checkpoint_path)
            logger.debug("new_var_models_idx saved at %s", new_var_models_idx_path)

    nested_loop(ctx.calculations_iter_lists, classify_under_loop)

    if new_var_models_idx is None:
        raise ValueError("No new_var_models_idx was set during processing")
    #  display new_var_models_idx
    for model_idx, model_name in new_var_models_idx.items():  # type: ignore
        logger.debug("%s: %s", model_idx, model_name)

    conf_mats_save_path = Path(ctx.save_fig_path) / "Confusion_Matrices"
    conf_mats_save_path.mkdir(parents=True, exist_ok=True)
    confusion_matrices_dict: Dict[str, Dict[str, List[np.ndarray]]] = {
        key: value["confusion_matrices"] for key, value in results_dict.items()
    }
    plot_confusion_matrices(ctx.xp_config, confusion_matrices_dict, new_var_models_idx, conf_mats_save_path)

    pipe_summary_dict: Dict[str, Dict[str, Any]] = {key: value["summary"] for key, value in results_dict.items()}
    generate_personalized_figures(
        ctx.xp_config,
        calculations_config,
        ctx.calculations_iter_lists,
        new_var_models_idx,
        ctx.save_fig_path,
        pipe_summary_dict,
        pipe_summary_mode=True,
    )

    # Default Figure plots # TODO OR NOT ? maybe not for the moment
    # plot_features_importance(stored_pipelines[temperature], classification_config, temp_saving_path, SHOW) # TODO
    # plot_performance_of_best_classifiers(stored_pipelines[temperature], temp_saving_path, SHOW)  # TODO
