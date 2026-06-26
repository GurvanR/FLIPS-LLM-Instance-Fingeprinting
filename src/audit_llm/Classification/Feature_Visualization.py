import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import polars as pl
from tqdm import tqdm

from pathlib import Path

from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)

from audit_llm.data_transforms import nested_loop, revert_dictionary
from audit_llm.file_io import open_pickle_file, save_pickle_file
from audit_llm.plot_configs import *
from audit_llm.plotting.figure_io import save_fig_and_show
from audit_llm.plotting.perf_curves import generate_personalized_figures
from audit_llm.xp_tools import (
    dict_product_with_fix_item,
    get_calculation_item_name,
    get_iter_idx_from_calculations_config,
    prepare_dataset_features,
    prepare_experiment_context,
    remap_features_index,
    remap_model_index,
    select_features_for_token_pair,
)
from audit_llm.xp_tools.model_filtering import (
    full_var_model_name_to_original_model_name,
    group_models_idx_by_var_or_orig,
    truncate_model_name,
)

from audit_llm.Classification.compare_nist_and_clf_acc import compare_nist_perf_with_classsif_accuracy

def Seq_Length_visualization(Experiment_config: Dict[str, Any]) -> None:
    """
    Visualize sequence length distribution across datasets and models.
    Updated to work with 3D intra_samples_features_dict structure.
    """
    
    # Extract configuration parameters
    model_idx = Experiment_config['model_idx']
    datasets = Experiment_config['datasets']

    plot_all_datasets = Experiment_config['xp_config'].get('plot_all_datasets', False)
    
    # Get ExperimentContext (or FLiPSExperimentContext) from Experiment_config
    ctx = prepare_experiment_context(Experiment_config)
    
    
    remapped_model_index = remap_model_index(model_idx, ctx.models_indices)
    xp_min_length_threshold = 100
    
    # Get temperature index if needed
    temp_idx = 0  # Default, adjust based on your experiment configuration
    if 'temperature' in ctx.xp_config.get('calculations', {}):
        # You may need to adjust this based on how you want to select temperature
        temp_idx = 0
        
    # Create one plot per temperature combining all datasets
    plt.figure(figsize=(12, 8))
    
    # Collect max values across all datasets
    max_vals = []
    for dataset in datasets:
        seq_length_idx = ctx.intra_samples_feature_index_dict[dataset]['seq_length']
        # Shape: (N_samples, N_models, N_features)
        X = ctx.intra_samples_features_dict[dataset]
        data_slice = X[:, :, seq_length_idx]  # Shape: (N_samples, N_models)
        max_val = np.nanmax(data_slice)
        if not np.isnan(max_val):
            max_vals.append(max_val)

    # Get overall maximum, with fallback to 1000
    overall_max = max(max_vals) if max_vals else 1000
    x_vals = np.linspace(xp_min_length_threshold, min(overall_max, 1000), 500)
    
    # To accumulate CCDFs for averaging
    ccdf_accumulator = {}  # Key: model index, Value: list of CCDF arrays
    
    # Use a colormap for better color distribution when many models
    n_models = len(ctx.models_indices)
    colors = cm.tab20(np.linspace(0, 1, n_models))  # type: ignore
    
    for dataset in datasets:
        # Shape: (N_samples, N_models, N_features)
        X = ctx.intra_samples_features_dict[dataset]
        seq_length_idx = ctx.intra_samples_feature_index_dict[dataset]['seq_length']
        
        # Extract sequence length data: shape (N_samples, N_models)
        data = X[:, :, seq_length_idx]
        
        # Filter by models if needed
        data = data[:, ctx.models_indices]
        
        n_samples, n_models_filtered = data.shape
        logger.debug(f"Dataset: {dataset}, Shape: {data.shape}")

        for m in range(n_models_filtered):
            values = data[:, m]
            valid_values = values[~np.isnan(values)]
            
            if len(valid_values) == 0:
                logger.warning(f"Dataset {dataset}, Model {m} has no valid values (all NaN)")
                continue
                
            sorted_vals = np.sort(valid_values)
            ccdf = 1.0 - (np.searchsorted(sorted_vals, x_vals, side='left') / len(valid_values))
            
            # Store CCDF for averaging later
            ccdf_accumulator.setdefault(m, []).append(ccdf)
            
            # Plot individual dataset CCDF with assigned color
            if plot_all_datasets:
                model_name = remapped_model_index[m]
                plt.plot(x_vals, ccdf, 
                        label=f"{truncate_model_name(model_name, k=40)}", 
                        alpha=0.7, color=colors[m], linewidth=0.8)
        
        if plot_all_datasets:
            # Add vertical line MaxTokens_param = 750
            MaxTokens_param = 750
            plt.axvline(x=MaxTokens_param, color='black', linestyle='--', linewidth=1)
            plt.text(MaxTokens_param, plt.ylim()[1]*0.25, f'MaxTokens = {MaxTokens_param}', 
                    rotation=90, va='top', ha='right', fontsize=9, color='gray')
            
            
            plt.xlabel('Sequence Length', fontsize=10)
            plt.ylabel('Proportion of Samples ≥ Sequence Length', fontsize=10)

            # Adaptive legend handling for many models
            if n_models_filtered <= 10:
                plt.legend(loc='upper right', fontsize=8)
            elif n_models_filtered <= 20:
                plt.legend(loc='upper right', fontsize=7, ncol=1)
            elif n_models_filtered <= 30:
                plt.legend(loc='upper right', fontsize=7, ncol=2)
            else:
                plt.legend(loc='upper right', fontsize=7, ncol=2)

            plt.grid(True, alpha=0.3)
            plt.tick_params(axis='both', which='major', labelsize=9)
            plt.tight_layout()

            # Adjust figure size for better readability with many models
            plt.gcf().set_size_inches(12, 8)

            temp_save_fig_path = ctx.save_fig_path  # Adjust if you have temperature-specific paths
            save_fig_and_show(
                save_path=temp_save_fig_path,
                show=False,
                fig_name=f"{dataset}_seq_length_ccdf.pdf"
            )

    # ---------- Plot Averaged CCDFs ----------
    plt.figure(figsize=(14, 8))

    for m, ccdf_list in ccdf_accumulator.items():
        ccdf_array = np.vstack(ccdf_list)
        avg_ccdf = np.mean(ccdf_array, axis=0)
        model_name = remapped_model_index[m]
        plt.plot(x_vals, avg_ccdf, 
                label=f"{truncate_model_name(model_name, k=40)}", 
                linewidth=1.2, color=colors[m])
    
    # Add vertical lines
    MaxTokens_param = 500
    plt.axvline(x=MaxTokens_param, color='grey', linestyle='--', linewidth=0.5)
    plt.text(MaxTokens_param, plt.ylim()[1]*0.79, f'MaxTokens = {MaxTokens_param}', 
            rotation=270, va='bottom', ha='left', fontsize=11, color='gray')
    
    
    plt.xlabel('Sequence Length', fontsize=20)
    plt.ylabel('Proportion of Samples ≥ Sequence Length', fontsize=17)

    # Adaptive legend for averaged plot
    if n_models_filtered <= 15:
        plt.legend(loc='upper right', fontsize=11)
    elif n_models_filtered <= 25:
        plt.legend(loc='upper right', fontsize=11)
    elif n_models_filtered <= 35:
        plt.legend(loc='upper right', fontsize=11, ncol=2)
    else:
        plt.legend(loc='upper right', fontsize=10, ncol=2)

    plt.xticks(np.arange(xp_min_length_threshold, min(overall_max, 1000) + 1, 100))
    plt.grid(**GRID_CONFIG)
    #setting legend title
    legend = plt.gca().get_legend()
    legend.set_title('LLM', prop={'size': 20})
    plt.tick_params(axis='both', which='major', **XTICKS_CONFIG)
    
    plt.tight_layout()
    
    temp_save_fig_path = ctx.save_fig_path
    save_fig_and_show(
        save_path=temp_save_fig_path,
        show=False,
        fig_name="avg_seq_length_ccdf.pdf"
    )
    
    # Display model index
    for idx, model in remapped_model_index.items():
        logger.debug(f"{idx}: {model}")

def Save_pv_in_parquet(Experiment_config: Dict[str, Any]) -> None:
    """ 
    logprobs_df    
    logger.debug("Column Details:")
    logger.debug("-" * 60)
    logger.debug("  token_pair: The token pair being tested")
    logger.debug("  model_name: Name/ID of the LLM model used")
    logger.debug("  dataset_idx: Index into the original dataset")
    print("  • output_logprobs:      List[dict] - Logprobs for each generation step")
    print("                          Each dict has: logprob_tA, rank_tA, floored_tA,")
    print("                          logprob_tB, rank_tB, floored_tB")
    print("  • prompt_idx:           Index of the prompt template used")
    print("  • temperature:          Sampling temperature parameter")
    print("  • frequency_penalty:    Frequency penalty parameter")
    print("  • system_prompt_idx:    Index of the system prompt used")
    """
    # Get ExperimentContext (or FLiPSExperimentContext) from Experiment_config
    ctx = prepare_experiment_context(Experiment_config)
    

    Analysis_save_path = Path(Experiment_config["Answers_dfPath"]).parent
    logprobs_path = Analysis_save_path / "Logprobs.parquet"

    logprobs_df = pd.read_parquet(logprobs_path)

    selected_features = select_features_for_token_pair(
        ctx.intra_samples_feature_index_dict, ctx.xp_config
    )  # Dict[feature_name -> feat_idx]

    model_idx_reverted = revert_dictionary(Experiment_config["model_idx"])
    model_variations_indices = Experiment_config["model_variations_indices"]

    # ------------------------------------------------------------
    # Ensure feature columns exist
    # ------------------------------------------------------------
    PLACEHOLDER = "__UNSET__"

    for feat in selected_features:
        if feat not in logprobs_df.columns:
            logprobs_df[feat] = PLACEHOLDER

    feature_names = list(selected_features.keys())
    feature_indices = list(selected_features.values())

    # ------------------------------------------------------------
    # Checkpoint setup
    # ------------------------------------------------------------
    checkpoint_dir = Path(ctx.checkpoint_dir_path) / "pv_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Main loop (checkpoint per token_pair)
    # ------------------------------------------------------------
    for token_pair, X in ctx.intra_samples_features_dict.items():
        logger.info(f"Processing token_pair: {token_pair}")
        checkpoint_file = checkpoint_dir / f"{token_pair}.parquet"

        # Resume if checkpoint exists
        if checkpoint_file.exists():
            logger.info(f"Loading checkpoint for {token_pair}")
            token_pair_df = pd.read_parquet(checkpoint_file)
        else:
            token_pair_df = logprobs_df[logprobs_df["token_pair"] == token_pair].copy()

        for model_idx in range(X.shape[1]):
            model_name = model_idx_reverted[model_idx]

            for temp_str, sample_indices in model_variations_indices.items():
                temperature = float(temp_str.split("-")[-1])

                for sample_idx in sample_indices:
                    mask = (
                        (token_pair_df["model_name"] == model_name)
                        & (token_pair_df["temperature"] == temperature)
                        & (token_pair_df["dataset_idx"] == sample_idx)
                    )

                    if not mask.any():
                        continue

                    # Vectorized assignment
                    token_pair_df.loc[mask, feature_names] = X[
                        sample_idx, model_idx, feature_indices
                    ]

        # --------------------------------------------------------
        # Save checkpoint for this token_pair
        # --------------------------------------------------------
        token_pair_df.to_parquet(checkpoint_file, index=False)
        logger.info(f"Checkpoint saved: {checkpoint_file}")

        # Update main df
        logprobs_df.loc[
            logprobs_df["token_pair"] == token_pair, feature_names
        ] = token_pair_df[feature_names].values

    # ------------------------------------------------------------
    # Final integrity check
    # ------------------------------------------------------------
    placeholder_mask = (logprobs_df[feature_names] == PLACEHOLDER)

    if placeholder_mask.any().any():
        bad_rows = logprobs_df.loc[
            placeholder_mask.any(axis=1),
            ["token_pair", "model_name", "temperature", "dataset_idx"]
        ]
        raise ValueError(
            f"Not all feature values were set.\n"
            f"Affected rows: {len(bad_rows)}\n"
            f"Sample rows:\n{bad_rows.head(10)}"
        )
    # ------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------
    final_path = Analysis_save_path / "Logprobs_with_pv.parquet"
    logprobs_df.to_parquet(final_path, index=False)
    logger.info(f"Final parquet saved to: {final_path}")

def Nist_perf_chart(Experiment_config: Dict[str, Any]) -> None:
    # Get ExperimentContext (or FLiPSExperimentContext) from Experiment_config
    ctx = prepare_experiment_context(Experiment_config)
    

    calculations_config = ctx.xp_config['calculations'] # typically iterator_1: token_pairs; iterator_2: temperature.
    alpha = ctx.xp_config['alpha']
    checkpoint_path = Path(ctx.checkpoint_dir_path) / 'checkpoint.pkl'
    if checkpoint_path.exists():
        logger.info("Loading checkpoint: %s", checkpoint_path)
        nist_perfs_dict = open_pickle_file(checkpoint_path)
    else:
        nist_perfs_dict = {}

    selected_features_for_classification = select_features_for_token_pair( # features_index depends on dataset (i.e. token pairs) because there are some features linked to token stats.
        ctx.intra_samples_feature_index_dict, ctx.xp_config
        )  # Dict[feat_name: feat_idx] 
    remapped_selected_features_for_classification_index = remap_features_index(
        selected_features_for_classification
    )
    dataset_iter_idx = get_iter_idx_from_calculations_config(iterator_name='token_pairs', xp_config=ctx.xp_config)

    checkpoint_count = 0

    def classify_under_loop(calculation_item):
        """
        calculation_item: Dict[iterator_idx: iterator_item]
        e.g. 
        - if iterator_idx corresponds to token_pairs, iterator_item is a token_pair
        - if iterator_idx corresponds to temperature, iterator_item is a temperature
        """
        nonlocal checkpoint_count
        nonlocal nist_perfs_dict
        
        # Checkpoint_path
        calculation_item_name = get_calculation_item_name(calculations_config, calculation_item)
        assert calculation_item_name != ''
        
        # Loading Checkpoint
        if calculation_item_name in nist_perfs_dict:
            return
        else:
            # Gathering data on which classify:    
                
            dataset = calculation_item[dataset_iter_idx]

            # X.shape = (samples, models, features) where features are p-values only.
            X, _ = prepare_dataset_features(
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

            assert X.shape[1]==X.shape[2]==1 # X is supposed to be shape (n_samples, 1, 1)
            X = X.squeeze(axis=1).squeeze(axis=1) # Here X is shape (n_samples, )
            # Here X contains ts and pv values for a specific token_pair (and maybe iterator_2 and iterator_3...).
            # Setting every values of X to 0 if < alpha, 1 else:
            X = np.where(np.isnan(X), np.nan, np.where(X >= alpha, 1, 0)) # nan values remain nan, p-values >= alpha become 1 (not rejected), p-values < alpha become 0 (rejected)
            nist_perfs_dict[calculation_item_name] = np.nanmean(X) # store the proportion of values (that are p-values) that are above alpha (i.e. not rejected). shape (n_models, n_features)
            assert isinstance(nist_perfs_dict[calculation_item_name], (float))
            # saving nist_perfs_dict checkpoint
            checkpoint_count += 1
            if checkpoint_count % 1000 == 0:
                save_pickle_file(nist_perfs_dict, checkpoint_path)
                logger.debug("Saved checkpoint for checkpoint_count=%d: %s", checkpoint_count, calculation_item_name)

    nested_loop(ctx.calculations_iter_lists, classify_under_loop)
    
    # Function that compares with classification accuracy
    compare_nist_perf_with_classsif_accuracy(nist_perfs_dict, ctx.xp_config, ctx.save_fig_path)

    # For nist_perfs_dict
    generate_personalized_figures(
        ctx.xp_config,
        calculations_config,
        ctx.calculations_iter_lists,
        revert_dictionary(ctx.new_var_models_idx),
        ctx.save_fig_path,
        nist_perfs_dict,
        pipe_summary_mode=False
    )

def Valid_count_chart(Experiment_config: Dict[str, Any]) -> None:
    """
    Answers_df header: ['Token_pair', 'Model', 'Dataset_Question Index', 'gen_fail', 'gen_counter', 'Answer', 'prompt_idx', 'temperature', 'frequency_penalty', 'system_prompt_idx']

    gen_counter: int = number of generation done to have a valid sample.
    gen_fail: bool = if true, means that the maximum number of generations has been reached and no valid sample were obtained.
    
    'Token_pair' col corresponds to 'token_pairs'.

    For each calculation item, filter Answers_df with the correpsonding iterator_idx and computes the number of fails in 'fail_count' and average of gen.counter in 'gen_count'.
    (careful, 'Token_pair' col corresponds to 'token_pairs' iterator index, and 'Model' col corresponds to 'models' iterator index)

    """
    # Get ExperimentContext from Experiment_config
    ctx = prepare_experiment_context(Experiment_config)
    

    calculations_config = ctx.xp_config['calculations'] # typically iterator_1: token_pairs; iterator_2: temperature.
    dataset_iter_idx = get_iter_idx_from_calculations_config(iterator_name='token_pairs', xp_config=ctx.xp_config)

    # Checkpoint setup
    checkpoint_path = Path(ctx.checkpoint_dir_path) / 'valid_count_checkpoint.pkl'
    if checkpoint_path.exists():
        logger.info("Loading checkpoint: %s", checkpoint_path)
        valid_count_dict = open_pickle_file(checkpoint_path)
    else:
        valid_count_dict = {}

    checkpoint_count = 0
    
    # Calculate total iterations for progress bar
    total_iterations = 1
    for iter_list in ctx.calculations_iter_lists:
        total_iterations *= len(iter_list)
    
    # Initialize progress bar
    pbar = tqdm(total=total_iterations, desc="Processing Valid Counts")
    
    def classify_under_loop(calculation_item):
        """
        calculation_item: Dict[iterator_idx: iterator_item]
        e.g. 
        - if iterator_idx corresponds to token_pairs, iterator_item is a token_pair
        - if iterator_idx corresponds to temperature, iterator_item is a temperature

        but iterator_idx is not directly 'temperature', it'll be like 'iterator_2' that corresponds to temperature
        calculations_config = ctx.xp_config['calculations'] # typically iterator_1: token_pairs; iterator_2: temperature.
        """
        nonlocal valid_count_dict
        nonlocal checkpoint_count
        
        # Checkpoint_path
        calculation_item_name = get_calculation_item_name(calculations_config, calculation_item)
        assert calculation_item_name != ''
        
        # Loading Checkpoint
        if calculation_item_name in valid_count_dict:
            pbar.update(1)
            return
        else:
            # Get the dataset (token_pair) from calculation_item
            dataset = calculation_item[dataset_iter_idx]
            
            # Initialize dict entry if needed
            if calculation_item_name not in valid_count_dict:
                valid_count_dict[calculation_item_name] = {}
            
        filtered_df = ctx.Answers_df.clone()

        for iter_idx, iter_value in calculation_item.items():
            iter_col_name = calculations_config[iter_idx]
            # correct artifacts cols errors
            if iter_col_name == 'token_pairs':
                iter_col_name = 'Token_pair'
            if iter_col_name == 'models':
                iter_col_name = 'Model'
            filtered_df = filtered_df.filter(pl.col(iter_col_name)==iter_value)

        # Check for NaN/null values and show diagnostic information
        fail_col = filtered_df['gen_fail']
        counter_col = filtered_df['gen_counter']

        # Check for null/NaN presence (handle different dtypes)
        has_null_fail = fail_col.is_null().any()
        has_nan_counter = counter_col.is_nan().any()
        has_null_counter = counter_col.is_null().any()

        if has_null_fail or has_nan_counter or has_null_counter:
            logger.warning("Missing values detected in %s:", calculation_item_name)
            if has_null_fail:
                null_count_fail = fail_col.is_null().sum()
                logger.warning("  - 'gen_fail': %d null values out of %d", null_count_fail, len(fail_col))
            if has_nan_counter:
                nan_count_counter = counter_col.is_nan().sum()
                logger.warning("  - 'gen_counter': %d NaN values out of %d", nan_count_counter, len(counter_col))
            if has_null_counter:
                null_count_counter = counter_col.is_null().sum()
                logger.warning("  - 'gen_counter': %d null values out of %d", null_count_counter, len(counter_col))

            # Show filter conditions
            logger.warning("  Filter conditions: %s", calculation_item)

        # Show values being aggregated (optional: can be commented out if too verbose)
        if len(filtered_df) <= 100:  # Only show values for small datasets
            logger.debug("Aggregation for %s:", calculation_item_name)
            logger.debug("  Filters: %s", calculation_item)
            logger.debug("  Row count: %d", len(filtered_df))
            logger.debug("  gen_fail values: %s", fail_col.to_list())
            logger.debug("  gen_counter values (non-null/NaN): %s", counter_col.drop_nulls().drop_nans().to_list())

        # Perform calculations
        valid_count_dict[calculation_item_name]['fail_count'] = filtered_df['gen_fail'].sum()  # Use sum() for boolean
        valid_count_dict[calculation_item_name]['fail_mean'] = filtered_df['gen_fail'].mean()  # Use sum() for boolean
        valid_count_dict[calculation_item_name]['gen_count'] = filtered_df['gen_counter'].mean()  # Use mean() 

        if len(filtered_df) <= 100:  # Show results for debugging
            logger.debug("  Result - fail_count: %s", valid_count_dict[calculation_item_name]['fail_count'])
            logger.debug("  Result - gen_count: %s", valid_count_dict[calculation_item_name]['gen_count'])

        # Saving checkpoint
        checkpoint_count += 1
        if checkpoint_count % 1000 == 0:
            save_pickle_file(valid_count_dict, checkpoint_path)
            logger.debug("Saved checkpoint for checkpoint_count=%d: %s", checkpoint_count, calculation_item_name)
        pbar.update(1)

    nested_loop(ctx.calculations_iter_lists, classify_under_loop)
    
    # Close progress bar
    pbar.close()
    
    # Save final checkpoint
    save_pickle_file(valid_count_dict, checkpoint_path)
    logger.info("Saved final checkpoint with %d items", len(valid_count_dict))

    # For valid_count_dict
    generate_personalized_figures(
        ctx.xp_config,
        calculations_config,
        ctx.calculations_iter_lists,
        revert_dictionary(ctx.new_var_models_idx),
        ctx.save_fig_path,
        valid_count_dict,
        pipe_summary_mode=False
    )


# ---------------------------------------------------------------------------
# Dimensionality reduction (UMAP / t-SNE) visualization
# ---------------------------------------------------------------------------

_MAX_SAMPLES_PER_MODEL = 100


def _impute_nan(X: np.ndarray) -> np.ndarray:
    """Replace NaN values with per-column mean (fallback 0)."""
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0, col_means)
    nan_mask = np.isnan(X)
    X_imputed = X.copy()
    for col in range(X.shape[1]):
        X_imputed[nan_mask[:, col], col] = col_means[col]
    return X_imputed


def _run_dr(X: np.ndarray, method: str, random_state: int = 42) -> np.ndarray:
    """Run dimensionality reduction. Returns (n, 2) embedding."""
    X_clean = _impute_nan(X)
    n = X_clean.shape[0]
    if n < 2:
        logger.warning("DR skipped: only %d point(s)", n)
        return X_clean[:, :2] if X_clean.shape[1] >= 2 else np.zeros((n, 2))

    if method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(30, n - 1)
        return TSNE(n_components=2, random_state=random_state, perplexity=perplexity).fit_transform(X_clean)
    else:
        import umap
        n_neighbors = min(15, n - 1)
        return umap.UMAP(n_components=2, random_state=random_state, n_neighbors=n_neighbors).fit_transform(X_clean)


_DISTINCT_COLORS = [
    "#e6194b",  # red
    "#f58231",  # orange
    "#3cb44b",  # green
    "#4363d8",  # blue
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#f032e6",  # magenta
    "#469990",  # teal
    "#9a6324",  # brown
    "#bfef45",  # lime
]


def _plot_dr_scatter(
    embedding: np.ndarray,
    labels: List[str],
    title: str,
    save_path: str,
    fig_name: str,
    point_size: int = 40,
    alpha: float = 1.0,
):
    """Scatter plot of 2D embedding coloured by label."""
    unique_labels = list(dict.fromkeys(labels))  # preserve order
    n_labels = len(unique_labels)

    if n_labels <= len(_DISTINCT_COLORS):
        cmap = [_DISTINCT_COLORS[i] for i in range(n_labels)]
    elif n_labels <= 20:
        cmap = cm.tab20(np.linspace(0, 1, n_labels))
    elif n_labels <= 40:
        cmap = np.vstack([
            cm.tab20(np.linspace(0, 1, 20)),
            cm.tab20b(np.linspace(0, 1, min(n_labels - 20, 20))),
        ])
    else:
        cmap = cm.gist_ncar(np.linspace(0, 0.95, n_labels))

    label_to_color = {lab: cmap[i] for i, lab in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(12, 8))
    for lab in unique_labels:
        mask = [l == lab for l in labels]
        pts = embedding[mask]
        ax.scatter(
            pts[:, 0], pts[:, 1],
            c=[label_to_color[lab]],
            label=truncate_model_name(lab, k=35),
            s=point_size,
            alpha=alpha,
            edgecolors="none",
        )

    ax.set_xlabel("Component 1", fontsize=12)
    ax.set_ylabel("Component 2", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    if n_labels <= 15:
        ax.legend(fontsize=9, loc="best")
    else:
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

    plt.tight_layout()
    save_fig_and_show(save_path=save_path, show=False, fig_name=fig_name)
    plt.close(fig)
    logger.info("Saved DR scatter: %s/%s", save_path, fig_name)


def _build_labels_for_models_idx(
    new_var_models_idx: Dict[int, str],
    n_models: int,
    labelling: str,
    model_groups_config: Optional[Dict] = None,
) -> List[str]:
    """Return a list of length n_models with one label per model index (0..n_models-1)."""
    if labelling == "by_orig":
        orig_groups = group_models_idx_by_var_or_orig(new_var_models_idx, group_by="orig")
        idx_to_orig = {}
        for orig_name, members in orig_groups.items():
            for midx, _ in members:
                idx_to_orig[midx] = orig_name
        return [idx_to_orig.get(i, new_var_models_idx.get(i, f"model_{i}")) for i in range(n_models)]

    elif labelling == "by_var":
        var_groups = group_models_idx_by_var_or_orig(new_var_models_idx, group_by="var")
        idx_to_var = {}
        for var_name, members in var_groups.items():
            for midx, _ in members:
                idx_to_var[midx] = var_name
        return [idx_to_var.get(i, new_var_models_idx.get(i, f"model_{i}")) for i in range(n_models)]

    elif labelling == "by_group":
        from audit_llm.Classification.model_grouping import build_group_mapping
        model_to_group, group_names = build_group_mapping(new_var_models_idx, model_groups_config)
        return [group_names.get(model_to_group.get(i, -1), "ungrouped") for i in range(n_models)]

    else:  # "by_model"
        return [new_var_models_idx.get(i, f"model_{i}") for i in range(n_models)]


def _subsample_per_model(
    X_flat: np.ndarray,
    model_ids: np.ndarray,
    max_per_model: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample to at most max_per_model rows per model. Drop all-NaN rows."""
    valid_mask = ~np.all(np.isnan(X_flat), axis=1)
    X_flat = X_flat[valid_mask]
    model_ids = model_ids[valid_mask]

    keep = []
    for mid in np.unique(model_ids):
        indices = np.where(model_ids == mid)[0]
        if len(indices) > max_per_model:
            indices = rng.choice(indices, size=max_per_model, replace=False)
        keep.extend(indices.tolist())
    keep = sorted(keep)
    return X_flat[keep], model_ids[keep]


def _generate_dr_plots(
    X_matrix: np.ndarray,
    labels: List[str],
    dr_method: str,
    prefix: str,
    save_path: str,
    point_size: int = 40,
    alpha: float = 1.0,
):
    """Run DR and produce a scatter plot."""
    if X_matrix.shape[0] < 2:
        logger.warning("Skipping %s: fewer than 2 points", prefix)
        return
    embedding = _run_dr(X_matrix, dr_method)
    title = f"{dr_method.upper()} — {prefix}"
    fig_name = f"{dr_method}_{prefix}.pdf"
    _plot_dr_scatter(embedding, labels, title, save_path, fig_name, point_size=point_size, alpha=alpha)


def feature_space_visualization(Experiment_config: Dict[str, Any]) -> None:
    """Produce UMAP or t-SNE scatter plots of the classification feature space.

    Plots are generated at up to four aggregation levels:
    - Model-averaged, TP-averaged (one point per model)
    - Model-averaged, per token pair (one point per model per TP) — optional
    - Per-sample (one point per sample, capped at 100 per model)
    - Per-sample, per token pair — optional

    Per-token-pair plots are controlled by ``dr_per_token_pair`` (default False).

    Labelling depends on config:
    - model_variations active → by_orig + by_var
    - model_groups active    → by_orig + by_group
    - neither                → by_model only

    Output folder structure::

        <save_fig_path>/
          model_averaged/
            tp_averaged/        ← aggregated across all TPs
            per_token_pair/     ← one subfolder per TP (if enabled)
              <tp_name>/
          per_sample/
            tp_averaged/
            per_token_pair/
              <tp_name>/
    """
    ctx = prepare_experiment_context(Experiment_config)
    xp_config = ctx.xp_config
    dr_method = xp_config.get("dr_method", "umap")
    per_tp_enabled = xp_config.get("dr_per_token_pair", False)
    base_save_path = Path(ctx.save_fig_path)
    rng = np.random.default_rng(42)

    model_variations_indices = Experiment_config["model_variations_indices"]
    model_groups_config = xp_config.get("model_groups")
    has_variations = bool(model_variations_indices)
    has_groups = model_groups_config is not None

    # Determine labelling strategies
    if has_variations and not has_groups:
        labellings = ["by_orig", "by_var"]
    elif has_groups:
        labellings = ["by_orig", "by_group"]
    else:
        labellings = ["by_model"]

    # --- Collect features per token pair ---
    dataset_iter_idx = get_iter_idx_from_calculations_config(
        iterator_name="token_pairs", xp_config=xp_config
    )
    token_pairs = ctx.calculations_iter_lists[dataset_iter_idx]
    calculation_items_list = dict_product_with_fix_item(
        ctx.calculations_iter_lists, fix_iterator_idx=dataset_iter_idx
    )
    # Use first non-TP calculation item (or empty dict)
    calculation_item = calculation_items_list[0] if calculation_items_list else {}

    selected_features = select_features_for_token_pair(
        ctx.intra_samples_feature_index_dict, xp_config
    )

    X_s: Dict[str, np.ndarray] = {}
    new_var_models_idx = None
    for tp in token_pairs:
        X_tp, current_idx = prepare_dataset_features(
            tp, calculation_item,
            ctx.intra_samples_features_dict,
            ctx.intra_samples_feature_index_dict,
            selected_features,
            ctx.new_var_models_idx,
            ctx.models_indices,
            xp_config,
            Experiment_config,
            ctx.MainDataset_df_iterators,
            ctx.Answers_df,
        )
        X_s[tp] = X_tp  # (n_samples, n_models, n_features)
        if new_var_models_idx is None:
            new_var_models_idx = current_idx

    n_models = next(iter(X_s.values())).shape[1]
    logger.info(
        "Feature space visualization: %d token pairs, %d models, dr=%s, per_tp=%s",
        len(X_s), n_models, dr_method, per_tp_enabled,
    )

    # --- Filter out ungrouped models when hardcoded groups are provided ---
    if has_groups and "group_by" not in model_groups_config:
        from audit_llm.Classification.model_grouping import build_group_mapping
        model_to_group, _ = build_group_mapping(new_var_models_idx, model_groups_config)
        grouped_indices = sorted(model_to_group.keys())
        if len(grouped_indices) < n_models:
            logger.info(
                "Excluding %d ungrouped model(s) from visualization (hardcoded groups active)",
                n_models - len(grouped_indices),
            )
            X_s = {tp: X_tp[:, grouped_indices, :] for tp, X_tp in X_s.items()}
            new_var_models_idx = {new_i: new_var_models_idx[old_i] for new_i, old_i in enumerate(grouped_indices)}
            n_models = len(grouped_indices)

    # --- A. Model-averaged, TP-averaged ---
    model_avg_tp_avg_dir = str(base_save_path / "model_averaged" / "tp_averaged")
    Path(model_avg_tp_avg_dir).mkdir(parents=True, exist_ok=True)

    X_avg_per_tp = [np.nanmean(X_tp, axis=0) for X_tp in X_s.values()]
    X_avg_all = np.nanmean(np.stack(X_avg_per_tp, axis=0), axis=0)  # (n_models, n_features)

    for labelling in labellings:
        labels = _build_labels_for_models_idx(new_var_models_idx, n_models, labelling, model_groups_config)
        _generate_dr_plots(X_avg_all, labels, dr_method, labelling, model_avg_tp_avg_dir, point_size=80)

    # --- B. Model-averaged, per TP (optional) ---
    if per_tp_enabled:
        for tp, X_tp in X_s.items():
            tp_safe = tp.replace("/", "-")
            tp_dir = str(base_save_path / "model_averaged" / "per_token_pair" / tp_safe)
            Path(tp_dir).mkdir(parents=True, exist_ok=True)
            X_avg_tp = np.nanmean(X_tp, axis=0)  # (n_models, n_features)
            for labelling in labellings:
                labels = _build_labels_for_models_idx(new_var_models_idx, n_models, labelling, model_groups_config)
                _generate_dr_plots(X_avg_tp, labels, dr_method, labelling, tp_dir, point_size=80)

    # --- C. Per-sample, TP-averaged ---
    per_sample_tp_avg_dir = str(base_save_path / "per_sample" / "tp_averaged")
    Path(per_sample_tp_avg_dir).mkdir(parents=True, exist_ok=True)

    min_samples = min(X_tp.shape[0] for X_tp in X_s.values())
    X_stack = np.stack([X_tp[:min_samples] for X_tp in X_s.values()], axis=0)
    X_mean_tp = np.nanmean(X_stack, axis=0)  # (min_samples, n_models, n_features)
    n_samples = X_mean_tp.shape[0]

    X_flat = X_mean_tp.reshape(n_samples * n_models, -1)
    model_ids = np.tile(np.arange(n_models), n_samples)
    X_flat_sub, model_ids_sub = _subsample_per_model(X_flat, model_ids, _MAX_SAMPLES_PER_MODEL, rng)

    for labelling in labellings:
        all_labels = _build_labels_for_models_idx(new_var_models_idx, n_models, labelling, model_groups_config)
        sample_labels = [all_labels[mid] for mid in model_ids_sub]
        _generate_dr_plots(
            X_flat_sub, sample_labels, dr_method, labelling, per_sample_tp_avg_dir,
            point_size=15, alpha=0.5,
        )

    # --- D. Per-sample, per TP (optional) ---
    if per_tp_enabled:
        for tp, X_tp in X_s.items():
            tp_safe = tp.replace("/", "-")
            tp_dir = str(base_save_path / "per_sample" / "per_token_pair" / tp_safe)
            Path(tp_dir).mkdir(parents=True, exist_ok=True)
            n_s = X_tp.shape[0]
            X_tp_flat = X_tp.reshape(n_s * n_models, -1)
            tp_model_ids = np.tile(np.arange(n_models), n_s)
            X_tp_sub, tp_mids_sub = _subsample_per_model(X_tp_flat, tp_model_ids, _MAX_SAMPLES_PER_MODEL, rng)

            for labelling in labellings:
                all_labels = _build_labels_for_models_idx(new_var_models_idx, n_models, labelling, model_groups_config)
                sample_labels = [all_labels[mid] for mid in tp_mids_sub]
                _generate_dr_plots(
                    X_tp_sub, sample_labels, dr_method, labelling, tp_dir,
                    point_size=15, alpha=0.5,
                )

    logger.info("Feature space visualization complete. Figures saved to %s", base_save_path)
