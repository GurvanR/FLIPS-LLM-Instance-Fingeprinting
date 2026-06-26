# DOCUMENTED CLONE-ONLY: This script imports the upstream pasquini-dario/LLMmap module
# (Fingerprinting_methods.LLMmap.LLMmap), which is NOT vendored here. It is NOT imported by the
# out-of-box gate-A path. To reproduce the LLMmap cross-classification, clone the upstream
# repository pasquini-dario/LLMmap.

from Fingerprinting_methods.LLMmap.cross_classif_task.make_dataset_from_Answers_format import *

from audit_llm.Classification.results_tables import make_model_wise_tables, plot_confusion_matrices_on_tr_size_dict
from audit_llm.xp_tools.checkpoint_utils import arrayify_confusion_matrices, set_key_as_int
from audit_llm.Classification.dca_analysis import compute_dca_showcase_data, plot_dca_showcase_by_original_model
from audit_llm.Classification.training_size_analysis import plot_trainsize_wise_curves

from audit_llm.plot_tools import full_var_model_name_to_full_safe_var_model_name_mapper

def LLMmap_classification(Experiment_config):
    (
        _,
        _,
        calculations_iter_lists,
        models_indices,
        new_models_idx,
        save_fig_path,
        _,
        xp_config,
        checkpoint_dir_path,
        MainDataset_df_iterators,
        Answers_df,
    ) = prepare_experiment_context(Experiment_config)

    classification_config = xp_config['classification_config']

    if classification_config.get('openset', False):
        save_fig_path= Path(save_fig_path) / 'OpenSet'
    else:
        save_fig_path= Path(save_fig_path) / 'ClosedSet'

    checkpoint_dir = Path(save_fig_path) / "train_size_checkpoints"

    if checkpoint_dir.exists() and not classification_config.get('use_checkpoint', True):
        # Delete dir:
        print(f"Removing existing checkpoint directory: {checkpoint_dir}")
        import shutil
        shutil.rmtree(checkpoint_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_size_dict = {}
    new_var_models_idx = None
    new_var_models_idx_path = checkpoint_dir_path / "new_var_models_idx.json"

    # Build list of calculation items from iterator products
    calculation_items_list = dict_product_with_fix_item(
        calculations_iter_lists,
        fix_iterator_idx=None
    )

    for calculation_item in calculation_items_list:

        # Generate readable calculation item name
        calculations_config = xp_config.get("calculations", None)
        calculation_item_name = get_calculation_item_name(
                calculations_config,
                calculation_item
            )

        # Path for global entries and metadata saved per calculation item
        calculation_item_name_checkpoint_dir = checkpoint_dir / calculation_item_name
        calculation_item_name_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Update the path to .parquet
        global_entries_checkpoint_path = calculation_item_name_checkpoint_dir / "global_entries.parquet"

        # Load or compute global entries
        if global_entries_checkpoint_path.exists():
            print(f"Loading global entries from {global_entries_checkpoint_path}")
            # Read the parquet file
            lf = pl.scan_parquet(global_entries_checkpoint_path)

            # Recover metadata from the file schema
            import pyarrow.parquet as pq
            meta = pq.read_metadata(global_entries_checkpoint_path).metadata

            # Extract nb_of_samples_per_class (stored as bytes in Parquet metadata)
            nb_raw = meta.get(b"nb_of_samples_per_class")
            nb_of_samples_per_class = int(nb_raw) if nb_raw else None

            # Convert to list of dicts for the rest of your script
            global_entries = lf.collect().to_dicts()
        else:
            print(f"No global entries found. Computing for: {calculation_item_name}")
            global_entries, nb_of_samples_per_class = get_global_entries(
                MainDataset_df_iterators,
                Answers_df,
                calculation_item,
                xp_config,
                models_indices,
                Experiment_config,
                new_models_idx
            )

            # Convert list of dicts to Polars DataFrame
            df = pl.DataFrame(global_entries)

            # Convert to PyArrow Table to add custom metadata
            import pyarrow as pa
            table = df.to_arrow()

            # Add custom metadata to the schema
            custom_meta = {
                b"nb_of_samples_per_class": str(nb_of_samples_per_class).encode()
            }
            existing_meta = table.schema.metadata or {}
            combined_meta = {**existing_meta, **custom_meta}
            table = table.replace_schema_metadata(combined_meta)

            # Write the table to Parquet
            import pyarrow.parquet as pq
            pq.write_table(table, global_entries_checkpoint_path)

            print(f"Global entries (and {nb_of_samples_per_class = }) saved to {global_entries_checkpoint_path}")

        # Iterate through train sizes
        for train_size in xp_config["train_sizes"]:

            # Ensure nested structure is initialized
            train_size_dict.setdefault(train_size, {})

            train_size_save_dir = calculation_item_name_checkpoint_dir / str(train_size)
            train_size_save_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = train_size_save_dir / f"results.json"

            # Load checkpoint if available
            if checkpoint_path.exists():
                print(f"Loading checkpoint for train_size={train_size}, item={calculation_item_name}")
                results = load_json(path=checkpoint_path)
                print(f"Checkpoint loaded from {checkpoint_path}")

                train_size_dict[train_size][calculation_item_name] = results

                if new_var_models_idx_path.exists():
                    loaded_new_var_models_idx = load_json(path=new_var_models_idx_path, set_keys_as_int=True)
                if new_var_models_idx is None:
                    new_var_models_idx = loaded_new_var_models_idx
                elif new_var_models_idx != loaded_new_var_models_idx:
                    raise ValueError(
                        f"Inconsistent new_var_models_idx: existing differs from loaded for {calculation_item_name}"
                    )

                continue

            # No checkpoint → compute pipeline
            print(
                f"No checkpoint found for train_size={train_size}, item={calculation_item_name}, "
                f"running cross classification."
            )


            results, current_new_var_models_idx = cross_classification(
                xp_config,
                train_size_save_dir,
                global_entries,
                nb_of_samples_per_class,
                train_size,
            )

            if new_var_models_idx is None:
                new_var_models_idx = current_new_var_models_idx
            elif new_var_models_idx != current_new_var_models_idx:
                raise ValueError(
                    f"Inconsistent new_var_models_idx: existing differs from computed for {calculation_item_name}"
                )
            print(f"{new_var_models_idx = }")

            train_size_dict[train_size][calculation_item_name] = results

            if xp_config.get("save_dca_showcase_data", False):
                unique_model_vars = get_unique_models_from_global_entries(global_entries)

                full_var_model_name_to_full_safe_var_model_name_map = full_var_model_name_to_full_safe_var_model_name_mapper(unique_model_vars)
                print(f"{full_var_model_name_to_full_safe_var_model_name_map = }")
                compute_dca_showcase_data(train_size_dict, classification_config, full_var_model_name_to_full_safe_var_model_name_map)
                print("DCA showcase data computed and added to train_size_dict. for train size:", train_size, "and calculation_item", calculation_item_name)

            write_json(new_var_models_idx, new_var_models_idx_path)

            # Save checkpoint
            write_json(results, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    # Display new_var_models_idx

    # converting confusion_matrices that are lists to np.arrays

    train_size_dict = arrayify_confusion_matrices(train_size_dict)
    train_size_dict = set_key_as_int(train_size_dict)

    print(f"{new_var_models_idx = }")

    if xp_config.get("save_dca_showcase_data", False):

        plot_dca_showcase_by_original_model(
            train_size_dict,
            classification_config,
            save_fig_path,
        )

    plot_trainsize_wise_curves(
        train_size_dict, save_fig_path, classification_config,
        batch_sizes=[bs for bs in classification_config['batch_prediction_sizes'] if bs <=8],
        datasets=None, models_idx=new_var_models_idx
    )

    if False:
        make_model_wise_tables(
        train_size_dict, save_fig_path, classification_config,
        batch_sizes=classification_config['batch_prediction_sizes'],
        datasets=['no_token_pairs'],
        models_idx=new_var_models_idx,
        ds_group_names=['no_grouping'],
        )

        plot_confusion_matrices_on_tr_size_dict(
            train_size_dict, save_fig_path,
            xp_config,
            classification_config,
            batch_sizes=classification_config['batch_prediction_sizes'],
            datasets=['no_token_pairs'],
            models_idx=new_var_models_idx,
            ds_group_names=['no_grouping'],
        )


from typing import List, Dict, Any

def get_unique_models_from_global_entries(
    global_entries: List[Dict[str, Any]]
) -> List[str]:
    """
    Extract unique model names from global entries.

    Args:
        global_entries: List of global entry dictionaries. Each entry is expected
                        to contain a key 'llm' with the model (and variation) name.

    Returns:
        List of unique model names (order preserved).
    """
    unique_models = []
    seen = set()

    for entry in global_entries:
        model_name = entry.get("llm")
        if model_name is not None and model_name not in seen:
            seen.add(model_name)
            unique_models.append(model_name)

    return unique_models

def get_global_entries(
    MainDataset_df_iterators,
    Answers_df,
    calculation_item,
    xp_config,
    models_indices,
    Experiment_config,
    new_models_idx
):
    """
    Compute and return the global entries for the experiment.

    Args:
        MainDataset_df_iterators: Iterators for the main dataset.
        Answers_df: DataFrame containing the model answers.
        calculation_item: The current calculation item key/name.
        xp_config: The experiment configuration for sampling.
        models_indices: List or mapping of model indices.
        Experiment_config: Full experiment configuration dict.
        new_models_idx: Indices of new models introduced in the experiment.

    Returns:
        global_entries: Output of the model trace generation process.
    """

    # Retrieve sampling iterators
    all_sampling_iterators, all_answers_df_iterators = (
        get_sampling_and_answers_iterators_from_Dataset(
            MainDataset_df_iterators,
            Answers_df
        )
    )

    # Compute valid sample indices
    samples_indices, _, _ = get_samples_indices(
        calculation_item,
        xp_config,
        models_indices,
        all_sampling_iterators,
        MainDataset_df_iterators,
        Experiment_config,
    )

    # Process model variation indices (if present)
    model_variations_indices = Experiment_config.get("model_variations_indices")
    if model_variations_indices:
        # Keep only indices that exist in sample_indices
        model_variations_indices = {
            key: [idx for idx in idx_list if idx in samples_indices]
            for key, idx_list in model_variations_indices.items()
        }

        variations = list(model_variations_indices.items())
    else:
        # No variations: process as single "default" variation
        variations = [('default', samples_indices)]

    # Generate global entries
    global_entries = generate_model_traces(
        answers_df=Answers_df,
        new_models_idx=new_models_idx,
        variations=variations,
        xp_config=xp_config,
        MainDataset_df_iterators=MainDataset_df_iterators,
    )

    nb_of_samples_per_class = len(variations[0][1])

    return global_entries, nb_of_samples_per_class


def generate_model_traces(
    answers_df: pl.DataFrame,
    new_models_idx: Dict[int, str],
    variations,
    xp_config: Dict,
    MainDataset_df_iterators,
) -> List[Dict[str, Any]]:
    """
    Generate model traces from answer dataframe for multiple models and variations.

    Each trace groups all prompt-index responses for one (model, variation) pair into
    a flat list of [question, answer] pairs ordered by prompt_idx.

    Parameters
    ----------
    answers_df : pl.DataFrame
        DataFrame containing columns: 'Model', 'Dataset_Question Index', 'prompt_idx', 'Answer'
    new_models_idx :
       variations:

    Returns
    -------
    List[Dict[str, Any]]
        List of trace dictionaries, where each dictionary has the structure:
        [
            {
                'llm': str,              # Format: 'model_name_variation_name'
                'traces': List[List]     # List of [question, answer] pairs, one per prompt_idx
            },
            ...
        ]

    Output Format Details
    ---------------------
    Each entry in the returned list represents one sample from one model/variation:

    {
        'llm': 'gpt-4_baseline',  # Model name + variation name
        'traces': [
            ['What is 2+2?', '4'],              # prompt_idx=2
            ['Explain further', 'The sum...'],  # prompt_idx=3
            ...
            ['Final question', 'Final answer']  # prompt_idx=9
        ]
    }

    Notes
    -----
    - For N samples and M prompt indices, each model/variation produces N entries
    - Each entry contains M traces (one per prompt index)
    - Traces are ordered by prompt_idx in ascending order
    """
    prompt_range: Tuple[int, int] = (2, 10)
    prompt_start, prompt_end = prompt_range
    global_entries = []

    abliterated_models = xp_config.get('abliterated_models', None)
    if abliterated_models is not None:
        abliterated_models = xp_config['abliterated_models']
        if len(abliterated_models)==0:
            abliterated_models = ABLITERATED_MODELS # empty list means all abliterated models

    # sample indices of abliterated models (corresponding to unique sampling params that is temp = 1.0 and system_prompt_idx = -1)
    abliterated_samples_indices = list(compute_model_variations_indices(xp_config,
                                                                    main_dataset_df= MainDataset_df_iterators,
                                                                    model_variations = [{'temperature': [1.0], 'system_prompt_idx': [-1]}]
                                                                    ).values())[0]

    abliterated_samples_variations = {'ablit': abliterated_samples_indices}
    models = [model_name for model_name in new_models_idx.values()]
    if abliterated_models is not None:
        models = models + abliterated_models

    print(f"Generating model traces for models: {models}")
    for model_name in models:
        # Filter answers for current model
        model_answers_df = answers_df.filter(pl.col('Model') == model_name)

        if 'abliterated' in model_name.lower():
            new_variations = list(abliterated_samples_variations.items())
        else:
            new_variations = variations
        for variation_name, question_indices in new_variations:
            model_entries = _process_model_variation(
                model_answers_df=model_answers_df,
                question_indices=question_indices,
                prompt_start=prompt_start,
                prompt_end=prompt_end,
            )

            # Create global entries for each sample
            n_samples = len(next(iter(model_entries.values()))) if model_entries else 0

            for sample_idx in range(n_samples):
                global_entries.append({
                    'llm': f"{model_name}_{variation_name}" if variation_name != 'default' else model_name,
                    'traces': [
                        model_entries[prompt_idx][sample_idx]
                        for prompt_idx in range(prompt_start, prompt_end)
                    ]
                })

    return global_entries


def _process_model_variation(
    model_answers_df: pl.DataFrame,
    question_indices: List[int],
    prompt_start: int,
    prompt_end: int,
) -> Dict[int, List[List[str]]]:
    """
    Process a single model variation and extract question-answer pairs.

    Parameters
    ----------
    model_answers_df : pl.DataFrame
        Filtered dataframe for specific model
    question_indices : List[int]
        Question indices to include
    prompt_start : int
        Starting prompt index (inclusive)
    prompt_end : int
        Ending prompt index (exclusive)
    prompt_idx_to_actual_prompt : callable
        Function to convert prompt index to prompt text

    Returns
    -------
    Dict[int, List[List[str]]]
        Mapping of prompt_idx -> list of [question, answer] pairs
        {
            2: [['question', 'answer'], ['question', 'answer'], ...],
            3: [['question', 'answer'], ['question', 'answer'], ...],
            ...
        }
    """
    model_entries = defaultdict(list)

    # Filter to relevant question indices
    filtered_df = model_answers_df.filter(
        pl.col('Dataset_Question Index').is_in(question_indices)
    )

    # Process each prompt index
    for prompt_idx in range(prompt_start, prompt_end):
        # Get data for this specific prompt
        prompt_df = filtered_df.filter(pl.col('prompt_idx') == prompt_idx)

        # Get the actual prompt text
        question = prompt_idx_to_actual_prompt(
            prompt_idx=prompt_idx,
            token_pair="no_token_pairs"
        )

        # Extract all answers for this prompt
        # Assuming each row is a unique sample
        n_samples = len(prompt_df)

        for sample_idx in range(n_samples):
            answer = prompt_df['Answer'][sample_idx]
            model_entries[prompt_idx].append([question, answer])

    # Verify consistency: all prompt indices should have same number of samples
    sample_counts = {idx: len(pairs) for idx, pairs in model_entries.items()}
    if len(set(sample_counts.values())) > 1:
        raise ValueError(
            f"Inconsistent sample counts across prompts: {sample_counts}. "
            "All prompt indices should have the same number of samples."
        )

    return model_entries
