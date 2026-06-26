# DOCUMENTED CLONE-ONLY: This script imports the upstream pasquini-dario/LLMmap module
# (Fingerprinting_methods.LLMmap.LLMmap), which is NOT vendored here. It is NOT imported by the
# out-of-box gate-A path. To reproduce the LLMmap cross-classification, clone the upstream
# repository pasquini-dario/LLMmap.
# NOTE: The "ds_wise" result key is intentionally retained here to match the upstream LLMmap schema consumed by the checkpoint_utils.py shim.


from pathlib import Path
import torch
import json
import tqdm
import copy
from audit_llm.xp_init_fun import *
from audit_llm.os_tools import *
from audit_llm.Bits_Generation.Bits_Dataset_Making.Prompts import prompt_idx_to_actual_prompt
from audit_llm.xp_tools.model_filtering import full_var_model_name_to_original_model_name

from pprint import pprint


import random
import json
from pathlib import Path
from collections import defaultdict

def train_test_split_global_entries(
    global_entries,
    xp_config,
    nb_of_samples_per_class: int,
    save_dir: Path,
    train_size: int,
    test_size: float,
    seed:int,
    k:int,
    bs: int = 8, # batch size (number of prompts per entry)
):
    """
    Split into train and test set by assigning entire entries (not splitting within entries).
    Uses k as a random seed for deterministic but randomized splitting.

    Args:
        - global_entries: List[Dict[str, Any]]
            List of trace dictionaries, where each dictionary has the structure:
            [
                {
                    'llm': str,              # Format: 'model_name_variation_name'
                    'traces': List[List]     # List of [question, answer] pairs, one per prompt_idx, and there are 8 different prompt_idx so len(traces)==8
                },
                ...
            ]
        - nb_of_samples_per_class: Total number of samples for each 'llm' class
        - train_size: int - hard number of samples present in train (has to be a multiple of 8)
        - test_size: float - test_size ratio from total number of samples
        - k: int - random seed for reproducible shuffling
    Returns:
        List of dictionaries with added 'dataset' field indicating 'train' or 'test'
    """
    save_path = save_dir / f"train_test_dataset_bs{bs}_{k}.jsonl"
    # --- Check if save_path exists ---
    if save_path.exists():
        print(f"train_test_dataset_bs{bs}_{k}.jsonl dataset already exists at {save_path}. Skipping split generation.")
        return save_path
    # Set random seed for reproducibility
    random.seed(seed)

    # Validate that train_size is a multiple of 8
    assert train_size % 8 == 0, f"train_size must be a multiple of 8, got {train_size}"

    # Calculate number of test samples and round to multiple of 8
    nb_test_samples = test_size #  int(nb_of_samples_per_class * test_size)
    if nb_test_samples % 8 != 0:
        print(f"Rounding down nb_test_samples from {nb_test_samples} to be multiple of 8, as it currently is {nb_test_samples}")
        nb_test_samples = (nb_test_samples // 8) * 8

    # Validate that train + test doesn't exceed total samples
    assert train_size + nb_test_samples <= (nb_of_samples_per_class)*(bs/8), \
        f"train_size ({train_size}) + test_samples ({nb_test_samples}) = {train_size + nb_test_samples} " \
        f"exceeds nb_of_samples_per_class ({nb_of_samples_per_class})"

    # Group entries by LLM
    llm_groups = defaultdict(list)
    for entry in global_entries:
        llm_groups[entry['llm']].append(entry)

    result = []

    # Process each LLM group
    for llm, entries in llm_groups.items():
        # Shuffle entries using k as seed for reproducibility
        shuffled_entries = copy.deepcopy(entries)
        random.shuffle(shuffled_entries)

        # Count cumulative samples for entries
        cumulative_samples = 0
        train_entries = []
        test_entries = []

        for entry in shuffled_entries:
            entry['traces'] = entry['traces'][:bs] # cutting the first bs queries.
            entry_samples = len(entry['traces'])
            assert entry_samples == bs, f"Each entry must have {bs} traces, got {entry_samples} for llm {llm}"

            # Assign to train if we haven't filled train_size yet
            if cumulative_samples < train_size:
                train_entries.append(entry)
                cumulative_samples += entry_samples
            # Assign to test if we haven't filled test quota yet
            elif cumulative_samples < train_size + nb_test_samples:
                test_entries.append(entry)
                cumulative_samples += entry_samples
            # Otherwise, skip this entry (unused samples)

        # Add train entries to result
        for entry in train_entries:
            result.append({
                "dataset": "train",
                "llm": entry['llm'],
                "traces": entry['traces']
            })

        if xp_config.get("save_dca_showcase_data", False):
            for entry in test_entries:
                result.append({
                    "dataset": "test",
                    "llm": full_var_model_name_to_original_model_name(entry['llm']), # getting original name of the llm so original llmmap clf model recognizes it. Note that this operations multiplies by nb_of_variations the test set, but doesn't mean we have to adjust in yaml config.
                    "var_llm_name": entry['llm'],
                    "traces": entry['traces']
                })
        else:
            # Add test entries to result
            for entry in test_entries:
                result.append({
                    "dataset": "test",
                    "llm": entry['llm'],
                    "traces": entry['traces']
                })

    # Save result in save_dir

    with open(save_path, 'w') as f:
        for entry in result:
            f.write(json.dumps(entry) + '\n')

    print(f"Saved train/test split dataset to {save_path.resolve()}")
    return save_path

from Fingerprinting_methods.LLMmap.LLMmap import CONF_NAME, MODEL_NAME, TEMPLATE_NAME
from Fingerprinting_methods.LLMmap.LLMmap.dataset import load_datasets
from Fingerprinting_methods.LLMmap.LLMmap.trainer import train_model
from Fingerprinting_methods.LLMmap.LLMmap.utility import read_conf_file, write_conf_file

def train_and_save_model(
        dataset_path: Path,
        k:int,
        ckpt_dir: Path,
        export_dir: Path,
        conf_file_path: Path,
        is_closed:bool = False,
):
    """
    Based on train.py file of LLMmap repository.
    """

    # 1) configuration --------------------------------------------------
    conf = read_conf_file(conf_file_path)

    ## updating dataset_path
    conf['dataset_path'] = str(dataset_path)

    conf['is_open'] = not is_closed
    print("\nLoaded configuration:")
    pprint(conf)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    model_export_path = export_dir / MODEL_NAME
    conf_export_path = export_dir / CONF_NAME

    # Skipping model training if already exists
    if model_export_path.exists():
        print(f"Model already exists at {model_export_path.resolve()}. Skipping training.")
        # opening json conf:
        conf = read_conf_file(conf_export_path)
        llms_map = conf['llms_map']
        return llms_map


    print(f"\nCheckpoints → {ckpt_dir.resolve()}")
    print(f"Export file → {export_dir.resolve()}\n")

    # 3) dataset --------------------------------------------------------
    print("Loading datasets...")
    (loader_train, loader_test), _, (ds_train, _) = load_datasets(
        conf, siamese=conf['is_open'],
        ks=conf.get('num_istances_dataset', None)
    )
    print("✓ Datasets loaded.")

    write_conf_file(conf_export_path, conf)

    # 4) train ----------------------------------------------------------
    print("\nStarting training...")
    trainer, model = train_model(
        ckpt_dir.as_posix(), siamese=conf['is_open'],
        loader_train=loader_train, loader_test=loader_test, conf=conf
    )

    # 5) export ---------------------------------------------------------
    torch.save(model.state_dict(), model_export_path)
    print("\n✓ Training finished")
    print("✓ Weights exported:", model_export_path.resolve())

    llms_map = conf['llms_map']
    return llms_map

from Fingerprinting_methods.LLMmap.LLMmap.dataset import load_datasets
from Fingerprinting_methods.LLMmap.LLMmap.inference import load_LLMmap, write_templates
from Fingerprinting_methods.LLMmap.LLMmap.templates import template_generation
from Fingerprinting_methods.LLMmap.LLMmap import TEMPLATE_NAME

def setup_templates(export_dir: Path):
    template_out = os.path.join(export_dir, TEMPLATE_NAME)

    # Load previous templates if file exists
    if os.path.exists(template_out):
        print(f"Resuming: '{template_out}' already exists. Loading previous templates...")
        existing = load_json(path=template_out)
    else:
        existing = {}

    conf, inf = load_LLMmap(export_dir, device='cpu')
    print("model loaded")

    if not conf['is_open']:
        print("Applicable only to open-set inference model. Aborting...")
        sys.exit(1)

    siamese = False
    (loader_train, loader_test), cache, (dataset_train, dataset_test) = load_datasets(
        conf,
        siamese=siamese,
        ks=conf.get('num_istances_dataset', None)
    )
    print("datasets loaded")

    results = template_generation(inf.model, loader_train, loader_test)
    print(f"Accuracy on test set: {results['accuracy']}")

    templates = results['templates']

    # Iterate and compute only missing ones
    for i in range(len(templates)):
        llm = inf.label_map[i]

        # Skip if already saved previously
        if llm in existing:
            print(f"Template for '{llm}' already exists. Skipping...")
            continue

        print(f"Computing template for '{llm}'...")
        existing[llm] = templates[i]

        # Write incrementally for checkpointing
        write_templates(template_out, existing)

    print(f"All templates saved to '{template_out}'")

from Fingerprinting_methods.LLMmap.LLMmap.dataset import read_dataset
from Fingerprinting_methods.LLMmap.LLMmap.inference import load_LLMmap

def get_topk_indices_from_distances(distances, k):
    """Return top-k indices (smaller distance means better)."""
    distances = np.asarray(distances)
    topk_idx = np.argsort(distances)[:k]
    return topk_idx

def get_metrics_from_confusion(confusion_matrix):
    """
    Calculate classification metrics from a confusion matrix.

    Args:
        confusion_matrix: numpy array of shape (num_classes, num_classes)
                         where confusion[i, j] = count of samples with true label i predicted as j

    Returns:
        Dict[str, float]: Dictionary containing accuracy, precision, recall, and f1 scores
    """
    # Total number of samples
    total = confusion_matrix.sum()

    # Accuracy: correct predictions / total predictions
    accuracy = np.trace(confusion_matrix) / total if total > 0 else 0.0

    # Per-class metrics
    num_classes = confusion_matrix.shape[0]
    precisions = []
    recalls = []
    f1_scores = []

    for i in range(num_classes):
        # True positives: diagonal element
        tp = confusion_matrix[i, i]

        # False positives: sum of column i minus true positives
        fp = confusion_matrix[:, i].sum() - tp

        # False negatives: sum of row i minus true positives
        fn = confusion_matrix[i, :].sum() - tp

        # Precision: tp / (tp + fp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precisions.append(precision)

        # Recall: tp / (tp + fn)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(recall)

        # F1: harmonic mean of precision and recall
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    # Macro-averaged metrics (unweighted average across classes)
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1_scores)

    return {
        'accuracy': float(accuracy),
        'precision': float(macro_precision),
        'recall': float(macro_recall),
        'f1': float(macro_f1),
    }


def evaluate_model(
        classification_config,
        xp_config,
        export_dir: Path,
):
    """
    Returns:
        metrics: Dict[str, float] = {<metric_name> : value}
        metrics typically: ['accuracy', 'f1', 'precision', 'recall'] + ['confusion_matrix']
    """
    topk = classification_config.get('topk', 1)
    conf, inf = load_LLMmap(export_dir, device='cpu')
    print("model loaded for evaluation.")

    if not conf.get('is_open', False):
        print('Applicable to only open-set inference model. Aborting...')
        sys.exit(1)

    if not inf.ready:
        print('No templates found for the model. Aborting...')
        sys.exit(1)

    train, test = read_dataset(conf['dataset_path'])
    k_values = tuple(range(1, topk + 1))

    idx_to_name = {k: v for (k, v) in inf.label_map.items()}  # int -> name
    # correcting the name change from meta-llama/Meta-Llama-3.1-8B-Instruct to meta-llama/Llama-3.1-8B-Instruct (typo error in original llmmap classifier)
    old = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    new = "meta-llama/Llama-3.1-8B-Instruct"

    # update idx_to_name (int -> name)
    for k, v in idx_to_name.items():
        if v == old:
            idx_to_name[k] = new

    # rebuild llms_map from idx_to_name
    llms_map = {v: k for k, v in idx_to_name.items()} # name -> int

    num_classes = len(idx_to_name)

    # Prepare containers
    num_samples = 0
    topk_correct_counter = {k: 0 for k in k_values}

    # confusion matrix for top-1
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    probs_save_map = defaultdict(list)
    print("Starting evaluation on test set...")

    import time
    total_items = len(test)
    # --- Timing Setup ---
    start_time = time.time()
    last_report_time = start_time
    # --------------------

    print(f"Starting evaluation on {total_items} samples...", flush=True)

    for i, entry in enumerate(tqdm(test, desc="Processing", unit="item")):
        llm_name = entry['llm']
        if llm_name not in llms_map:
            # skip unknown labels (defensive)
            print(f"Warning: ground-truth label '{llm_name}' not found in model.label_map; skipping sample.", flush=True)
            continue

        gt_idx = int(llms_map[llm_name])
        answers = [trace[1] for trace in entry['traces']]

        # Bottleneck: distance computation
        distances = np.asarray(inf(answers))



        num_samples += 1

        # Calculate Top-K
        for k in k_values:
            # Helper function assumed to be defined in your scope
            preds_k = get_topk_indices_from_distances(distances, k)
            if gt_idx in preds_k:
                topk_correct_counter[k] += 1

        # Top-1 prediction for confusion matrix
        top1_idx = int(get_topk_indices_from_distances(distances, 1)[0])
        confusion[gt_idx, top1_idx] += 1

        if xp_config.get("save_dca_showcase_data", False):
            var_model_name = entry["var_llm_name"]
            safe_label_index = llms_map[llm_name]
            # distance between variation and safe model
            probs_save_map[f"safe_{var_model_name}"].append(float(distances[safe_label_index]))
            probs_save_map[f"top1_pred_{var_model_name}"].append(idx_to_name[top1_idx])

        # --- 60-Second Progress Report ---
        current_time = time.time()
        if current_time - last_report_time >= 60:
            elapsed_sec = current_time - start_time
            processed_count = i + 1
            percent = (processed_count / total_items) * 100

            # Calculate ETA
            avg_time_per_item = elapsed_sec / processed_count
            remaining_items = total_items - processed_count
            eta_seconds = avg_time_per_item * remaining_items
            eta_min = eta_seconds / 60

            print(f"--- Progress Report ---", flush=True)
            print(f"Completed: {percent:.1f}% ({processed_count}/{total_items})", flush=True)
            print(f"Elapsed: {elapsed_sec/60:.1f} min | Est. Remaining: {eta_min:.1f} min", flush=True)
            print(f"-----------------------", flush=True)

            last_report_time = current_time
        # ---------------------------------

    print("Evaluation completed.", flush=True)

    # Calculate metrics from confusion matrix
    result_metrics: Dict[str, Any] = get_metrics_from_confusion(confusion)

    print( "cofusion_matrix done." )
    # Add confusion matrix to results
    result_metrics['confusion_matrix'] = confusion

    # Add top-k accuracies
    for k in k_values:
        topk_acc = topk_correct_counter[k] / num_samples if num_samples > 0 else 0.0
        result_metrics[f'top{k}_accuracy'] = float(topk_acc)

    # Add distances
    result_metrics['probs_save_map'] = dict(probs_save_map)

    return result_metrics



import json
import numpy as np
import copy
from pathlib import Path

def cross_classification(
    xp_config,
    save_dir: Path,
    global_entries,
    nb_of_samples_per_class,
    train_size,
):
    classification_config = xp_config['classification_config']
    batch_predictions_sizes = classification_config['batch_prediction_sizes']
    checkpoint_path = save_dir / "cross_classification_checkpoint.json"

    # --- Load Checkpoint if it exists ---
    if checkpoint_path.exists():
        print(f"--- Loading checkpoint from {checkpoint_path} ---")
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        summary = checkpoint_data.get('summary', {'ds_wise': {}})
        # metrics_tracker stores: { "batch_size": [list_of_metrics_dicts] }
        metrics_tracker = checkpoint_data['metrics_tracker']
        llms_map = checkpoint_data['llms_map']
    else:
        summary = {'ds_wise': {}}
        metrics_tracker = {}

    reference_llms_map = None

    for batch_size in batch_predictions_sizes:
        bs_key = str(batch_size)
        print(f"\n=== Starting processing for batch size: {batch_size} ===")
        print(f"{bs_key in metrics_tracker = }")
        if bs_key not in metrics_tracker:
            metrics_tracker[bs_key] = []

        # removing the skipping so that we can re-aggregate even if already done.
        if True:
            # Skip if this batch_size is already fully summarized
            if bs_key in summary.get('ds_wise', {}) and len(metrics_tracker[bs_key]) == classification_config["n_splits"]:
                print(f"Batch size {batch_size} already completed. Skipping.")
                continue

        seed = xp_config.get('dataset_cross_split_seed', 42)
        DEBUG_MODE: bool = xp_config.get('debug_mode', False)
        for k in range(classification_config["n_splits"]):
            # Check if this split was already processed
            if k < len(metrics_tracker[bs_key]):
                print(f"Skipping Batch {batch_size} | Split {k} (already done).")
                continue

            # Or skip
            if xp_config.get("stop_computing_splits", False):
                print(f"Skipping Batch size {batch_size} as stop_computing_splits option is activated.")
                continue

            print(f"\n>>> Processing Batch {batch_size} | Split {k}/{classification_config['n_splits']} <<<")

            # 1. Dataset Generation
            if DEBUG_MODE:
                llmmap_dataset_path = Path('src/Fingerprinting_methods/LLMmap/data/datasets/default_dataset.jsonl')
            else:
                llmmap_dataset_path = train_test_split_global_entries(
                    global_entries, xp_config, nb_of_samples_per_class, save_dir,
                    train_size, classification_config["test_size"],
                    seed=seed + k, k=k, bs=batch_size
                )

            conf_file_path = xp_config.get('conf_file_path', Path(get_repository_level_path()) / 'src/Fingerprinting_methods/LLMmap/confs/default.json')

            if xp_config.get("cuda_visible_devices") is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = xp_config["cuda_visible_devices"]

            # 2. Model Loading/Training
            if xp_config.get("save_dca_showcase_data", False):
                export_dir_k = Path(get_repository_level_path()) / "src/Fingerprinting_methods/LLMmap/data/pretrained_models/default"
                default_model_conf_file_path = export_dir_k / "conf.json"

                with open(default_model_conf_file_path, 'r') as f:
                    clf_conf_llmmap = json.load(f)
                clf_conf_llmmap['dataset_path'] = str(llmmap_dataset_path)

                with open(default_model_conf_file_path, 'w') as f:
                    json.dump(clf_conf_llmmap, f, indent=4)
                llms_map = clf_conf_llmmap['llms_map']
            else:
                ckpt_dir_k = save_dir / f"ckpt_split_bs_{batch_size}_{k}"
                export_dir_k = save_dir / f"export_split_{batch_size}_{k}"
                llms_map = train_and_save_model(
                    llmmap_dataset_path, k, ckpt_dir_k, export_dir_k,
                    is_closed=classification_config.get("is_closed", False),
                    conf_file_path=conf_file_path
                )
                if xp_config.get("train_only", False):
                    continue
                if not classification_config.get("is_closed", False):
                    setup_templates(export_dir_k)

            # Consistency Check for llms_map
            if reference_llms_map is None: reference_llms_map = llms_map
            elif llms_map != reference_llms_map:
                raise ValueError(f"llms_map inconsistency at split {k}")

            # 3. Evaluation
            metrics = evaluate_model(classification_config, xp_config, export_dir_k)

            # --- Prepare metrics for JSON serialization ---
            metrics_serialized = copy.deepcopy(metrics)
            if 'confusion_matrix' in metrics_serialized:
                metrics_serialized['confusion_matrix'] = metrics_serialized['confusion_matrix'].tolist()

            metrics_tracker[bs_key].append(metrics_serialized)

            # --- Save Progress to Checkpoint ---
            with open(checkpoint_path, 'w') as f:
                json.dump({'summary': summary, 'metrics_tracker': metrics_tracker, 'llms_map': llms_map}, f, indent=4)

        if xp_config.get("train_only", False):
            continue

        # 4. Aggregation (After all splits for current batch_size are done)
        summary['ds_wise'][bs_key] = {'no_token_pairs': {'llmmap_clf': {}}}
        clf_summary = summary['ds_wise'][bs_key]['no_token_pairs']['llmmap_clf']

        current_metrics_list = metrics_tracker[bs_key]
        metric_names = ['accuracy', 'f1', 'precision', 'recall']


        # 1. Identify indices based ONLY on the "accuracy" metric
        accuracy_vals = [m['accuracy'] for m in current_metrics_list]
        acc_mean_threshold = np.mean(accuracy_vals)

        # Keep track of indices that pass the accuracy filter
        kept_indices = [i for i, val in enumerate(accuracy_vals) if val > acc_mean_threshold]

        for m_name in metric_names:
            no_filtered_vals = [m[m_name] for m in current_metrics_list]
            clf_summary[f'{m_name}_no_filtered_mean'] = float(np.mean(no_filtered_vals))
            clf_summary[f'{m_name}_no_filtered_std'] = float(np.std(no_filtered_vals))
            clf_summary[f'{m_name}_no_filtered_all'] = no_filtered_vals

            # 2. Filter this metric using the PRE-CALCULATED accuracy indices
            filtered_vals = [no_filtered_vals[i] for i in kept_indices]

            clf_summary[f'{m_name}_mean'] = float(np.mean(filtered_vals)) if filtered_vals else 0.0
            clf_summary[f'{m_name}_std'] = float(np.std(filtered_vals)) if filtered_vals else 0.0
            clf_summary[f'{m_name}_all'] = filtered_vals

        # 3. Average Confusion Matrix using the same indices
        cms_all = [np.array(m['confusion_matrix']) for m in current_metrics_list]
        cms_filtered = [cms_all[i] for i in kept_indices]

        # Store Unfiltered CM stats
        clf_summary['confusion_matrix_no_filtered_mean'] = np.mean(cms_all, axis=0).tolist()

        # Store Filtered CM stats (aligned with accuracy performance)
        clf_summary['confusion_matrix_mean'] = np.mean(cms_filtered, axis=0).tolist()
        clf_summary['confusion_matrix_std'] = np.std(cms_filtered, axis=0).tolist()

        if xp_config.get("save_dca_showcase_data", False):
            clf_summary['probs_save_map'] = {
                key: [x for m in current_metrics_list for x in m['probs_save_map'][key]]
                for key in current_metrics_list[0]['probs_save_map'].keys()
            }

            clf_summary['n_splits'] = classification_config["n_splits"]
            clf_summary['batch_size'] = batch_size

        # Update checkpoint with the new summary section
        with open(checkpoint_path, 'w') as f:
            json.dump({'summary': summary, 'metrics_tracker': metrics_tracker, 'llms_map': llms_map}, f, indent=4)

    # Reconstruct the index mapping for the return value
    # If we resumed, we get llms_map from the checkpoint or the last loop
    if 'llms_map' not in locals():
        with open(checkpoint_path, 'r') as f:
            llms_map = json.load(f)['llms_map']

    nex_var_models_idx = revert_dictionnary(llms_map)
    return summary, nex_var_models_idx
