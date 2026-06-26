# Classification Module

---

## Folder structure

| File | Description |
|------|-------------|
| `__init__.py` | Public API re-exports |
| `single_classification.py` | Core single-token-pair classification pipeline with multi-split CV |
| `multi_classification.py` | Multi-token-pair batch classification: tp-wise, mix-at-pred, mix-at-train |
| `openset_classification.py` | Open-set classification via composition with known/unknown class splitting |
| `classify_single.py` | Orchestration for `classify()` entry point |
| `classify_cross.py` | Orchestration for `classify_cross_token_pairs()` |
| `classify_batch.py` | Orchestration for `batch_classification_across_token_pairs()` |
| `Preprocessing_data.py` | NaN imputation, per-feature normalization, `FeatureNormalizer` |
| `classification_constants.py` | Classifier templates, CV splitter map, metric definitions |
| `token_pair_mixing.py` | Combinatorial/random mixing of feature samples across token pairs |
| `batch_results.py` | Metric summarization, per-class accuracy from confusion matrices |
| `results_tables.py` | Statistical table generation (Markdown, LaTeX, NxM), heatmap orchestration |
| `results_plotting.py` | Cross-accuracy heatmaps, per-class accuracy bar plots |
| `confusion_matrix_utils.py` | Confusion matrix grouping/reordering, bimodal-colormap plotting |
| `training_size_analysis.py` | Training-size curve plots, per-unique-tp curves |
| `feature_importance.py` | Feature importance bar plots |
| `Feature_Visualization.py` | Sequence length CCDF, p-value export, NIST performance charts |
| `dca_analysis.py` | DCA: Wasserstein distance, abliterated-vs-safe model plotting |

---

## Three entry points

The Classification module has three main entry points:

1. **`classify(Experiment_config)`** (`classify_single.py`) — Single token-pair classification.
   Iterates over calculation items (token pairs × other iterators), runs
   `SingleTokenPairClassification.fit_evaluate()` per item.

2. **`classify_cross_token_pairs(Experiment_config)`** (`classify_cross.py`) — Train on one token
   pair, test on another. Produces NxN heatmaps of cross-generalization.

3. **`batch_classification_across_token_pairs(Experiment_config)`** (`classify_batch.py`) —
   Multi-token-pair batch classification. Loads all token pairs, creates
   `MultiTokenPairClassification` instances, runs batch classification with varying train sizes and
   batch types (`tp_wise`, `mix_tp_at_pred`, `mix_tp_at_train`).

---

## Data flow

### Preparation pipeline

```
Raw X: shape (n_samples, n_models, n_features)
         |
         v
    prepare_data_common()
    |-- preprocess_X(): NaN imputation per model column
    |-- Flatten to 2D: (n_samples * n_models, n_features)
    |-- Drop all-NaN rows
    +-- Re-index labels contiguously
         |
         v
    y: integer class labels (0..n_remaining_models-1)
    X: (n_total_samples, n_features)
         |
         v
    [train/test split via StratifiedShuffleSplit]
         |
         v
    balance() [training data only]
         |
         v
    fit_transform_normalize(X_train, X_test)
    |-- Creates FeatureNormalizer
    |-- Per-feature normalization (standard/minmax/robust/power/quantile/log/none/auto)
    |-- Fit on training data, transform both train and test
         |
         v
    Ready for classification
```

### Training & evaluation

**`SingleTokenPairClassification.fit_evaluate()`:**
- Multi-split CV via `StratifiedShuffleSplit`
- Per split: balance train → normalize → train all classifiers → predict on test → record metrics
- Classifiers cloned from `CLASSIFIERS_TEMPLATES_MAP`; balanced class weights applied automatically
- Batch prediction supported: aggregate via soft voting, hard voting, or decision function

**`MultiTokenPairClassification.batch_classification()`:**
- Routes to `tp_wise`, `mix_tp_at_pred`, or `mix_tp_at_train`
- `tp_wise`: per-split, trains and predicts on each token pair independently, then batch-aggregates
- `mix_tp_at_pred`: same training as tp_wise, but test batches mix samples from different token
  pairs (uplets)
- `mix_tp_at_train`: concatenate features from multiple token pairs before training

#### Training budget truncation (`mix_tp_at_pred`)

In `mix_tp_at_pred`, per-token-pair training sets are truncated to keep total training budget equal
to `train_size`:

```python
truncation = self.train_size // min(bs, self.unique_tp_in_mix)
```

This ensures a prediction batch of size `bs` aggregates training data totalling `train_size`
samples per class.

### Results collection

- Metrics (accuracy, precision, recall, F1, balanced_accuracy) recorded per split per classifier
- Confusion matrices stored per split, then averaged
- `summarize_metrics()` produces:
  `summary[batch_type][batch_size][dataset][clf][metric_mean/std]`

---

## Train/test leakage — design guarantees

1. `fit_evaluate()`: split → balance on train only → `fit_transform_normalize` fit on train → train
   → predict on test. No leakage.
2. Cross-token-pair: train and test are separate token-pair arrays; no leakage.
3. Multi-token-pair: `_prepare_data()` calls `prepare_data_common()` only (no balancing before
   split).
4. `test_size` is an integer (samples per class), making the actual test set size explicit.

---

## Configuration keys

Classification behavior is controlled by `xp_config["classification_config"]`:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `classifiers` | `List[str]` | `["XGBoost"]` | Names from `CLASSIFIERS_TEMPLATES_MAP` (`XGBoost`, `Random Forest`, `Logistic Regression`, `SVM`, `KNN`, `MLP`, …) |
| `n_splits` | `int` | `1` | Number of train/test splits |
| `splitter_type` | `str` | `"StratifiedShuffleSplit"` | CV splitter |
| `test_size` | `int` | `64` | Test samples **per class** |
| `random_seed` | `int` | `42` | Random seed |
| `force_class_size` | `int\|"auto"\|None` | `"auto"` | Per-class cap: `"auto"`=balance to smallest class (undersample only); `int`=cap each class to ≤N (must be `> test_size`); `None`=no balancing |
| `batch_prediction_sizes` | `List[int]` | `[1]` | Batch sizes for `MultiTokenPairClassification` |
| `batch_types` | `List[str]` | derived | Batch modes; derived from `batch_prediction_sizes` when unset (size `1` → `tp_wise`, size `>1` → `mix_tp_at_pred`) |
| `unique_tp_in_mix` | `int\|List[int]\|str` | `['max']` | Distinct token pairs per mixed batch (`'max'` = `bs`; needs `C(n_tp, bs) >= max_nb_of_uplet` for each plotted `bs`) |
| `openset` | `bool` | `False` | Enable open-set classification |
| `default_normalization` | `str` | `"auto"` | Default normalization method |
| `compute_confusion_matrices` | `bool` | `False` | Compute and plot confusion matrix heatmaps |
| `train_size_dict_map` | `Dict[str,str]` | `None` | "Merged-wrapper mode": load named checkpoints for comparison plots |

---

## Group-based classification

By default each model is its own class. `model_groups` in the experiment config groups models into
named categories so the classifier predicts the category instead.

```yaml
model_groups:
  abliterated:
    - failspy/Meta-Llama-3-8B-Instruct-abliterated-v3
  non-abliterated:
    - meta-llama/Llama-3.1-8B-Instruct
```

Models not listed in any group are excluded. Confusion matrices and accuracy plots automatically
show group names.

---

## Related documentation

| File | Content |
|------|---------|
| [features.md](features.md) | Feature computation: the 3D arrays consumed here |
| [experiment.md](experiment.md) | How experiment dispatch invokes the three entry points |
| [configuration.md](configuration.md) | `ClassificationConfig` and `model_groups` full schema |
