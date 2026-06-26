"""Classification — ML classification pipeline for LLM audit.

Public API re-exports for clean imports::

    from audit_llm.Classification import classify, SingleTokenPairClassification, ...
"""

from audit_llm.Classification.classify_batch import batch_classification_across_token_pairs
from audit_llm.Classification.classify_cross import classify_cross_token_pairs
from audit_llm.Classification.classify_single import classify
from audit_llm.Classification.Feature_Visualization import (
    Nist_perf_chart,
    Save_pv_in_parquet,
    Seq_Length_visualization,
    Valid_count_chart,
)
from audit_llm.Classification.multi_classification import MultiTokenPairClassification
from audit_llm.Classification.openset_classification import OpenSetClassification
from audit_llm.Classification.Preprocessing_data import FeatureNormalizer, fit_transform_normalize
from audit_llm.Classification.single_classification import SingleTokenPairClassification

__all__ = [
    # Orchestration entry points
    "batch_classification_across_token_pairs",
    "classify",
    "classify_cross_token_pairs",
    # Classification classes
    "SingleTokenPairClassification",
    "MultiTokenPairClassification",
    "OpenSetClassification",
    # Preprocessing
    "FeatureNormalizer",
    "fit_transform_normalize",
    # Feature visualization
    "Nist_perf_chart",
    "Save_pv_in_parquet",
    "Seq_Length_visualization",
    "Valid_count_chart",
]
