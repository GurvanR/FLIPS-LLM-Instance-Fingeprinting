"""Bits_Generation — bit-sequence analysis pipeline.

Public API re-exports for clean imports::

    from audit_llm.Bits_Generation import compute_similarities, ...
"""

from audit_llm.Bits_Generation.Bits_Seqs_similarity import (
    SIM_FUNC_MAP,
    compute_similarities,
)
from audit_llm.Bits_Generation.bits_tools import (
    compute_intra_samples_bit_feature_matrix,
    fill_inter_samples_features_map,
    make_random_bit_sequences,
)
from audit_llm.Bits_Generation.parsing_bits_tools import (
    answer_to_bit_string,
    bits_token_pair_to_scrapper,
    compute_proper_bit_sequences,
    token_pair_name_to_items,
    token_pair_to_string,
)
from audit_llm.Bits_Generation.Token_Pairs_Sets import TOKEN_PAIRS_SETS_DICT
