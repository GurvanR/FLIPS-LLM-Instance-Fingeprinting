"""Predefined token-pair sets used for bit-sequence experiments.

.. note:: This module performs non-trivial computation at **import time**
   (calls to ``get_random_token_pairs_from_intersection``).  If import
   latency becomes an issue, consider lazy-initialising these constants
   behind a builder function.
"""

from typing import List, Dict
from audit_llm.Tokens_analysis.token_sampling import get_random_token_pairs_from_intersection
from audit_llm.Bits_Generation.parsing_bits_tools import token_pair_to_string
monochar_token_pairs = get_random_token_pairs_from_intersection(nb_of_uplets=10, seed=70,monochar=True )
flips_token_pairs = get_random_token_pairs_from_intersection(nb_of_uplets=20, seed=70, monochar=False )


full_monochar_token_pairs = get_random_token_pairs_from_intersection(nb_of_uplets=50, seed=70,monochar=True )
full_flips_token_pairs = get_random_token_pairs_from_intersection(nb_of_uplets=100, seed=70, monochar=False )
icml_token_pairs = get_random_token_pairs_from_intersection(nb_of_uplets=30, seed=70, monochar=False )

SMALL_SET_OF_TOKEN_PAIRS= [['0','1']] + monochar_token_pairs + flips_token_pairs
TOY_SET_OF_TOKEN_PAIRS = [['0','1']] + monochar_token_pairs[:1] + flips_token_pairs[:2]
FULL_SET_OF_TOKEN_PAIRS = [['0','1']]  + full_monochar_token_pairs + full_flips_token_pairs
ICML_SET_OF_TOKEN_PAIRS= [['0','1']] + icml_token_pairs
#SP_SIZE_10 = random.Random(70).sample(load_SP(), 10), # choosing 10 random system prompts with fixed seed 70

TOKEN_PAIRS_SETS_DICT: Dict[str, List[str]]= {
    'NO_TOKEN_PAIRS': ['no_token_pairs'],
    'SMALL_SET_OF_TOKEN_PAIRS': [token_pair_to_string(token_pair) for token_pair in SMALL_SET_OF_TOKEN_PAIRS],
    'TOY_SET_OF_TOKEN_PAIRS': [token_pair_to_string(token_pair) for token_pair in TOY_SET_OF_TOKEN_PAIRS],
    'FULL_SET_OF_TOKEN_PAIRS': [token_pair_to_string(token_pair) for token_pair in FULL_SET_OF_TOKEN_PAIRS],
    'ICML_SET_OF_TOKEN_PAIRS': [token_pair_to_string(token_pair) for token_pair in ICML_SET_OF_TOKEN_PAIRS],

}