import logging
import math
from collections import Counter
from typing import Any, Callable, Dict, List, Tuple, Union

logger = logging.getLogger(__name__)

import networkx as nx
import numpy as np
from tqdm.notebook import tqdm

from audit_llm.Bits_Generation.Bits_Seqs_similarity import compute_similarities
from audit_llm.Bits_Generation.NIST_Tests_lib.Full_Nist_Testsuite import Nist_testing, Nist_index, NIST_TESTS
from audit_llm.Tokens_analysis.vocab_io import load_tokenizer_vocabs
from audit_llm.Tokens_analysis.token_decoding import token_decoder


def extract_proper_graph(model_df, prompt_config, compute_config, dataset):
    """
    Run the following snippet to undestand the function.

    import networkx as nx
    n_Renyi=6
    edge_list_0=[(1,3), (4,6)]
    edge_list_1=[(1,3),(4,6),(5,8)]
    edge_list_2=[(1,3),(4,6),(0,2)]

    graph= nx.Graph(edge_list)
    for node in sorted(graph.nodes(), reverse=True):
        if node > n_Renyi:
            graph.remove_node(node)
        elif node == n_Renyi:
            if 0 in graph.nodes: 
                graph.remove_node(n_Renyi)
            else:
                # Create a mapping: old_node -> new_node
                mapping = {node: node - 1 for node in graph.nodes()}
                graph = nx.relabel_nodes(graph, mapping)

    graph.add_nodes_from(range(n_Renyi))
    logger.debug("graph.edges=%s, graph.nodes=%s", graph.edges, graph.nodes)
    
    """
    graph_config=prompt_config['graph_type_config']
    n_Renyi=graph_config['nb_of_nodes']

    edge_lists = model_df["Answer"].to_list()
    graphs=[]
    for edge_list in edge_lists:
        graph= nx.Graph(edge_list)
        for node in sorted(graph.nodes(), reverse=True):
            if node > n_Renyi:
                graph.remove_node(node)
            elif node == n_Renyi:
                if 0 in graph.nodes: 
                    graph.remove_node(n_Renyi)
                else:
                    # Create a mapping: old_node -> new_node
                    mapping = {node: node - 1 for node in graph.nodes()} # We relabel, which is good to handle graphs, but surely, we have to take labels as produced by the LLM when it comes to analyse the output as characters.
                    graph = nx.relabel_nodes(graph, mapping)

        graph.add_nodes_from(range(n_Renyi))

        graphs.append(graph)
        
        
    #graphs=[nx.Graph([edge for edge in edge_list if edge[0] != edge[1]])for edge_list in graph_lists] # remove self-loops. NOW DONE IN SCRAPPING
    graphs=[graph for graph in graphs if len(graph.nodes())<=n_Renyi]
    
    return {'extracted_seq_answers': graphs}

def make_random_bit_sequences(model: str, max_tokens: int, N: int = 100, random_type: str = '', seed: int = 1000) -> Dict[str, List[str]]:
    """
    Returns a list of random bit sequences generated using a specified PRNG model.

    Parameters:
    - model: str
        The PRNG model to use. Options:
            ['numpy_default', 'numpy_mt19937', 'numpy_pcg64', 'numpy_sfc64',
             'python_random', 'secrets', 'xor_shift', 'lcg']
    - max_tokens: int
        Number of bits per sequence.
    - N: int
        Number of sequences to generate.
    - random_type: str
        Reserved for future extensions.
    - seed: int
        Seed for reproducibility.

    Returns:
    - Dict with:
        'extracted_seq_answers' : List[str]
        'extracted_bit_seqs'    : List[str]
    """

    import random
    import secrets

    nb_of_bits = max_tokens
    sequences: List[str] = []

    # --- Initialize PRNG based on selected model ---
    if model == 'numpy_default':
        np.random.seed(seed)
        rng = lambda size: np.random.randint(0, 2, size=size)

    elif model == 'numpy_mt19937':
        gen = np.random.Generator(np.random.MT19937(seed))
        rng = lambda size: gen.integers(0, 2, size=size)

    elif model == 'numpy_pcg64':
        gen = np.random.Generator(np.random.PCG64(seed))
        rng = lambda size: gen.integers(0, 2, size=size)

    elif model == 'numpy_sfc64':
        gen = np.random.Generator(np.random.SFC64(seed))
        rng = lambda size: gen.integers(0, 2, size=size)

    elif model == 'python_random':
        random.seed(seed)
        rng = lambda size: [random.randint(0, 1) for _ in range(size)]

    elif model == 'secrets':
        rng = lambda size: [secrets.randbits(1) for _ in range(size)]

    elif model == 'xor_shift':
        def xor_shift(seed_val: int):
            x = seed_val & 0xFFFFFFFFFFFFFFFF
            while True:
                x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
                x ^= (x >> 7)
                x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
                yield x & 1
        bit_gen = xor_shift(seed)
        rng = lambda size: [next(bit_gen) for _ in range(size)]

    elif model == 'lcg':
        raise ValueError("not working for the moment, only 010101010101 pattern everytime.")
        a, c, m = 1664525, 1013904223, 2**32
        state = seed
        def lcg_bits():
            nonlocal state
            while True:
                state = (a * state + c) % m
                yield state & 1
        bit_gen = lcg_bits()
        rng = lambda size: [next(bit_gen) for _ in range(size)]

    else:
        raise ValueError(f"Unknown PRNG model: {model}")

    # --- Generate bit sequences ---
    for _ in range(N):
        bits = rng(nb_of_bits)
        sequence = "".join(str(int(b)) for b in bits)
        sequences.append(sequence)

    return {
        'extracted_seq_answers': sequences,
        'extracted_bit_seqs': sequences
    }





def compute_intra_samples_bit_feature_matrix(extracted_seq_answers_dict: Dict[str, List], N_iter:int,
                                  compute_config:dict, dataset: str) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Computes a feature matrix for a list of bits sequences. Each row in the matrix corresponds to
    one graph and columns represent various network features.
    """

    features_config = compute_config['features']

    rng_tests_to_use = features_config['nist_tests']

    test_features_index=Nist_index(rng_tests_to_use)

    intra_samples_features_index={name: idx for idx, name in enumerate(features_config['intra_samples_features'], start=len(test_features_index))}
    
    features_index = {**test_features_index, **intra_samples_features_index}
    
    token_stats_dict=compute_config.get('token_stats_dict', None)
    nb_of_test_and_intra_features=len(features_index)

    if token_stats_dict is not None:
        token_union_set: List = token_stats_dict[dataset]['union']

        token_union_set_length = len(token_union_set)
        token_active_frequency_feature_index={f"active_token_freq: {token}": feature_idx for feature_idx, token in enumerate(
            token_union_set, start=len(features_index))}
        
        token_passive_frequency_feature_index={f"passive_token_freq: {token}": feature_idx for feature_idx, token in enumerate(
            token_union_set, start=token_union_set_length + len(features_index))}
        
        features_index = {**features_index, **token_active_frequency_feature_index, **token_passive_frequency_feature_index}

    feature_vectors = np.full((N_iter, len(features_index)), np.nan)

    constant_len = compute_config.get('set_constant_seq_length', None)

    bit_sequences = extracted_seq_answers_dict['extracted_bit_seqs']
    bit_sequences_bar= tqdm(bit_sequences, desc=f"Processing intra samples features on every samples", position=3, leave=False)
    for i, bit_seq in enumerate(bit_sequences_bar):
        original_len = len(bit_seq)

        if constant_len is not None:
            if original_len < constant_len:
                feature_vector = [np.nan] * len(test_features_index)
                feature_vector.append(original_len)
                feature_vectors[i, :nb_of_test_and_intra_features] = feature_vector
                continue
            bit_seq = bit_seq[:constant_len]

        if len(bit_seq) > compute_config['minimal_seq_length']:
            feature_vector=Nist_testing(bit_seq, rng_tests_to_use)
        else:
            feature_vector= [np.nan]*len(test_features_index)

        feature_vector.append(original_len)
        feature_vectors[i, :nb_of_test_and_intra_features] = feature_vector
    

    # Token related features
    if token_stats_dict is not None:
        raise NotImplementedError("Token related features computation is not implemented yet.")
        model_name= compute_config['model']
        if model_name != 'True_rng':
            # Adding token frequency within active manner    
            extracted_seq_token_ids = extracted_seq_answers_dict['extracted_seq_token_ids'] # List[int]
            if token_union_set is not None:
                model_vocab: Dict[str, int] | None = load_tokenizer_vocabs().get(model_name, None) # model_vocab = {token: id}. None values typically for proprietary models.
                
                if model_vocab is not None:
                    id_to_token= {id: token for token, id in model_vocab.items()} # we reverse it

                    for i, bit_token_ids_seq in enumerate(extracted_seq_token_ids):
                        feature_vector = []

                        bit_tokens_seq = [id_to_token.get(tok_id, f"<UNK:{tok_id}>") for tok_id in bit_token_ids_seq]
                        for token in token_union_set:
                            token_count=bit_tokens_seq.count(token)
                            feature_vector.append(token_count)
                        
                        feature_vectors[i, nb_of_test_and_intra_features: nb_of_test_and_intra_features + token_union_set_length] = feature_vector # filling the rest of the vector
                else:
                    logger.warning("No token vocab found for %s", model_name)
                    
            
            # Adding token frequency within passive manner
            extracted_seq_answers = extracted_seq_answers_dict['extracted_seq_answers'] 
            # As a reminder, here the extracted_seq_answers are the raw full answers of llms.
            if token_union_set is not None:
                for i, answer in enumerate(extracted_seq_answers):
                    feature_vector = []
                    for token in token_union_set:
                        token_count=answer.count(token_decoder(token)) # non-overlapping count
                        feature_vector.append(token_count)
                    
                    feature_vectors[i, nb_of_test_and_intra_features + token_union_set_length:] = feature_vector # filling the rest of the vector
        else:
            pass # Values will stay NaN for rng model.

    return feature_vectors, features_index

def compute_bits_covariance(filtered_sequences: List[str],
                             array_bit_seqs: np.ndarray,
                            seq_length_for_covariance:int,
                              compute_config:Dict) -> Dict[str, np.ndarray]:
    """
    Args: 
    - array_bit_seqs: shape (len(filtered_sequences), seq_length_for_inter_sample_features)
    Computes covariance matrix over the bit_sequences. 
    
    Computes the covariance matrix that will be shaped (len(filtered_sequences), len(filtered_sequences))
    """

    if len(filtered_sequences) == 0:
        # Return zero matrix if no valid sequences
        return {
            'covariance_matrix_across_samples': np.empty((0, 0)),
            'covariance_matrix_across_bits': np.empty((0, 0)),
        }

    return {
        'covariance_matrix_across_samples': np.cov(array_bit_seqs),
        'covariance_matrix_across_bits': np.cov(array_bit_seqs, rowvar=False),
    }

def compute_similarities_on_bit_sequences(filtered_sequences: List[str], array_bit_seqs: np.ndarray,
                                           seq_length_for_covariance:int, compute_config:Dict) -> Dict[str, np.ndarray]:
    """
    Args: 
    - array_bit_seqs: shape (len(filtered_sequences), seq_length_for_inter_sample_features)
    """
    if len(filtered_sequences) == 0:
        return {}
    
    metrics: List[str] | None = compute_config.get('similarity_metrics', None)
    k_list: List[int] | None = compute_config.get('similarity_k_list', None)
    strategies: List[str] | None = compute_config.get('similarity_agg_strategies', None)

    scores_matrices, aggregated_scores = compute_similarities(filtered_sequences, metrics, k_list, strategies, disable_tqdm= False)

    return {**scores_matrices, **aggregated_scores}



def block_entropy(bit_sequence, block_size=2):
    """
    Compute the block entropy of a binary sequence.
    
    Args:
        bit_sequence (str): A string of '0's and '1's, e.g., "0110101"
        block_size (int): Size of the block (n-gram length)
        
    Returns:
        float: Block entropy in bits
    """
    if len(bit_sequence) < block_size:
        raise ValueError("Block size must be smaller than sequence length.")
    
    # Build list of blocks
    blocks = [bit_sequence[i:i+block_size] for i in range(len(bit_sequence) - block_size + 1)]
    
    # Count block occurrences
    block_counts = Counter(blocks)
    total_blocks = sum(block_counts.values())
    
    # Compute probabilities
    probs = [count / total_blocks for count in block_counts.values()]
    
    # Compute entropy
    entropy = -sum(p * math.log2(p) for p in probs)
    
    return entropy

def compute_Bits_Block_Entropy(filtered_sequences: List[str], array_bit_seqs: np.ndarray,
                            seq_length_for_covariance:int, compute_config:Dict) -> Dict[str, np.ndarray]:
    """ 
    Args: 
    - array_bit_seqs: shape (len(filtered_sequences), seq_length_for_inter_sample_features)
    Returns: Dict[f'block_entropy_{block_size}': np.ndarray of shape (len(filtered_sequences), )
    """
    block_sizes=compute_config.get('block_entropy_block_sizes', list(range(1, 6)))
    if len(filtered_sequences) == 0:
        return {f'block_entropy_{block_size}': np.array([np.nan]) for block_size in block_sizes}

    return {f'block_entropy_{block_size}': np.array([block_entropy(bit_sequence, block_size) for bit_sequence in filtered_sequences])
             for block_size in block_sizes }



def compute_Bits_Bernoulli_probs(filtered_sequences: List[str], array_bit_seqs: np.ndarray,
                            seq_length_for_covariance:int, compute_config:Dict) -> np.ndarray:
    """ 
    Args: 
    - array_bit_seqs: shape (len(filtered_sequences), seq_length_for_inter_sample_features)
    Returns: np.ndarray of shape (seq_length_for_inter_sample_features, )

    Represents the probability to see a 1 for each position of the sequence. For now not used, the idea was to derive a likelihood from these probs and see if a good proxy for classification.
    """
    if len(filtered_sequences) == 0:
        return np.full(seq_length_for_covariance, np.nan)

    return np.mean(array_bit_seqs, axis=0) 


def fill_inter_samples_features_map(k: int,
                                    extracted_seq_answers_dict: Dict[str, List],
                                    compute_config: dict
                                   ) -> Dict[Tuple[int, int, str], Any]:
    """
    For a given model index `k`, computes a map of inter-sample features
    for all bit sequences in `bit_sequences`. Sequences shorter than
    `seq_length_for_inter_sample_features` are dropped; the rest are truncated to that length.

    Returns:
      - inter_samples_features_map: Dict[(k, feature_name) -> feature_value]
      - feature_index: Dict[feature_name -> column_index]
    """
    features_config = compute_config['features']
    inter_samples_features = features_config['inter_samples_features']
    seq_length = compute_config['seq_length_for_inter_sample_features']
    bit_sequences= extracted_seq_answers_dict['extracted_bit_seqs'] # List[str]
    # 1) Filter out too-short sequences and truncate the rest
    filtered_seqs = [
        seq[:seq_length]
        for seq in bit_sequences
        if len(seq) >= seq_length
    ]
    nb_kept = len(filtered_seqs)

    # 2) Convert to numeric array for vectorized routines
    array_bit_seqs = np.array([[int(b) for b in seq] for seq in filtered_seqs])

    inter_samples_features_map = {}

    # 3) Compute each requested inter-sample feature
    
    feature_bar = tqdm(inter_samples_features, desc=f"inter_samples_features", position=2, leave=False)
    for feat_name in feature_bar:
        feature_bar.set_description(f"Processing: {feat_name}")
        feat_val = INTER_SAMPLES_FEATURES_FUN[feat_name](
            filtered_seqs,
            array_bit_seqs,
            seq_length,
            compute_config
        )
        inter_samples_features_map[(k, feat_name)] = feat_val

    # 4) Also record how many sequences survived the length filter
    inter_samples_features_map[(k, 'nb_of_sequences_after_seq_length_filter')] = nb_kept

    return inter_samples_features_map

def rng_tests_dict_to_nb_of_tests(nist_tests: Dict):
    nb_of_test = 0
    for test, parameters in nist_tests.items():
        if parameters:
            nb_of_test += len(parameters)
        else:
            nb_of_test += 1
    return nb_of_test


INTER_SAMPLES_FEATURES_FUN ={
    'Bits_Covariance_matrix': compute_bits_covariance,
    'Similarities_on_bit_sequences': compute_similarities_on_bit_sequences,
    'Bits_Block_Entropy': compute_Bits_Block_Entropy,
    'Bits_Seqs_Bernoulli_probs': compute_Bits_Bernoulli_probs,
}