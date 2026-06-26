import logging
logger = logging.getLogger(__name__)

import math
from collections import Counter
from scipy.stats import chi2

def digit_frequency_test(digit_data: str, alpha: float = 0.01, verbose: bool = False):
    """
    Generalisation of the monobit/frequency test to decimal digits 0–9.
    
    We count occurrences of each digit, compare to the expected count N/10,
    and compute the chi-squared statistic:
    
        χ² = Σ_i (obs_i - E)² / E,    E = N/10,    df = 9
    
    :param digit_data:  The sequence of digits (each should be '0'–'9').
    :param alpha:       Significance level (default 0.01).
    :param verbose:     If True, print detailed counts and intermediate values.
    :return:            (chi2_stat, p_value, is_random)
    """
    N = len(digit_data)
    if N == 0:
        raise ValueError("Input string must be non-empty")
    
    counts = Counter(digit_data)
    # ensure only digits 0–9
    bad = [d for d in counts if d not in "0123456789"]
    if bad:
        raise ValueError(f"Invalid characters in input: {bad}")
    
    expected = N / 10.0
    chi2_stat = sum((counts[str(d)] - expected) ** 2 / expected for d in range(10))
    p_value = 1 - chi2.cdf(chi2_stat, df=9)
    
    if verbose:
        logger.debug("Digit Frequency Test (χ² Goodness-of-Fit) DEBUG BEGIN:")
        logger.debug(f"\tTotal length N:\t\t{N}")
        logger.debug(f"\tExpected count per digit:\t{expected:.3f}")
        for d in range(10):
            logger.debug(f"\tCount of '{d}':\t\t{counts[str(d)]}")
        logger.debug(f"\tχ² statistic:\t\t{chi2_stat:.4f}")
        logger.debug(f"\tDegrees of freedom:\t9")
        logger.debug(f"\tP-value:\t\t{p_value:.6f}")
        logger.debug("DEBUG END.")
    
    return p_value, (p_value >= alpha), chi2_stat


import math
from scipy.special import gammaincc
import random

def block_frequency_digits(decimal_data: str, block_size: int = 128, verbose: bool = False):
    """
    Block Frequency Test generalized to decimal digits (0-9).
    The purpose is to determine whether the frequency of each digit within M-digit blocks
    is approximately 1/10, as expected under randomness.

    :param decimal_data:  The sequence of decimal digits as a string, e.g. "0123456789..."
    :param block_size:    The length M of each block to test.
    :param verbose:       If True, print debug information.
    :return:              (p_value, is_random, chi2_statistic, df)
    """
    n = len(decimal_data)
    if n < block_size:
        block_size = n

    # Number of full blocks
    num_blocks = n // block_size
    if num_blocks == 0:
        raise ValueError("Insufficient data for even one block.")

    # Expected proportion
    p0 = 1.0 / 10.0

    chi2_sum = 0.0
    for i in range(num_blocks):
        block = decimal_data[i * block_size:(i + 1) * block_size]
        # count frequencies of digits 0-9
        counts = [0] * 10
        for ch in block:
            counts[int(ch)] += 1
        # compute proportions
        props = [count / block_size for count in counts]
        # block chi-square contribution: 10*M * sum((pi - p0)^2)
        sq = sum((pi - p0) ** 2 for pi in props)
        chi2_block = 10.0 * block_size * sq
        chi2_sum += chi2_block

    # degrees of freedom: num_blocks * (categories - 1)
    df = num_blocks * 9
    # compute p-value via upper tail of chi-square: P(Chi2_df >= chi2_sum)
    p_value = gammaincc(df / 2.0, chi2_sum / 2.0)

    if verbose:
        logger.debug("Block Frequency Test (Decimal) DEBUG BEGIN:")
        logger.debug(f"  Total length n: {n}")
        logger.debug(f"  Block size M: {block_size}")
        logger.debug(f"  Number of blocks: {num_blocks}")
        logger.debug(f"  Chi-square statistic: {chi2_sum:.6f}")
        logger.debug(f"  Degrees of freedom: {df}")
        logger.debug(f"  P-value: {p_value:.6f}")
        logger.debug("DEBUG END.")

    return p_value, (p_value >= 0.01), chi2_sum


from collections import Counter
from math import sqrt, erfc, fabs

def runs_test_decimal(data: str, alpha: float = 0.01, verbose: bool = False):
    """
    Runs test for decimal sequences (digits '0'–'9').

    :param data:   String of digits, each between '0' and '9'.
    :param alpha:  Significance level (default 0.01).
    :param verbose: If True, print intermediate values.
    :return:       (p_value, is_random: bool, runs_count R)
    """
    N = len(data)
    if N < 2:
        raise ValueError("Need at least 2 digits for runs test")
    # Validate
    bad = [c for c in data if c not in "0123456789"]
    if bad:
        raise ValueError(f"Invalid characters: {set(bad)}")
    # Count total changes
    V = sum(1 for i in range(1, N) if data[i] != data[i-1])
    R = V + 1

    # Under H0: q = P(change) = 1 - sum p_i^2 = 1 - 10*(1/10)^2 = 0.9
    q = 0.9
    E_V = (N - 1) * q
    Var_V = (N - 1) * q * (1 - q)
    if Var_V == 0:
        raise ValueError("Variance zero—sequence too short?")

    # Z-score and p-value
    Z = (V - E_V) / sqrt(Var_V)
    p_value = erfc(fabs(Z) / sqrt(2))
    is_random = (p_value >= alpha)

    if verbose:
        logger.debug("Runs Test (decimal) DEBUG BEGIN:")
        logger.debug(f"\tSequence length N:\t\t{N}")
        logger.debug(f"\tNumber of runs R:\t\t{R}")
        logger.debug(f"\tNumber of changes V=R-1:\t{V}")
        logger.debug(f"\tExpected V under H0:\t\t{E_V:.3f}")
        logger.debug(f"\tVar(V) under H0:\t\t{Var_V:.3f}")
        logger.debug(f"\tZ = (V - E_V)/sqrt(Var):\t{Z:.4f}")
        logger.debug(f"\tP-value:\t\t\t{p_value:.6f}")
        logger.debug("DEBUG END.")

    return p_value, is_random, R



from audit_llm.Bits_Generation.NIST_Tests_lib.Adapted_tests_to_digits import *

def generate_uniform_sequence(length: int) -> str:
    """Generate a uniform random decimal digit sequence."""
    return ''.join(str(random.randint(0, 9)) for _ in range(length))


def generate_biased_sequence(length: int, bias_digit: int = 7, bias_prob: float = 0.25) -> str:
    """Generate a sequence where `bias_digit` appears with probability `bias_prob`, others share the rest."""
    seq = []
    for _ in range(length):
        if random.random() < bias_prob:
            seq.append(str(bias_digit))
        else:
            seq.append(str(random.randint(0, 8) if bias_digit == 9 else random.choice([d for d in range(10) if d != bias_digit])))
    return ''.join(seq)

# -------------------------
# Example Test Cases
# -------------------------
def run_tests():
    test_lengths = [300, 1000, 10000]
    block_size = 100

    test_functions = [
        ("Block Frequency", block_frequency_digits),
        ("digit_frequency_test", digit_frequency_test),
        ("runs_test_decimal", runs_test_decimal)
        # ("Other Test Name", other_test_function),  # Add more as needed
    ]

    sequence_generators = [
        ("Uniform Sequences", generate_uniform_sequence),
        ("Biased Sequences (digit 7 biased 50%)", generate_biased_sequence)
    ]

    for test_name, test_func in test_functions:
        logger.debug(f"\n=== {test_name} ===")

        for gen_name, gen_func in sequence_generators:
            logger.debug(f"\n-- {gen_name} --")
            for L in test_lengths:
                data = gen_func(L)
                try:
                    p_val, is_rand, chi2  = test_func(data)
                    logger.debug(f"Length={L:6d}, p-value={p_val:.5f}, random={is_rand}, chi2={chi2:.2f}")
                except Exception as e:
                    logger.debug(f"Length={L:6d}, Error: {e}")

if __name__ == "__main__":
    run_tests()

