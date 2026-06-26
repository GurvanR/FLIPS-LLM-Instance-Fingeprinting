import logging
logger = logging.getLogger(__name__)

from numpy import abs as abs
from numpy import array as array
from numpy import floor as floor
from numpy import max as max
from numpy import sqrt as sqrt
from numpy import sum as sum
from numpy import zeros as zeros
from scipy.stats import norm as norm

class CumulativeSums:

    @staticmethod
    def cumulative_sums_test(binary_data: str, mode=0, verbose=False, terms='one'):
        """
        NIST SP800-22 Cumulative Sums Test.

        :param binary_data: A binary string (e.g., '1100101').
        :param mode: 0 = forward, 1 = backward.
        :param verbose: Print debug output if True.
        :param terms: 'one', 'two', or 'both' to determine term summation.
        :return: (p_value: float, passed: bool, terms_used: list of floats)
        """
        length_of_binary_data = len(binary_data)
        counts = zeros(length_of_binary_data)

        if mode != 0:
            binary_data = binary_data[::-1]

        for i, bit in enumerate(binary_data):
            step = 1 if bit == '1' else -1
            counts[i] = counts[i - 1] + step if i > 0 else step

        abs_max = max(abs(counts))

        # Abort early if abs_max is zero (flat walk) or data is too short
        if abs_max == 0 or length_of_binary_data < 10:
            if verbose:
                logger.debug("Data too short or flat for meaningful cumulative sums test.")
            return (0.0, False, [])

        # First term range
        start_1 = int(floor(0.25 * floor(-length_of_binary_data / abs_max + 1)))
        end_1 = int(floor(0.25 * floor(length_of_binary_data / abs_max - 1)))

        terms_one = []
        if terms in ('one', 'both'):
            for k in range(start_1, end_1 + 1):
                sub = norm.cdf((4 * k - 1) * abs_max / sqrt(length_of_binary_data))
                terms_one.append(norm.cdf((4 * k + 1) * abs_max / sqrt(length_of_binary_data)) - sub)

        # Second term range
        start_2 = int(floor(0.25 * floor(-length_of_binary_data / abs_max - 3)))
        end_2 = int(floor(0.25 * floor(length_of_binary_data / abs_max) - 1))

        terms_two = []
        if terms in ('two', 'both'):
            for k in range(start_2, end_2 + 1):
                sub = norm.cdf((4 * k + 1) * abs_max / sqrt(length_of_binary_data))
                terms_two.append(norm.cdf((4 * k + 3) * abs_max / sqrt(length_of_binary_data)) - sub)

        # Compute p-value
        p_value = 0.0
        if terms == 'one':
            p_value = 1.0 - sum(array(terms_one))
            terms_used = terms_one
        elif terms == 'two':
            p_value = sum(array(terms_two))
            terms_used = terms_two
        elif terms == 'both':
            p_value = 1.0 - sum(array(terms_one)) + sum(array(terms_two))
            terms_used = terms_one + terms_two
        else:
            raise ValueError("Invalid value for 'terms'. Expected 'one', 'two', or 'both'.")

        if verbose:
            logger.debug('Cumulative Sums Test DEBUG BEGIN:')
            logger.debug("\tLength of input:\t", length_of_binary_data)
            logger.debug('\tMode:\t\t\t\t', mode)
            logger.debug('\tValue of z:\t\t\t', abs_max)
            logger.debug('\tP-Value:\t\t\t', p_value)
            logger.debug('DEBUG END.')

        return (p_value, (p_value >= 0.01), sum(array(terms_used))) # we sum to get single float for less features

if __name__=="__main__":
    data = "01011010100101101101100110100001101100001111001001111011101100101100010101011100010010010001101101101011010100101101101100110100001101100001111001001111101110110010110001010101110001001001000110110110101101010010110110110011010000110110000111100100111110111011001011000101010111000100100100011011011"

    logger.debug(CumulativeSums.cumulative_sums_test(data, terms='one'))