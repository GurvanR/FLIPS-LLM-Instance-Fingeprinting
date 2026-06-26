import logging
logger = logging.getLogger(__name__)

from math import isnan as isnan
from numpy import abs as abs
from numpy import append as append
from numpy import array as array
from numpy import clip as clip
from numpy import cumsum as cumsum
from numpy import ones as ones
from numpy import sqrt as sqrt
from numpy import sum as sum
from numpy import transpose as transpose
from numpy import where as where
from numpy import zeros as zeros
from scipy.special import erfc as erfc
from scipy.special import gammaincc as gammaincc

import numpy as np
from scipy.special import gammaincc
from math import nan

class RandomExcursions:
    
    @staticmethod
    def random_excursions_test(binary_data: str,
                            verbose: bool = False,
                            state: int = 1):
        """
        NIST SP800-22 Random Excursions Test for a single state.
        
        :param binary_data:  the input bit‐string (e.g. '010101...')
        :param verbose:      if True, prints detailed debug info
        :param state:        the state to test (must be one of [-4,-3,-2,-1,1,2,3,4])
        :return:             (p_value, passed, t_stat)
        """
        # Quick safety checks
        n = len(binary_data)
        if n == 0 or binary_data.count('0') == 0 or binary_data.count('1') == 0:
            return nan, False, nan

        # map bits → ±1
        seq = np.array([1 if b=='1' else -1 for b in binary_data], dtype=float)

        # cumulative sums S1…Sn, with padding zeros at both ends
        S = np.concatenate(([0], np.cumsum(seq), [0]))

        # our states of interest and index lookup
        x_values = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
        try:
            idx = list(x_values).index(state)
        except ValueError:
            raise ValueError(f"Invalid state {state}; must be one of {x_values.tolist()}")

        # find zero‐crossing positions → extract cycles
        zero_pos = np.where(S == 0)[0]
        if len(zero_pos) < 2:
            return nan, False, nan

        cycles = [ S[zero_pos[i]:zero_pos[i+1]+1] for i in range(len(zero_pos)-1) ]
        num_cycles = len(cycles)
        if num_cycles == 0:
            return nan, False, nan

        # count visits to each state in each cycle, clipped at 5
        visits = np.array([
            np.clip([np.count_nonzero(cycle == x) for x in x_values], 0, 5)
            for cycle in cycles
        ]).T    # shape = (8 states) × (num_cycles)

        # su[k,j] = # of cycles with exactly k visits to state j, for k=0…5
        # → shape (6 visit‐counts) × (8 states) → transpose → (8 states) × 6
        su = np.array([
            [(visits[j] == k).sum() for k in range(6)]
            for j in range(len(x_values))
        ])

        # expected probabilities π(k) for each (k=0…5) and each state
        pi = np.array([
            [ RandomExcursions.get_pi_value(k, x) for k in range(6) ]
            for x in x_values
        ])

        # compute χ² for each state
        inner = num_cycles * pi
        # guard against zero‐division
        with np.errstate(divide='ignore', invalid='ignore'):
            chi2 = ((su - inner)**2 / inner).sum(axis=1)

        # p‐values via the incomplete gamma
        p_vals = gammaincc(2.5, chi2/2.0)

        # pick out our requested state
        t_stat  = float(chi2[idx]) if np.isfinite(chi2[idx]) else nan
        p_value = float(p_vals[idx]) if np.isfinite(p_vals[idx]) else nan
        passed  = bool(p_value >= 0.01)

        if verbose:
            logger.debug("Random Excursions Test DEBUG:")
            logger.debug(f"  Input length = {n}, cycles = {num_cycles}")
            logger.debug("  State   χ²-stat    p-value    pass?")
            for i, x in enumerate(x_values):
                χ2i = chi2[i] if np.isfinite(chi2[i]) else nan
                pi_val = p_vals[i] if np.isfinite(p_vals[i]) else nan
                logger.debug(f"    {x:>2}    {χ2i:8.4f}    {pi_val:8.4f}    {pi_val>=0.01}")
            logger.debug("-- end DEBUG --")

        return p_value, passed, t_stat


    @staticmethod
    def variant_test(binary_data:str, verbose=False):
        """
        from the NIST documentation http://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf

        :param binary_data:
        :param verbose:
        :return:
        """
        length_of_binary_data = len(binary_data)
        int_data = zeros(length_of_binary_data)

        for count in range(length_of_binary_data):
            int_data[count] = int(binary_data[count])

        sum_int = (2 * int_data) - ones(len(int_data))
        cumulative_sum = cumsum(sum_int)

        li_data = []
        index = []
        for count in sorted(set(cumulative_sum)):
            if abs(count) <= 9:
                index.append(count)
                li_data.append([count, len(where(cumulative_sum == count)[0])])

        j = RandomExcursions.get_frequency(li_data, 0) + 1

        p_values = []
        for count in (sorted(set(index))):
            if not count == 0:
                den = sqrt(2 * j * (4 * abs(count) - 2))
                p_values.append(erfc(abs(RandomExcursions.get_frequency(li_data, count) - j) / den))

        count = 0
        # Remove 0 from li_data so the number of element will be equal to p_values
        for data in li_data:
            if data[0] == 0:
                li_data.remove(data)
                index.remove(0)
                break
            count += 1

        if verbose:
            logger.debug('Random Excursion Variant Test DEBUG BEGIN:')
            logger.debug("\tLength of input:\t", length_of_binary_data)
            logger.debug('\tValue of j:\t\t', j)
            logger.debug('\tP-Values:')
            logger.debug('\t\t STATE \t\t COUNTS \t\t P-Value \t\t Conclusion')
            count = 0
            for item in p_values:
                logger.debug('\t\t', repr(li_data[count][0]).rjust(4), '\t\t', li_data[count][1], '\t\t', repr(item).ljust(14), '\t\t', (item >= 0.01))
                count += 1
            logger.debug('DEBUG END.')


        states = []
        for item in index:
            if item < 0:
                states.append(str(item))
            else:
                states.append('+' + str(item))

        result = []
        count = 0
        for item in p_values:
            result.append((states[count], li_data[count][0], li_data[count][1], item, (item >= 0.01)))
            count += 1

        return None, result, None

    @staticmethod
    def get_pi_value(k, x):
        """
        This method is used by the random_excursions method to get expected probabilities
        """
        if k == 0:
            out = 1 - 1.0 / (2 * abs(x))
        elif k >= 5:
            out = (1.0 / (2 * abs(x))) * (1 - 1.0 / (2 * abs(x))) ** 4
        else:
            out = (1.0 / (4 * x * x)) * (1 - 1.0 / (2 * abs(x))) ** (k - 1)
        return out

    @staticmethod
    def get_frequency(list_data, trigger):
        """
        This method is used by the random_excursions_variant method to get frequencies
        """
        frequency = 0
        for (x, y) in list_data:
            if x == trigger:
                frequency = y
        return frequency
    

if __name__=="__main__":
    data = "01011010100101101101100110100001101100001111001001111011101100101100010101011100010010010001101101101011010100101101101100110100001101100001111001001111101110110010110001010101110001001001000110110110101101010010110110110011010000110110000111100100111110111011001011000101010111000100100100011011011"

    logger.debug(RandomExcursions.random_excursions_test(data))