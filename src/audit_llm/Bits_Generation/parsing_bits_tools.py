import logging
import re
from typing import Callable, Dict, List, Tuple, Union

logger = logging.getLogger(__name__)

import numpy as np


def bits_scrapper_direct(string: str) -> List[int]:
    """
    Returns as a list, every 0 and 1 from a string, within the same order as their apparition.
    If none, returns an empty list.
    """
    # Collect only '0' and '1' from the string
    return [int(char) for char in string if char in ("0", "1")]


def bits_scrapper_direct_string(string: str, items: list[str] = ["0", "1"]) -> str:
    """Return the full answer string unchanged.

    Previously filtered to keep only items and whitespace; now kept as-is
    because the full text is more appropriate for passive token counting.
    """
    return string



def bit_string_to_ndarray(bit_string: str) -> np.ndarray:
    """
    Converts a string of 0s and 1s into a numpy ndarray of integers.
    Returns an empty ndarray if the input is empty.
    """
    if bit_string == "[]":
        return np.array([], dtype=np.int8)
    else:
        return np.array([int(bit) for bit in bit_string], dtype=np.int8)


def answer_to_bit_string(answer: str, items: list[str]) -> str:
    """
    Converts a string containing items into a bit string of indices.
    Matches are case-sensitive and prioritized by length (longest match first).
    def run_tests():
        # --- Provided Requirement Test ---
        # Orange (index 1) and O (index 0)
        # O r a n g e | O r a n g e | O | O r a n g e | O r a n g e | O | O | O | O r a n g e
        # 1           | 1           | 0 | 1           | 1           | 0 | 0 | 0 | 1
        assert answer_to_bit_string("OrangeOrangeOOrangeOrangeOOOOrange", ['O', 'Orange']) == "110110001"
        assert answer_to_bit_string("OrangeOrangeOOrangeOrangeOOOOrange", ['Orange', 'O']) == "001001110"


        # --- Case Sensitivity Tests ---
        assert answer_to_bit_string("Hat hat HAT", ['Hat', 'hat']) == "01"
        assert answer_to_bit_string("CASE case", ['CASE']) == "0"

        # --- Overlap & Priority Tests ---
        assert answer_to_bit_string("thatt", ['that', 't']) == "01"
        assert answer_to_bit_string("banana", ['ana', 'ban']) == "10" # 'ban' then 'ana'

        # --- Original Tests (Updated for Case Sensitivity) ---
        assert answer_to_bit_string("I picked 01", ['0','1']) == "01"
        assert answer_to_bit_string("23145020", ['0','1']) == "100"
        assert answer_to_bit_string("I took a car", ['hat', 'boat']) == ""
        assert answer_to_bit_string("Hat hat boat", ['hat', 'boat']) == "01" # First 'Hat' ignored, 'hat' 'boat' matched

        # --- Edge Cases ---
        assert answer_to_bit_string("", ['A', 'B']) == ""
        assert answer_to_bit_string("ABC", []) == ""
        assert answer_to_bit_string("aaaaa", ['aa']) == "00" # Matches two sets of 'aa', leaves one 'a'

        # --- Additional Tests ---
        assert answer_to_bit_string("ABBAAABBABAABBABABABAABABAB", ['ABA', 'BAB']) == "11001"

        logger.info("All tests passed!")

    if __name__ == "__main__":
        run_tests()
    """

    if not answer or not items:
        return ""

    # Create a mapping of item to its index in the original items list
    # Using a dict handles duplicates by keeping the last index,
    # but we iterate items in the matching loop to ensure correct index retrieval.
    item_to_index = {item: str(i) for i, item in enumerate(items)}

    # Sort items by length descending to ensure "Orange" is matched before "O"
    sorted_items = sorted(items, key=len, reverse=True)

    final_answer = ""
    i = 0
    while i < len(answer):
        matched = False
        for item in sorted_items:
            # Check if the substring at current index matches the item exactly
            if answer[i : i + len(item)] == item:
                final_answer += item_to_index[item]
                i += len(item)
                matched = True
                break

        if not matched:
            i += 1

    return final_answer


def token_pair_name_to_items(dataset_name: str):
    """
    dataset_name is like item1-item2_ip2_sp0_500_fs1_None_700
    """
    items = dataset_name.split("_")[0].split("-")
    return items


"""def bits_token_pair_to_scrapper(dataset_name: str, min_seq_length:int):
    items= token_pair_name_to_items(dataset_name)
    
    def answer_to_bit_string_threshold(string):
        scrapped=answer_to_bit_string(string, items=items)
        if len(scrapped) >= min_seq_length:
            return scrapped
        else:
            return ''
    
    return answer_to_bit_string_threshold """


def bits_token_pair_to_scrapper(min_seq_length: int) -> Callable[[str, str], str]:

    def answer_to_bit_string_threshold(string: str, token_pair: str):
        scrapped = answer_to_bit_string(string, items=token_pair_string_to_list(token_pair))
        if len(scrapped) >= min_seq_length:
            return scrapped
        else:
            return ""

    return answer_to_bit_string_threshold


def compute_proper_bit_sequences(generations, items: List[str], min_seq_length: int, mode: str = "vllm"):
    """
    Computes number of proper answers.
    """
    if generations:
        if mode == "vllm":
            if isinstance(generations, Dict):
                items_to_loop = generations.items()
            else:
                items_to_loop = enumerate(generations)
            bits_format = [
                len(answer_to_bit_string(output.outputs[0].text, items=items)) >= min_seq_length
                for idx, output in items_to_loop
            ]
        elif mode == "hf":
            bits_format = [len(answer_to_bit_string(answer, items=items)) >= min_seq_length for answer in generations]
        elif mode == "openrouter":
            bits_format = [
                len(answer_to_bit_string(answer.choices[0].message.content, items=items)) >= min_seq_length
                for answer in generations
            ]
        return sum(bits_format)
    else:
        return 0


def validate_seq(generation, vllm_output, mode: str, min_seq_length: int):
    """
    generation: LLM_Generation
    """
    if mode == "vllm":
        if generation.token_pair == "no_token_pairs":
            return True
        else:
            return len(answer_to_bit_string(vllm_output.outputs[0].text, items=token_pair_string_to_list(generation.token_pair))) >= min_seq_length  # type: ignore
    else:
        raise NotImplementedError("Only vllm mode is implemented in validate_seqs.")
        """elif mode == 'hf':
            bits_format = [len(answer_to_bit_string(answer, items=items)) >= min_seq_length for answer in generations]
        elif mode == 'openrouter':
                bits_format = [len(answer_to_bit_string(answer.choices[0].message.content, items=items)) >= min_seq_length for answer in generations]
        return sum(bits_format)"""


COMPLEMENTARY_SUBSTRINGS = [" ", ",", "\n"]


def token_pair_to_string(token_pair: List[str]):
    return "-".join(token_pair)


def token_pair_string_to_list(token_pair: str):
    return token_pair.split("-")
