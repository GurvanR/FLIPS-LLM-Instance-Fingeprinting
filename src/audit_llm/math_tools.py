import numpy as np

import random
import math
from itertools import combinations


def _generate_balanced_partition(total, num_parts):
    """
    Generate the most balanced partition of 'total' into 'num_parts' positive integers.
    Makes counts as equal as possible.
    
    Args:
        total: Total sum to partition
        num_parts: Number of parts in the partition
        
    Returns:
        Tuple representing the balanced partition (sorted descending)
        
    Example:
        _generate_balanced_partition(6, 3) -> (2, 2, 2)
        _generate_balanced_partition(5, 3) -> (2, 2, 1)
        _generate_balanced_partition(7, 3) -> (3, 2, 2)
    """
    base_count = total // num_parts
    remainder = total % num_parts
    
    # Create partition: 'remainder' parts get (base_count + 1), rest get base_count
    partition = [base_count + 1] * remainder + [base_count] * (num_parts - remainder)
    
    return tuple(partition)


def random_combinations(iterable, combination_size, num_samples, 
                       unique_elements=None, threshold=10**4, seed=70):
    """
    Generate num_samples unique random combinations of given size from iterable.
    
    Args:
        iterable: Collection to draw elements from
        combination_size: Size of each combination (number of elements per combination)
        num_samples: Number of random combinations to generate
        unique_elements: If set, max number of distinct elements per combination.
                        Elements can repeat to fill combination_size, with counts
                        as balanced as possible (e.g., [2,2,2] not [4,1,1]).
                        If None, standard combinations without repetition.
        threshold: Max combinations to generate exhaustively (default 1M)
        seed: Random seed for reproducibility
        
    Returns:
        List of num_samples random combinations (as lists)
        
    Raises:
        ValueError: If num_samples exceeds total possible combinations or 
                   if unique_elements > combination_size
    """
    random.seed(seed)
    pool = tuple(iterable)
    pool_size = len(pool)
    
    # Validate unique_elements
    if unique_elements is not None:
        if unique_elements > combination_size:
            return []  # skip: can't have more unique elements than combination size
        if unique_elements < 1:
            raise ValueError(f"unique_elements must be at least 1")
    
    # Case 1: Standard combinations (no repetition)
    if unique_elements is None:
        total_combinations = math.comb(pool_size, combination_size)
        
        if num_samples > total_combinations:
            raise ValueError(
                f"Requested {num_samples} samples, but only {total_combinations} "
                f"unique combinations exist."
            )
        
        # Strategy 1: Small sample space - generate all, then sample
        if total_combinations <= threshold:
            all_combo_indices = list(combinations(range(pool_size), combination_size))
            sampled_indices = random.sample(all_combo_indices, num_samples)
            return [[pool[i] for i in combo] for combo in sampled_indices]
        
        # Strategy 2: Large sample space - rejection sampling
        seen = set()
        result = []
        
        while len(result) < num_samples:
            combo_indices = tuple(sorted(random.sample(range(pool_size), combination_size)))
            
            if combo_indices not in seen:
                seen.add(combo_indices)
                result.append([pool[i] for i in combo_indices])
        
        return result
    
    # Case 2: Combinations with repetition (limited unique elements, balanced counts)
    else:
        # Generate the single balanced partition
        partition = _generate_balanced_partition(combination_size, unique_elements)
        
        # Total combinations = ways to choose unique_elements items from pool
        total_combinations = math.comb(pool_size, unique_elements)
        
        if num_samples > total_combinations:
            raise ValueError(
                f"Requested {num_samples} samples, but only {total_combinations} "
                f"unique combinations exist (with {unique_elements} unique elements)."
            )
        
        # Strategy 1: Small sample space
        if total_combinations <= threshold:
            all_combos = []
            
            # For each way to choose unique_elements items from pool
            for item_indices in combinations(range(pool_size), unique_elements):
                items = [pool[i] for i in item_indices]
                
                # Build combination with contiguous repeated elements (balanced counts)
                combo = []
                for item, count in zip(items, partition):
                    combo.extend([item] * count)
                all_combos.append(tuple(combo))
            
            sampled = random.sample(all_combos, num_samples)
            return [list(combo) for combo in sampled]
        
        # Strategy 2: Large sample space - rejection sampling
        seen = set()
        result = []
        
        while len(result) < num_samples:
            # Randomly choose unique_elements items
            item_indices = tuple(sorted(random.sample(range(pool_size), unique_elements)))
            
            if item_indices not in seen:
                seen.add(item_indices)
                items = [pool[i] for i in item_indices]
                
                # Build combination with balanced counts
                combo = []
                for item, count in zip(items, partition):
                    combo.extend([item] * count)
                
                result.append(combo)
        
        return result

def log_entire_part(n:int, base=2, superior = True):
    x = np.log(n) / np.log(base)
    if superior:
        return np.ceil(x)
    else:
        return np.floor(x)

def min_max_normalize(array, new_min=0, new_max=1):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val) 
    return normalized_array


def interpolation(x, Va, Vb, a, b):
    alpha = (Vb-Va)/(b-a)
    beta = (b*Va - a*Vb)/(b-a)
    return alpha*x + beta

def Finite_Law_of_X(k, n):
    return 1 - (2**(n-k) - 1)/(2**n - 1)


def nan_to_zero(M:np.ndarray):
    return np.nan_to_num(M, nan=0.0)