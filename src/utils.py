import re
from hashlib import md5


# def hash_family_gen(seed: int):
#     """Create a lambda hashing function given a seed.

#     Input: seed (int)
#     Returns: `lambda x: str(seed) + str(x) --> MD5 hex --> int`
#     """
#     # MD5   : 32-digit hexadecimal, 128-bit integer
#     # SHA256: 64-digit hexadecimal, 256-bit integer

#     return lambda x: int(md5((str(seed) + str(x)).encode("utf-8")).hexdigest(), 16)

import xxhash

def hash_family_gen(seed: int, algo="xxh128"):
    """Create a lambda hashing function given a seed.

    Input: seed (int), algo: `['xxh128', 'md5']`
    Returns: `lambda x: str(seed) + str(x) --> hash fn --> x-bit int`
    """
    if algo == "xxh128":
        return lambda x: xxhash.xxh128(f"{seed}{x}".encode("utf-8"), seed=1).intdigest()
    else:
        return lambda x: int(md5(f"{seed}{x}".encode("utf-8")).hexdigest(), 16)
    
def jaccard(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    """Compute element-wise MinHash jaccard approximation

    Input: a, b (tuple[str]): lists to compare. Truncates to shortest input
    Returns: ratio of equal entries to length of the (shortest) list.
    """
    # For MinHash, element-wise eq count is enough
    # For actual jaccard, need intersection and count
    min_length = min(len(a), len(b))
    equal_entries = sum(1 for a_i, b_i in zip(a, b) if a_i == b_i)
    return equal_entries / min_length


def tokenize(text: str) -> list[str]:
    """Remove punctuation, extra spaces, lower case, trim trailing space
    and split by space

    Input: text (str)
    Returns: list of tokens
    """
    # Combine regular expressions for better performance
    text = re.sub(r"[^\w\s]", " ", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    return text.lower().strip().split()

# def hash_to_int(obj):
#     return int(md5(obj.__repr__().encode("utf-8")).hexdigest(), 16)

def hash_to_int(obj):
    # return xxhash.xxh128(obj.__repr__().encode("utf-8")).intdigest()
    return xxhash.xxh32(obj.__repr__().encode("utf-8"), seed=1).intdigest()