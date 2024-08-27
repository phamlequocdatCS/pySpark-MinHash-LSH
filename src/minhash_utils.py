import numpy as np

from .utils import hash_family_gen, hash_to_int


def get_k_shingles(token_lst: list[str], k: int) -> list[tuple[str, ...]]:
    """Create k-shingles from tokenized text

    ### Inputs:
        token_lst (list[str]): output of tokenize()
        k (int): length of a shingle
    ### Returns: list of distinct shingles
    """
    len_tk_lst = len(token_lst)
    # List slicing
    ## Tuple the shingle because set needs immutable objects
    shingles = [tuple(token_lst[i : i + k]) for i in range(len_tk_lst - k + 1)]
    # Get distinct
    shingles = list(set(shingles))
    return shingles


def bool_vectorizer(
    shing_lst: list[tuple[str, ...]],
    shing_dict: dict[tuple[str, ...], int],
) -> list[int]:
    """Get a sparse representation of the boolean vector of a shingled document

    ### Inputs:
        shing_lst (list[tuple[str, ...]]): output of `get_k_shingles`
        shing_dict (dict[tuple[str,...], int]): `<shingle>-<idx>` pairs
    ### Returns: list of true indices
    """
    true_indices = [
        shing_dict[shingle] for shingle in shing_lst if shingle in shing_dict
    ]
    true_indices.sort()
    return true_indices


def hash_a_doc(
    true_indices: list[int],
    num_hash: int,
    hash_dict: dict[int, np.ndarray],
    is_64_bit_hash = False,
) -> tuple[str, ...]:
    """Compute the MinHash signature of a document

    ### Inputs:
        true_indices (list[int]): output of `bool_vectorizer`
        num_hash (int): number of hash functions to simulate permutations
        (dimensionality of the MinHash signature)
        hash_dict (dict[int, numpy.ndarray]): Pre-computed `<idx>-<Hash^num_hash>`
        to avoid repeated calculation of expensive hashes.
    ### Returns:
        MinHash signature (tuple of minimal hashes, in hexadecimal string format)
    """
    # Initialize the hash vector with max int128
    if is_64_bit_hash:
        dtype = np.uint64
        max_int: int | float = np.iinfo(np.uint64).max
    else:
        dtype = None
        max_int = np.inf

    min_hash_sig = np.full(num_hash, max_int, dtype=dtype)

    # Create the family of hash fns
    # (pyspark cannot broadcast lambdas, so we need to create them in scope)
    hash_family = [hash_family_gen(i) for i in range(num_hash)]

    for idx in true_indices:
        if idx in hash_dict:  # If we have precomputed the hash for this shingle
            hashed_vec = hash_dict[idx]
        else:  # Calculate as normal
            hashed_vec = np.array(
                [hash_fn(idx) for hash_fn in hash_family], dtype=dtype
            )
        min_hash_sig = np.minimum(min_hash_sig, hashed_vec)  # Element-wise min

    str_mh_sig = [hex(sig) for sig in min_hash_sig.tolist()]
    # Convert to hex string because
    #   pyspark df can't store integers with arbitrary length. (max 64-bit)
    #   MD5 is 128-bit
    # May be possible using DecimalType()? up to 38 digits
    # Memory may be a concern, easier to store hex strings than big integers?
    return tuple(str_mh_sig)


def buckenize(
    str_mh_sig: tuple[str, ...], num_rows: int, num_bands: int, num_buckets: int
) -> list[int]:
    """Hash MinHash signature into buckets

    ### Inputs:
        str_mh_sig (tuple[str]): output of `hash_a_doc`
        num_rows (int): Number of rows per band
        num_bands (int): Number of bands to split the signature
        num_buckets (int): Number of buckets to hash the bands into
    ### Returns:
        list of `<bucket_id>`
    """
    # Convert hex strings back to integers
    #   Slight jaccard approx difference, may be worse if not convert back to int?
    int_mh_sig = tuple([int(sig, 16) for sig in str_mh_sig])
    # Slice signature into bands
    sig_bands = [
        int_mh_sig[i * num_rows : (i + 1) * num_rows] for i in range(num_bands)
    ]
    # Hash bands into bucket
    #   Using python's hash gives bad results, potentionally because of collision rate
    band_buckets = [hash_to_int(band) % num_buckets for band in sig_bands]
    return list(set(band_buckets))


def bucket_filter(buckets, query_buckets, bucket_thres):
    common_buckets = [bucket for bucket in buckets if bucket in query_buckets]
    total_buckets = len(set(buckets + list(query_buckets)))
    return len(common_buckets) / total_buckets >= bucket_thres
