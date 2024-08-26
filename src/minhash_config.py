class MinHashCFG:
    NUM_SHINGLES = 2
    """Number of tokens per shingle"""
    NUM_HASH = 100
    """Number of hash functions to be generated and used for minhashing.\n
    Higher = more accurate
    [Wikipedia - MinHash](https://en.wikipedia.org/wiki/MinHash)
    """
    NUM_BANDS = 25
    """Number of bands to split the MinHashes signature"""
    NUM_BUCKETS = 500
    """Number of hash buckets to put bands into.\n
    Higher = more documents to be filtered in the LSH step\n
    [`num_buckets > num_doc`](https://stackoverflow.com/a/57790355)\n
    [`num_buckets = sqrt(num_doc)`](https://stackoverflow.com/a/37274097)\n
    `500` seems to be a good number. `5000` filters ~50%
    """

    NUM_ROWS = NUM_HASH // NUM_BANDS
    """Length of the band."""

    COMMON_THRES = 4
    """Occurrence threshold for common shingle, used to avoid storing uncommon/rare
    shingles in the hash_dict (appears only once or twice).
    `4` seems to be a good number"""

    DO_SORT_SHING_DICT = True
    """Sort `shing_dict` ascendingly to ensure consistency"""

    COL_ID = "id"
    """Index column"""
    COL_TEXT = "text"
    """Document's text column"""
    COL_SIG = "signature"
    """MinHash signature column"""
    COL_BUCKET = "bucket_id"
    """Bucket ID column"""
    COL_BUCKETS = "bucket_ids"
    """List of bucket ids column"""
    COL_SHINGLES = "shingles"
    """List of shingles column"""
    COL_BOOL_VEC = "bool_vec"
    """Boolean vector true indices list column"""
    COL_JACCARD = "jaccard"
    """MinHash Jaccard similarity Approximation column"""

    COL_SHINGLE = "shingle"
    """A shingle column"""
    COL_SHINGIDX = "shing_idx"
    """A shingle's index column"""
    COL_COUNT = "count"
    """A shingle's occurrence column"""