import logging
import os
import time

import numpy as np
import pandas as pd
import tqdm

from .minhash_config import MinHashCFG
from .minhash_utils import bool_vectorizer, buckenize, get_k_shingles, hash_a_doc
from .utils import hash_family_gen, jaccard, tokenize

try:
    from pandarallel import pandarallel

    pandarallel.initialize(progress_bar=True, nb_workers=4)
    use_parallel = True
except ImportError as e:
    print(f"Revert to single-threaded. {e}")
    use_parallel = False

tqdm.tqdm.pandas()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class InMemoryMinHashLSH(MinHashCFG):
    """Class for performing MinHash LSH using Pandas DataFrame. See `MinHashCFG` for config"""

    def __init__(
        self, documents: pd.DataFrame, do_cache_hash=True, hash_fn="xxh128"
    ) -> None:
        self.documents = documents
        """DataFrame storing documents, one per row.
        Index unique and increases, but not consecutive (when partitioned).
        Is consecutive when `partition=1` (use 1 core or coalesced)
        Columns: `<text>-<id>`"""

        self.shing_dict: dict[tuple[str, ...], int] = dict()
        """Dictionary storing `<shingle>--><idx>` pairs for fast index lookup"""

        self.shing_len = -1
        """Total number of shingles in dictionary."""

        self.max_len = -1
        """Length of the longest shingled document"""

        self.hash_family = [hash_family_gen(i, hash_fn) for i in range(self.NUM_HASH)]
        """Family of hash functions to simulate permutation"""

        self.hash_dict: dict[int, np.ndarray] = dict()
        """Storing `<true_idx>-->[hash_family]` results. `{<shing_idx>: <mh_arr>}`
        Only common shingles (appears more than common_thres) should be stored.\n
        Performance boost in minhashing step when `shing_len, num_hash` high.
        """

        self.lsh_df: pd.DataFrame | None = None
        self.do_cache_hash = do_cache_hash

        self.is_64_bit_hash = hash_fn not in self.HASH_128BIT
        self.dtype = np.uint64 if self.is_64_bit_hash else None

    def shingling(self, documents: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """Perform shingling step

        Input: documents (pd.DataFrame) `<text>`

        ### Returns:
            bool_vec DF `<true_indices>`
            elapsed_time (float): Time in seconds
        """
        logger.info("Starting shingling process")
        start_time = time.time()

        # Step 1: Shingling each document
        logger.info("Tokenizing documents")

        shingles_col: pd.Series = documents[self.COL_TEXT].progress_apply(
            lambda x: get_k_shingles(tokenize(x), k=self.NUM_SHINGLES)
        )  ## <shingles>

        # Step 2: Build shing_dict
        self.shing_dict, self.shing_len, shing_freq = self._build_shing_dict(
            shingles_col
        )
        logger.info(f"Total unique shingles: {self.shing_len}")

        if self.do_cache_hash:
            # Step 3: Pre-compute hash_dict (for minhashing step)
            self.hash_dict = self._build_hash_dict(self.shing_dict, shing_freq)

        # Step 4: Get the boolean vectors
        logger.info("Creating boolean vectors")

        ## <true_indices>
        bool_vec: pd.Series = shingles_col.progress_apply(
            bool_vectorizer, shing_dict=self.shing_dict
        )

        bool_vec_df = bool_vec.to_frame(name=self.COL_BOOL_VEC)

        elapsed_time = time.time() - start_time
        logger.info(f"Shingling process completed in {elapsed_time:.2f} seconds")

        return bool_vec_df, elapsed_time

    def minhashing(self, bool_vec: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """Perform minhashing step
        Input: bool_vec (pd.DataFrame) `<true_indices>` (output of `shingling()`)

        ### Returns:
            minhash_df (pd.DataFrame): All documents' MinHash signatures `<signature>`
            elapsed_time (float): Time taken in seconds
        """
        logger.info("MinHashing")
        start_time = time.time()
        if use_parallel:
            minhash_series: pd.Series = bool_vec[self.COL_BOOL_VEC].parallel_apply(
                hash_a_doc,
                num_hash=self.NUM_HASH,
                hash_dict=self.hash_dict,
                is_64_bit_hash=self.is_64_bit_hash,
            )
        else:
            minhash_series = bool_vec[self.COL_BOOL_VEC].progress_apply(
                hash_a_doc,
                num_hash=self.NUM_HASH,
                hash_dict=self.hash_dict,
                is_64_bit_hash=self.is_64_bit_hash,
            )

        minhash_df = minhash_series.to_frame(self.COL_SIG)
        elapsed_time = time.time() - start_time
        return minhash_df, elapsed_time

    def locality_sensity_hashing(self, str_mh_sig: tuple[str]) -> pd.DataFrame:
        """Perform LSH bucketing step on a document's MinHash signature
        Returns: `str_mh_sig-<bucket>`, one row for each bucket it hashes to
        """
        buckets = buckenize(str_mh_sig, self.NUM_ROWS, self.NUM_BANDS, self.NUM_BUCKETS)
        sig_df = pd.DataFrame(
            [[str_mh_sig, buckets]], columns=[self.COL_SIG, self.COL_BUCKETS]
        )
        return sig_df

    def run(self) -> None:
        """Run the MinHash LSH algorithm on the init document DF"""

        logger.info("Starting MinHash LSH algorithm")
        total_start = time.time()
        # Step one: Get bool vecs from shingling
        bool_vec, boolvec_elapsed = self.shingling(self.documents)

        # Step two: Get MinHash signatures for the entire corpus
        minhash_df, minhash_elapsed = self.minhashing(bool_vec)

        # Step three: LSH Bucketing on each row
        lsh_elapsed = self.run_lsh(minhash_df)
        assert self.lsh_df is not None, "Something went wrong with run_lsh"

        logger.info(f"LSH Success. Final length: {len(self.lsh_df)}.")
        logger.info(f"Shingling took {boolvec_elapsed:.5f} s")
        logger.info(f"Minhash   took {minhash_elapsed:.5f} s")
        logger.info(f"LSH       took {lsh_elapsed:.5f} s")
        logger.info(f"Total time: {time.time() - total_start:.5f} ")

    def run_lsh(self, minhash_df):
        logger.info("LSH Bucketing")
        start = time.time()
        lsh_df = minhash_df[self.COL_SIG].progress_apply(self.locality_sensity_hashing)
        # Add MinHash signature to each document
        self.documents[self.COL_SIG] = minhash_df[self.COL_SIG]
        # Concatenate rows' LSH buckets together into one DF
        self.lsh_df = pd.concat(lsh_df.values)
        self.lsh_df.reset_index(drop=True, inplace=True)

        lsh_elapsed = time.time() - start
        return lsh_elapsed

    def process_query(self, query: str) -> tuple[tuple[str, ...], list[int]]:
        """Preprocess the query for LSH Nearest Neighbor. The steps are:
        1. Tokenize
        2. Shingling
        3. Bool vectorizing
        4. MinHashing
        5. Bucketing

        ### Returns:
            minhash_sig (tuple[str,...]): MinHash signature of query
            buckets (list[int]): Buckets query hashed to
        """
        token_lst = tokenize(query)
        shingles = get_k_shingles(token_lst, self.NUM_SHINGLES)
        bool_vec = bool_vectorizer(shingles, self.shing_dict)
        minhash_sig = hash_a_doc(bool_vec, self.NUM_HASH, self.hash_dict)
        buckets = buckenize(
            minhash_sig, self.NUM_ROWS, self.NUM_BANDS, self.NUM_BUCKETS
        )
        return minhash_sig, buckets

    def any_bucket_in_list(self, bucket_list, buckets_to_check):
        return any(bucket in bucket_list for bucket in buckets_to_check)

    def get_signatures_df(self, buckets: list[int]) -> pd.DataFrame:
        assert self.lsh_df is not None, "lsh_df is not initialized. Did you .run()?"
        # Get rows/signatures that are in the query's buckets
        mask = self.lsh_df[self.COL_BUCKETS].apply(
            lambda x: self.any_bucket_in_list(x, buckets)
        )

        # Get rows that are in the query's buckets
        signatures = self.lsh_df.loc[mask, self.COL_SIG]

        # Get unique signatures
        # signatures = signatures.unique()
        signatures_df = pd.DataFrame(signatures, columns=[self.COL_SIG])

        return signatures_df

    def approxNearestNeighbors(self, key: str, n: int) -> pd.DataFrame:
        """Perform Approximated Nearest Neighbors search

        ### Inputs:
            key (str): The document to search
            n (int): Number of documents to return
        ### Returns:
            Result DF `<jaccard>-<doc_id>-<text>`
        """
        minhash_sig, buckets = self.process_query(key)

        signatures_df = self.get_signatures_df(buckets)
        logger.info(f"Length of potential documents: {len(signatures_df)}")

        # Calculate jaccard
        signatures_df[self.COL_JACCARD] = signatures_df[self.COL_SIG].apply(
            lambda x: jaccard(minhash_sig, x)
        )
        # Sort descendingly
        signatures_df = signatures_df.sort_values(
            by=[self.COL_JACCARD], ascending=False
        )
        # Take top n
        signatures_df = signatures_df[:n]
        signatures_df.reset_index(drop=True, inplace=True)
        # Merge to get matching documents' text
        result_df = self.get_result_df(signatures_df)
        return result_df

    def get_result_df(self, signatures_df):
        result_df = pd.merge(
            signatures_df,
            self.documents,
            left_on=self.COL_SIG,
            right_on=self.COL_SIG,
            how="left",
        )
        # Clean up the result
        result_df.drop(columns=[self.COL_SIG], inplace=True)
        ## <jaccard>-<doc_id>-<text>
        result_df.rename(columns={"index": self.COL_ID}, inplace=True)
        return result_df

    def _build_shing_dict(
        self, shingles_col: pd.Series
    ) -> tuple[dict[tuple[str, ...], int], int, pd.Series]:
        """Build shing_dict `<shingle>--><idx>` dictionary

        Input: shingles_col (pd.Series): Series of documents' shingles

        Returns:
            shing_dict
            shing_len: length of `shing_dict`
            shing_freq: series of shingles' occurrence
        """
        # Get series of shingles
        shing_col = shingles_col.explode()
        # Get list of unique shingles
        shing_distinct: list[tuple[str, ...]] = shing_col.unique().tolist()
        # max_len = max(shingles_col.apply(len))
        #   Will throw an error if num_shingles > max_len
        #   For now we just ignore

        if self.DO_SORT_SHING_DICT:
            shing_distinct = sorted(shing_distinct)

        # Assign an index to each shingle
        shing_dict = {shing: i for i, shing in enumerate(shing_distinct)}
        shing_len = len(shing_dict)

        # Get shingles' occurrence
        shing_freq = shing_col.value_counts()

        return shing_dict, shing_len, shing_freq

    def _build_hash_dict(
        self, shing_dict: dict[tuple[str, ...], int], shing_freq: pd.Series
    ) -> dict[int, np.ndarray]:
        """Pre-compute Hashes for MinHash step. Expect outputs from `_build_shing_dict()`

        ### Inputs:
            shing_dict (dict[tuple[str, ...], int]): `<shingle>--><idx>` pairs
            shing_freq (pd.Series): `<shingle>-<count>`
        ### Returns:
            hash_dict `{<shing_idx>: <mh_arr>}`
        """
        logger.info("Pre-computing hash dictionary")
        start_time = time.time()
        # Filter uncommon shingles
        shing_freq = shing_freq[shing_freq >= self.COMMON_THRES]
        # Get their (common) index
        common_shing = [shing_dict[shing] for shing in shing_freq.index]

        # Precompute MinHashes
        #   Could be done inside the MinHashing step, but do here at once for consistency
        hash_dict = dict()  ## {<shing_idx>: <mh_arr>}
        for shing_idx in tqdm.tqdm(common_shing, desc="Precomputing minhashes"):
            hash_dict[shing_idx] = np.array(
                [hash_fn(shing_idx) for hash_fn in self.hash_family],
                dtype=self.dtype,
            )

        elapsed_time = time.time() - start_time
        logger.info(f"MinHashing process completed in {elapsed_time:.2f} seconds")

        return hash_dict

    @classmethod
    def read_from_txt(
        cls, filepath: str, do_cache_hash=True, do_parallel = True, hash_fn="xxh128", trim: int = 0
    ) -> "InMemoryMinHashLSH":
        """Open text file as a Pandas DataFrame. Each line is a row.

        Optional Input: trim (int): Number of documents to take. By default loading all."""
        if not os.path.isfile(filepath):
            raise ValueError(f"Could not find file at: {filepath}")

        start_time = time.time()
        # https://stackoverflow.com/a/76560355
        df = pd.read_csv(
            filepath,
            delimiter="I_am_a_delimiter",
            engine="python",
            header=None,
            skip_blank_lines=True,
        )
        if trim > 0:
            df = df[:trim]
        ## <text>
        df = df.rename(columns={0: cls.COL_TEXT})
        df = df.reset_index()
        elapsed_time = time.time() - start_time
        logger.info(
            f"Load success. Received [{len(df)}] {filepath}. Took {elapsed_time} s"
        )
        global use_parallel
        if not do_parallel and use_parallel:
            use_parallel = False
            print(f"{use_parallel=}")
        return InMemoryMinHashLSH(df, do_cache_hash, hash_fn)
