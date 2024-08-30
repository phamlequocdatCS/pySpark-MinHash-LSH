import logging
import os
import time
from typing import Any

import numpy as np
import pyspark.sql as sql
from pyspark.broadcast import Broadcast
from pyspark.context import SparkContext
from pyspark.sql.functions import (
    col,
    explode,
    monotonically_increasing_id,
    udf,
)
from pyspark.sql.session import SparkSession
from pyspark.sql.types import ArrayType, BooleanType, FloatType, IntegerType, StringType

from .minhash_config import MinHashCFG
from .minhash_pyspark_utils import is_memory_safe, py_is_memory_safe
from .minhash_utils import (
    bool_vectorizer,
    buckenize,
    bucket_filter,
    get_k_shingles,
    hash_a_doc,
)
from .utils import hash_family_gen, jaccard, tokenize

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class PySparkMinHashLSH(MinHashCFG):
    """Class for performing MinHash LSH using PySpark DataFrame"""

    def __init__(
        self, documents: sql.DataFrame, sc: SparkContext, sqlContext: SparkSession
    ) -> None:
        self.sc = sc
        """The spark context"""
        self.sqlContext = sqlContext
        """The spark sql session"""

        self.documents = documents
        """DataFrame storing documents, one per row.
        Index unique and increases, but not consecutive (when partitioned).
        Is consecutive when `partition=1` (use 1 core or coalesced)
        Columns: `<text>-<id>`"""
        self.shing_dict: dict[tuple[str, ...], int] = dict()
        """Dictionary storing `<shingle>--><idx>` pairs for fast index lookup"""
        self.shing_len = -1
        """Total number of shingles in dictionary."""

        self.hash_family = [hash_family_gen(i) for i in range(self.NUM_HASH)]
        """Family of hash functions to simulate permutation"""
        self.hash_dict: dict[int, np.ndarray] = dict()
        """Storing `<true_idx>-->[hash_family]` results. `{<shing_idx>: <mh_arr>}`
        Only common shingles (appears more than common_thres) should be stored.\n
        Performance boost in minhashing step when `shing_len, num_hash` high.
        """

        sc_num_shingles = self.broadcast(self.NUM_SHINGLES)
        self.fn_k_shingler = udf(
            lambda x: get_k_shingles(tokenize(x), sc_num_shingles.value),
            ArrayType(ArrayType(StringType())),
        )
        """UDF to perform shingling on DF"""

        self.fn_bool_vectorizer: Any | None = None
        """UDF to perform boolean vectorize DF (shingling step)"""
        self.fn_hash_a_doc: Any | None = None
        """UDF to perform minhashing on DF"""
        self.fn_lsh: Any | None = None
        """UDF to perform LSH Bucketing"""

        self.minhash_df: sql.DataFrame | None = None
        """DF Storing `<id>-<signature>`"""
        self.lsh_df: sql.DataFrame | None = None
        """DF Storing `<id>-<buckets>`"""

    def broadcast(self, obj) -> Broadcast:
        return self.sc.broadcast(obj)

    def shingling(self, documents: sql.DataFrame) -> sql.DataFrame:
        """Perform shingling step

        Input: documents (sql.DataFrame) `<text>-<id>`
        Returns: bool_vec DF `<id>-<true_indices>`
        """

        # Step 1: Shingling each document
        ## <text>-<id>-<shingles>
        shing_df = documents.withColumn(
            self.COL_SHINGLES, self.fn_k_shingler(col(self.COL_TEXT))
        )

        ## <id>-<shingles>
        shing_df = shing_df.select(self.COL_ID, self.COL_SHINGLES)

        # Step 2: Build shing_dict
        self.shing_dict, self.shing_len, collect_item_counts = self._build_shing_dict(
            shing_df
        )

        # Step 3: Pre-compute hash_dict (for minhashing step)
        self.hash_dict = self._build_hash_dict(self.shing_dict, collect_item_counts)

        # Step 4: Get the boolean vectors
        #   Because of pyspark DF length limitations
        #   When shing_len very large -> dim(bool_vec) very large -> error
        #   We store list of true indices instead
        #   Sparse vector also only store true indices
        #   But still error with length too long
        ## <id>-<true_indices>
        bool_vec = self._build_bool_vec(shing_df)

        return bool_vec

    def minhashing(self, bool_vec: sql.DataFrame) -> sql.DataFrame:
        """Perform minhashing step

        Input: bool_vec (sql.DataFrame) `<id>-<true_indices>` (output of `shingling()`)
        Returns: minhash_df `<id>-<signature>`
        """
        bc_num_hash = self.broadcast(self.NUM_HASH)
        bc_hash_dict = self.broadcast(self.hash_dict)

        self.fn_hash_a_doc = udf(
            lambda x: hash_a_doc(x, bc_num_hash.value, bc_hash_dict.value),
            ArrayType(StringType()),
        )

        ## <id>-<sparse_bool_vec>-<signature>
        minhash_df = bool_vec.withColumn(
            self.COL_SIG, self.fn_hash_a_doc(col(self.COL_BOOL_VEC))
        )

        ## <id>-<signature>
        minhash_df = minhash_df.select(self.COL_ID, self.COL_SIG)

        return minhash_df

    def locality_sensity_hashing(self, minhash_df: sql.DataFrame) -> sql.DataFrame:
        """Perform LSH buckeing step

        Input: minhash_df `<id>-<signature>` (output of `minhashing()`)
        Returns: lsh_df `<id>-<buckets>`
        """
        ## minhash_df: <id>-<signature>
        bc_num_rows = self.broadcast(self.NUM_ROWS)
        bc_num_bands = self.broadcast(self.NUM_BANDS)
        bc_num_buckets = self.broadcast(self.NUM_BUCKETS)

        self.fn_lsh = udf(
            lambda x: buckenize(
                x, bc_num_rows.value, bc_num_bands.value, bc_num_buckets.value
            ),
            ArrayType(IntegerType()),
        )

        ## <id>-<signature>-<buckets>
        lsh_df = minhash_df.withColumn(self.COL_BUCKETS, self.fn_lsh(col(self.COL_SIG)))

        ## <id>-<buckets>
        lsh_df = lsh_df.select(self.COL_ID, self.COL_BUCKETS)

        return lsh_df

    def run(self):
        """Run the MinHash LSH algorithm on the init document DF"""
        ## bool_vec: <id>-<sparse_bool_vec>
        bool_vec = self.shingling(self.documents)
        ## minhash_df: <id>-<signature>
        self.minhash_df = self.minhashing(bool_vec)
        ## lsh_df: <id>-<buckets>
        self.lsh_df = self.locality_sensity_hashing(self.minhash_df)

        if self.DO_CACHE:
            self.cache_dfs()

        logger.info("LSH Actions Completed.")

    def process_query(self, query: str) -> tuple[list[str], list[int]]:
        """Perform all steps needed to do LSH Nearest Neighbor

        ### Returns:
            minhash_sig (list[str]): MinHash signature of query
            buckets (list[int]): Buckets query hashed to
        """
        token_lst = tokenize(query)
        shingles = get_k_shingles(token_lst, self.NUM_SHINGLES)
        bool_vec = bool_vectorizer(shingles, self.shing_dict)
        minhash_sig = hash_a_doc(bool_vec, self.NUM_HASH, self.hash_dict)
        buckets = buckenize(
            minhash_sig, self.NUM_ROWS, self.NUM_BANDS, self.NUM_BUCKETS
        )
        return list(minhash_sig), buckets

    def approxNearestNeighbors(
        self, key: str, n: int, bucket_thres=0.0
    ) -> sql.DataFrame:
        """Perform Approximated Nearest Neighbors search

        ### Inputs:
            key (str): The document to search
            n (int): Number of documents to return
            bucket_thres (float): Threshold for portion of buckets shared between query and a document. Default to 0
                (ignore the threshold and pick documents sharing at least 1 bucket as candidate).
                    If bucket_thres is too high and cannot find matching documents, reverts to 0 behavior.
        ### Returns:
            Result DF `<id>-<text>-<jaccard>`
        """
        nn_start_time = time.time()

        # Step 1: Get query's MH signature and buckets
        minhash_sig, buckets = self.process_query(key)

        bc_mh_sig = self.broadcast(minhash_sig)

        bucket_filterer_one, bucket_filterer = self.get_bucket_filter(
            bucket_thres, buckets
        )

        # Step 2: Get candidate documents hashed into query's buckets
        ## <id>
        query_idxes = self.get_candidates(
            bucket_thres, bucket_filterer_one, bucket_filterer
        )

        # Step 3: Get candidate documents' MH signatures
        ## <id>-<signature>
        assert self.minhash_df is not None, "Couldn't find minhash_df, did you .run()?"
        result = self.minhash_df.join(query_idxes, on=[self.COL_ID], how="inner")

        ## <id>-<text>-<jaccard>
        result_df = self.get_result_df(n, result, bc_mh_sig)

        total_elapsed_time = time.time() - nn_start_time
        logger.info(f"Took {total_elapsed_time:.5f} s")
        return result_df

    def get_bucket_filter(self, bucket_thres: float, buckets: list[int]):
        bc_query_buckets = self.broadcast(buckets)
        bucket_filterer_one = udf(
            lambda x: any(bucket in bc_query_buckets.value for bucket in x),
            BooleanType(),
        )

        if bucket_thres > 0:
            bc_bucket_thres = self.broadcast(bucket_thres)

            bucket_filterer = udf(
                lambda x: bucket_filter(
                    x, bc_query_buckets.value, bc_bucket_thres.value
                ),
                BooleanType(),
            )
        else:
            bucket_filterer = bucket_filterer_one
        return bucket_filterer_one, bucket_filterer

    def get_result_df(self, n, result: sql.DataFrame, bc_mh_sig: Broadcast):
        jaccard_udf = udf(lambda x: jaccard(x, bc_mh_sig.value), FloatType())
        # Step 4: Compute Approx Jaccard Sim
        ## <id>-<signature>-<jaccard>
        result_jaccard = result.withColumn(
            self.COL_JACCARD, jaccard_udf(col(self.COL_SIG))
        )
        ## <id>-<jaccard>
        result_jaccard = result_jaccard.select(self.COL_ID, self.COL_JACCARD)

        # Step 5: Sort results descendingly on Approx Jaccard Sim
        result_jaccard = result_jaccard.orderBy(result_jaccard[self.COL_JACCARD].desc())

        # Step 6: Getting top n results
        logger.info(f"Collecting {n} results to driver")
        jaccard_collect_start_time = time.time()
        result_jaccard_collect = result_jaccard.head(
            n
        )  # This is done to preserve the sorted order

        logger.info(f"Collecting took {time.time() - jaccard_collect_start_time:.5} s")

        ## <id>-<jaccard>
        result_df: sql.DataFrame = self.sqlContext.createDataFrame(
            result_jaccard_collect, [self.COL_ID, self.COL_JACCARD]
        )

        # Step 7: Get matching documents' text
        ## <id>-<text>-<jaccard>
        result_df = self.documents.join(result_df, on=[self.COL_ID], how="inner")

        return result_df

    def get_candidates(self, bucket_thres, bucket_filterer_one, bucket_filterer):
        query_idxes, query_res_count = self._filter_by_query(bucket_filterer)

        if query_res_count == 0:
            logger.info(
                f"Found no result. Changing bucket_thres from {bucket_thres} to 0 (matching at least 1 bucket) "
            )
            query_idxes, query_res_count = self._filter_by_query(bucket_filterer_one)

        logger.info(f"Found {query_res_count} candicate documents")
        return query_idxes

    def _filter_by_query(self, bucket_filter_udf):
        ## <id>-<bucket>
        query_idxes = self.lsh_df.filter(
            bucket_filter_udf(self.lsh_df[self.COL_BUCKETS])
        )
        ## <id>
        query_idxes = query_idxes.select(self.COL_ID)
        query_res_count = query_idxes.count()

        return query_idxes, query_res_count

    def _build_bool_vec(self, shing_df: sql.DataFrame) -> sql.DataFrame:
        """Build bool_vec df `<id>-<true_indices>`

        Input: shing_df (sql.DataFrame) `<id>-<shingles>`
        """
        ## shing_df: <id>-<shingles>
        bc_shing_dict = self.broadcast(self.shing_dict)

        self.fn_bool_vectorizer = udf(
            lambda x: bool_vectorizer(x, bc_shing_dict.value), ArrayType(IntegerType())
        )

        ## <id>-<shingles>-<true_indices>
        bool_vec = shing_df.withColumn(
            self.COL_BOOL_VEC, self.fn_bool_vectorizer(col(self.COL_SHINGLES))
        )

        ## <id>-<true_indices>
        bool_vec = bool_vec.select(self.COL_ID, self.COL_BOOL_VEC)

        return bool_vec

    def _build_shing_dict(
        self, shing_df: sql.DataFrame
    ) -> tuple[dict[tuple[str, ...], int], int, list[sql.Row]]:
        """Build shing_dict `<shingle>--><idx>` dictionary

        Input: shing_df (sql.DataFrame): `<id>-<shingles>`
        Returns:
            shing_dict
            shing_len: length of shing_dict
            collect_item_counts: list of `<shingle>-<count>` rows
        """
        start_time = time.time()
        logger.info("Building shing_dict")
        ## <shingles> -> [<shingle>]
        temp_df = shing_df.select(
            explode(shing_df[self.COL_SHINGLES]).alias(self.COL_SHINGLE)
        )

        ## <shingle>-<count>
        item_counts = temp_df.groupBy(self.COL_SHINGLE).count()

        if self.DO_SORT_SHING_DICT:
            item_counts = item_counts.sort(self.COL_SHINGLE)

        est_size_gb, do_est_safe = is_memory_safe(
            self.SAMPLE_FRACTION, self.MAX_MEM_SIZE, item_counts, "shing_dict"
        )

        if not do_est_safe:
            raise MemoryError(
                f"shing_dict={est_size_gb} GB > {self.MAX_MEM_SIZE} GB. Please increase memory"
            )

        collect_item_counts = item_counts.collect()

        # Assign an index to each shingle
        shing_dict = {
            tuple(row[self.COL_SHINGLE]): i for i, row in enumerate(collect_item_counts)
        }

        # Length of the bool vector
        shing_len = len(shing_dict)
        elapsed_time = time.time() - start_time
        logger.info(
            f"Shing_dict[len={shing_len}] build is done. Took {elapsed_time:.5f} s."
        )

        # Return collect_item_counts for the hash_dict step
        return shing_dict, shing_len, collect_item_counts

    def _build_hash_dict(
        self, shing_dict: dict[tuple[str, ...], int], collect_item_counts: list[sql.Row]
    ):
        """Pre-compute Hashes for MinHash step. Expect outputs from `_build_shing_dict()`

        ### Inputs:
            shing_dict (dict[tuple[str, ...], int]): `<shingle>--><idx>` pairs
            collect_item_counts (list[sql.Row]): List of `<shingle>-<count>` rows
        ### Returns:
            hash_dict `{<shing_idx>: <mh_arr>}`
        """
        start_time = time.time()
        # Use the collected item_counts df instead
        ## list of common shingles' index
        common_shing = [
            shing_dict[tuple(row[self.COL_SHINGLE])]
            for row in collect_item_counts
            if row["count"] > self.COMMON_THRES
        ]

        common_shing_set = set(common_shing)

        logger.info("Precomputing minhashes")
        # Precompute MinHashes
        # (Since we can't modify the hash_dict while computing on the RDD)
        hash_dict = dict()  ## {<shing_idx>: <mh_arr>}

        for i, shing_idx in enumerate(common_shing_set):
            hash_dict[shing_idx] = np.array(
                [hash_fn(shing_idx) for hash_fn in self.hash_family]
            )
            est_size_gb, do_est_safe = py_is_memory_safe(hash_dict, self.MAX_MEM_SIZE)
            if i % self.HASH_DICT_MEM_CHECK_FREQ == 0 and not do_est_safe:
                raise MemoryError(
                    f"hash_dict={est_size_gb} GB > {self.MAX_MEM_SIZE} GB. Please increase memory"
                )

        elapsed_time = time.time() - start_time

        logger.info(
            f"Precomputed {len(hash_dict)} * {self.NUM_HASH} minhashes. Took {elapsed_time:.5f} s"
        )
        return hash_dict

    def cache_dfs(self):
        """Caching minhash_df and lsh_df for faster processing (Memory intensive)"""
        logger.info("Caching minhash_df and lsh_df")
        self.minhash_df.cache()
        self.lsh_df.cache()
        logger.info("Caching done")

    def free_dfs(self):
        """Clearing cached minhash_df and lsh_df"""
        logger.info("Clearing minhash_df and lsh_df")
        self.minhash_df.unpersist()
        self.lsh_df.unpersist()
        logger.info("Clearing done")

    @classmethod
    def read_from_txt(
        cls, filepath: str, sc: SparkContext, sqlContext: SparkSession, trim: int = 0
    ) -> "PySparkMinHashLSH":
        """Open text file as a PySpark DataFrame. Each line is a row.
        Order preserved, index unique and monotonically increasing, but not consecutive

        Optional Input: trim (int): Number of documents to take. By default loading all."""
        if not os.path.isfile(filepath):
            raise ValueError(f"Could not find file at: {filepath}")
        start_time = time.time()
        logger.info("Reading the file")
        # Read the text file into a DataFrame
        df: sql.DataFrame = sqlContext.read.text(filepath).withColumn(
            cls.COL_ID, monotonically_increasing_id()
        )

        if trim > 0:
            logger.info("Trimming")
            df = df.limit(trim)

        ## <text>-<id>
        df = df.withColumnRenamed("value", cls.COL_TEXT)
        elapsed_time = time.time() - start_time
        logger.info(
            f"Load success. Received [{df.count()}] {filepath}. Took {elapsed_time:.5f} s"
        )

        return PySparkMinHashLSH(df, sc, sqlContext)
