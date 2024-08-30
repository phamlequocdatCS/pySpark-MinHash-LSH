import sys

import pyspark.sql as sql


def py_is_memory_safe(obj, MAX_MEM_SIZE) -> tuple[float, bool]:
    # """Check if object is smaller than `MAX_MEM_SIZE`. Return `True` if it is"""
    # est_size_byte = sys.getsizeof(obj)
    # est_size_gb = est_size_byte / (1024**3)
    # if est_size_gb < MAX_MEM_SIZE:
    #     return est_size_gb, True
    # return est_size_gb, False
    return 0.0, True


def is_memory_safe(
    SAMPLE_FRACTION, MAX_MEM_SIZE, df: sql.DataFrame, obj_name: str = "object"
) -> tuple[float, bool]:
    # """Check if DF's size is smaller than `MAX_MEM_SIZE`, by sampling a
    # fraction and extrapolate
    # ### Inputs:
    #     df (sql.DataFrame): A PySpark DataFrame
    #     obj_name (str) = `'object'`: Name of the object to display message
    # ### Returns:
    #     `True` if size(df) approximately < `MAX_MEM_SIZE`
    # """
    # # Sample x% of the DataFrame
    # sample_df = df.sample(SAMPLE_FRACTION)
    # # Collect the sample to the driver node
    # # Measure the size of the sample
    # sample_size = sys.getsizeof(sample_df.collect())
    # # Estimate the size of the full DataFrame (in GB)
    # est_size_byte = sample_size / SAMPLE_FRACTION
    # est_size_mb = est_size_byte / (1024**2)
    # est_size_gb = est_size_byte / (1024**3)
    # print(f"Estimated {obj_name} size: {est_size_gb:.5f} GB [{est_size_mb:.5f} MB].\
    #     Max Memory size: {MAX_MEM_SIZE} GB")
    # if est_size_gb < MAX_MEM_SIZE:
    #     return est_size_gb, True
    # return est_size_gb, False
    return 0.0, True
