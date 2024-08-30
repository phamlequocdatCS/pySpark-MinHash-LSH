# pySpark-MinHash-LSH
Implementation of MinHash-LSH for documents similarity, in-memory and pySpark

## Get started

Get the WebOfScience dataset

https://data.mendeley.com/datasets/9rw3vkcfy4/6

Take `WOS5736\X.txt` and rename it as `WebOfScience-5736.txt`

See `run_in_memory.ipynb` and `run_pyspark.ipynb` for examples


### In-memory
```python
from src.minhash_in_memory import InMemoryMinHashLSH

text_file = "WebOfScience-5736.txt"
key_short = "Phytoplasmas are i..."

wos = InMemoryMinHashLSH.read_from_txt(text_file)
wos.run()

print(wos.approxNearestNeighbors(key_short, 10))
```

### PySpark
```python
import findspark
from pyspark import SparkContext
from pyspark.sql import SparkSession

from src.minhash_pyspark import PySparkMinHashLSH

findspark.init()

sc = SparkContext(master="local[*]", appName="MinHasher")
sqlContext = SparkSession.builder.getOrCreate()

text_file = "WebOfScience-5736.txt"
key_short = "Phytoplasmas are i..."

wos_spark = PySparkMinHashLSH.read_from_txt(text_file, sc, sqlContext)
wos_spark.run()

result = wos_spark.approxNearestNeighbors(key_short, 10)
result.show()
wos_spark.free_dfs()
```