import os
pyspark_submit_args = '--packages org.mongodb.spark:mongo-spark-connector_2.11:2.4.0 pyspark-shell'
os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args

from pyspark import SparkContext
from pyspark.sql import SparkSession

sparkConf = SparkConf().setMaster("local[3]")\
                       .setAppName("project")\
                       .setAll([('spark.excutor.memory', '32g'),
                                ('spark.executor.cores', '14'),
                                ('spark.cores.max', '42'),
                                ('spark.driver.memory', '32g'),
                                ('spark.driver.maxResultSize', '32g')])

spark = SparkSession \
    .builder \
    .appName("myApp") \
    .config(conf=sparkConf)\
    .getOrCreate()

 df_load = spark.read.csv("hdfs:///user/hadoop/cc_out.csv") # load data from hdf

 df_load.coalesce(1).write.csv("s3n://nyptdistgroup/output.csv") 