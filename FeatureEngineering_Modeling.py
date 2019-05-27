import os
pyspark_submit_args = '--packages org.mongodb.spark:mongo-spark-connector_2.11:2.4.0 pyspark-shell'
os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import *
from pyspark.ml.evaluation import ClusteringEvaluator
import math
import time
from matplotlib import pyplot as plt
from pyspark.sql.window import Window as W

instance_name = "m4_xlarge"

###################
## Set Configure ##
###################

sparkConf = SparkConf().setMaster("local[3]")\
                       .setAppName("project")\
                       .setAll([('spark.excutor.memory', '16g'),
                                ('spark.executor.cores', '8'),
                                ('spark.cores.max', '12'),
                                ('spark.driver.memory', '16g'),
                                ('spark.driver.maxResultSize', '16g')])\
                       .setLogLevel("OFF")

##########################
## load data from mongo ##
##########################

sparkConf.set("spark.mongodb.input.uri", "mongodb://3.88.54.198/nypt.f_15_new")
#sc = SparkContext(conf = sparkConf)

spark = SparkSession \
    .builder \
    .appName("myApp") \
    .config(conf=sparkConf)\
    .getOrCreate()

df_15 = spark.read.format("com.mongodb.spark.sql.DefaultSource").load()

df_15.printSchema()

sparkConf.set("spark.mongodb.input.uri", "mongodb://3.88.54.198/nypt.f_16_new")
spark = SparkSession \
    .builder \
    .appName("myApp") \
    .config(conf=sparkConf)\
    .getOrCreate()

df_16 = spark.read.format("com.mongodb.spark.sql.DefaultSource").load()
df_16.printSchema()

sparkConf.set("spark.mongodb.input.uri", "mongodb://3.88.54.198/nypt.f_17_new")
spark = SparkSession \
    .builder \
    .appName("myApp") \
    .config(conf=sparkConf)\
    .getOrCreate()

df_17 = spark.read.format("com.mongodb.spark.sql.DefaultSource").load()

df_17.printSchema()

sparkConf.set("spark.mongodb.input.uri", "mongodb://3.88.54.198/nypt.f_18_new")
spark = SparkSession \
    .builder \
    .appName("myApp") \
    .config(conf=sparkConf)\
    .getOrCreate()

df_18 = spark.read.format("com.mongodb.spark.sql.DefaultSource").load()

df_18.printSchema()

df_56 = df_15.union(df_16)
df_567 = df_56.union(df_17)
df_567 = df_567.drop("plate_type")
df = df_567.union(df_18)

df.printSchema()

##############################################
## data clearning: filtering parking ticket ##
##############################################

parking_code = [6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27, 32,
				35, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55,
             	56, 57, 58, 59, 60, 61, 62, 63, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
             	76, 77, 78, 80, 82, 83, 84, 85, 86, 89, 91, 92, 93, 96, 97, 98, 99]

cluster_cols = ['veh_color_group',\
        'issue_month', 'issue_quarter', 'issue_weekday',\
        'registration_state','violation_time',\
        'plate_type_class','veh_body_type_class',\
        'veh_make_class']

local_cols = ['violation_precinct']

df_parking = df.filter(df["violation_code"].isin(parking_code))
df_parking = df.filter(df["violation_code"].isin(parking_code))

df_nona = df_parking.filter(
    (col("violation_time") < 24) & (col("violation_time")>=0)).persist()

df_model = df_nona.select(cluster_cols)

print(df_model.columns)

str_cols = ['veh_color_group',\
        'registration_state',
            'plate_type_class','veh_body_type_class', 'veh_make_class']

df_nona.count()

##############################################
## feature engineering: OneHotEncoding & SI ##
##############################################

def indexStringColumns(df, cols):
    #variable newdf will be updated several times
    newdf = df
    for c in cols:
        #For each given colum, fits StringIndexerModel.
        si = StringIndexer(inputCol=c, outputCol=c+"-num")
        sm = si.fit(newdf)
        newdf = sm.transform(newdf).drop(c)
        newdf = newdf.withColumnRenamed(c+"-num", c)
    
    return newdf

dfnumeric = indexStringColumns(df_model, str_cols)

# 2. one hot encoder to transform categorical data
def oneHotEncodeColumns(df, cols):
    newdf = df
    for c in cols:
        #For each given colum, create OneHotEncoder. 
        onehotenc = OneHotEncoder(inputCol=c, outputCol=c+"-onehot", dropLast=False)
        newdf = onehotenc.transform(newdf).drop(c)
        newdf = newdf.withColumnRenamed(c+"-onehot", c)
    return newdf
    
dfhot = oneHotEncodeColumns(dfnumeric,dfnumeric.columns)

# 3. feature vectors

va = VectorAssembler(outputCol='features',inputCols=dfhot.columns)
df_transformed = va.transform(dfhot).select('features').cache()
df_transformed.count()


###################
## Model: kmeans ##
###################


start = time.time()
kmeans = KMeans().setK(10).setSeed(42).setFeaturesCol("features")
model = kmeans.fit(df_transformed)

# Make predictions

#evaluate the kmeans
wssse = model.computeCost(df_transformed) 
time_model = time.time() - start

print("Use K = 10" )
#print("Within Set Sum of Squared Errors = " + str(wssse))
average_dist = math.sqrt(wssse/df_transformed.count())
print("Average distance from the center = " + str(average_dist))

print("K=10 time:" + str(time_model))

##################################
## Model: kmeans k from 2 to 20 ##
##################################

cost = [0 for x in range(20)]
time_train = [0 for x in range(20)]
for k in range(2,20):
    start = time.time()
    kmeans = KMeans().setK(k).setSeed(42).setFeaturesCol("features")
    model_kmeans = kmeans.fit(df_transformed.sample(False,0.1, seed=42))
    cost[k] = model_kmeans.computeCost(df_transformed)
    time_train[k] = time.time()-start

fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,20),[math.sqrt(c/df_transformed.count()) for c in cost[2:20]])
ax.set_xlabel('k')
ax.set_ylabel('Within Set Average distance from the center')
ax.set_title('Cost vs Different k in K-Means Clustering')
fig.savefig(instance_name+'.png')

fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,20),time_train[2:20])
ax.set_xlabel('k')
ax.set_ylabel('Training time (seconds)')
ax.set_title('Training time vs Different k in K-Means Clustering')
fig.savefig(instance_name+'_time.png')

## Best_model
start = time.time()
k = 12
kmeans = KMeans(k = k, maxIter = 200, tol = 0.1, seed=42)
model = kmeans.fit(df_transformed)

# Make predictions

#evaluate the kmeans
wssse = model.computeCost(df_transformed) 
time_model = time.time() - start
print("Use K = " + str(k))
#print("Within Set Sum of Squared Errors = " + str(wssse))
best_average_dist = math.sqrt(wssse/df_transformed.count())

print("Average distance from the center = " + str(best_average_dist))
best_model_time = time.time() - start

print("k=12 time:"+ str(best_model_time))

############################################
## make prediction and do recommentdation ##
############################################

transformed = model.transform(df_transformed)

df_nona = df_nona.withColumn("idx", monotonically_increasing_id())
windowSpec = W.orderBy("idx")
df_nona = df_nona.withColumn("idx", row_number().over(windowSpec))

transformed = transformed.withColumn("idx", monotonically_increasing_id())
windowSpec = W.orderBy("idx")
transformed = transformed.withColumn("idx", row_number().over(windowSpec))

# now, we can join the dataframes
df_final = df_nona.join(transformed.select(['prediction', "idx"]), 
                        df_nona.idx == transformed.idx, how = "left").persist()

df_final.groupby("prediction").count().orderBy("count",ascending=False).show()

#####################
## for top 3 group ##
#####################

df_final.filter("prediction == 5").groupby(
	"violation_precinct").count().orderBy("count",ascending=False).show(5)

df_final.filter("prediction == 7").groupby(
	"violation_precinct").count().orderBy("count",ascending=False).show(5)

df_final.filter("prediction == 11").groupby(
	"violation_precinct").count().orderBy("count",ascending=False).show(5)

with open("results.txt", "w") as f:
    f.write("K=10, time:" + str(time_model))
    f.write("K=10 average_distance: " + str(average_dist))
    f.write("best_model_time:" + str(best_model_time))
    f.write("best_average_dist:" + str(best_average_dist))
