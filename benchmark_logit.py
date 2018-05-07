## Benchmark testing, pyspark.ml.classification vs sklearn.linear_model
## Target classification algorithm: Binary logistic regression
## Use the MNIST dataset for demonstration
## Label: Whether the digit is 5

import sklearn
from sklearn.datasets import fetch_mldata
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import numpy as np
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
import time

# import dataset

mnist = fetch_mldata('MNIST original')
mnist_X = mnist['data']

# target: if the digit is 5

mnist_y = mnist['target']==5
mnist_data = np.c_[mnist_X, mnist_y]

## 1. PySpark
# Create a SparkSession

spark = SparkSession.builder \
     .master("local") \
     .appName("Logistic Regression") \
     .getOrCreate()

# data preprocessing

ncols = mnist_data.shape[1]
colnames = []
for i in np.arange(1,ncols, 1):
    colnames.append('V'+str(i))
colnames.append('label')

schema = []
for cname in colnames:
    schema.append(StructField(cname, IntegerType(), True))
schema[-1] = StructField('label', IntegerType(), True)
schema = StructType(schema)
mnist_data = mnist_data.tolist()
mnist_dataframe = spark.createDataFrame(mnist_data, schema)

colnames.pop()
assembler = VectorAssembler(
    inputCols=colnames,
    outputCol="features")

mnist_dataframe = assembler.transform(mnist_dataframe)

# Fit the model and measure the execution time

lr_pyspark = LogisticRegression()

start = time.time()
lrPySpark = lr_pyspark.fit(mnist_dataframe)
end = time.time()
print("Model fitting using the pyspark.ml.classification module took {} seconds.".format(round(end-start,2)))

## 2. Do the same thing using scikit-learn

from sklearn.linear_model import LogisticRegression

lr_sklearn = sklearn.linear_model.LogisticRegression()

start = time.time()
lr_sklearn.fit(mnist_X, mnist_y)
end = time.time()

print("Model fitting using the sklearn.linear_model module took {} seconds.".format(round(end-start,2)))
