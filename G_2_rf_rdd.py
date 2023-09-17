#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import dill

from scipy import stats

import sklearn
from sklearn.decomposition import PCA 

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, col, skewness, kurtosis, lead, when, to_timestamp, to_date
from pyspark.sql.window import Window

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.evaluation import BinaryClassificationEvaluator,  MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.util import MLUtils
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


# In[2]:


spark = SparkSession.builder.config("spark.ui.port","4041").master("yarn").appName('Siemens_Stock_Market_Prediction').getOrCreate()
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")


# In[3]:


spark


# In[5]:


df = spark.read.option('header', 'true').csv("gs://bigdata_project_ct/SIEMENS_minute_data_with_indicators.csv", sep = ',' , inferSchema = True)
# df = spark.read.option('header', 'true').csv("SIEMENS_minute_data_with_indicators.csv", sep = ',' , inferSchema = True).limit(1000)


# In[6]:


df.show()


# In[7]:


columns_list = df.columns
print(columns_list)


# ## Label (buy_or_sell) column 
# 
# ### NOTE: COMMENT WHEN NOT RUNNING ORIGINAL CSV

# In[10]:


df.createOrReplaceTempView("stock_data")
data = df


# In[11]:


# Create a new DataFrame with the 'buy_or_sell' column using Spark SQL
buy_sell_df = spark.sql("SELECT date, high, LAG(high) OVER (ORDER BY date) AS next_high FROM stock_data")
buy_sell_df = buy_sell_df.withColumn('buy_or_sell', when(buy_sell_df['next_high'] > buy_sell_df['high'], 1).otherwise(0))


# In[12]:


data = data.join(buy_sell_df.select('date', 'buy_or_sell'), on='date')


# In[13]:


print(data.columns)


# In[15]:


# features = ['close', 'high', 'low', 'open', 'volume', 'sma5', 'sma10', 'sma15', 'sma20', 'ema5', 'ema10', 'ema15', 'ema20', 'upperband', 'middleband', 'lowerband', 'HT_TRENDLINE', 'KAMA10', 'KAMA20',
#             'KAMA30', 'SAR', 'TRIMA5', 'TRIMA10', 'TRIMA20', 'ADX5', 'ADX10', 'ADX20', 'APO', 'macd510', 'macd520', 'macd1020', 'macd1520', 'macd1226', 'MFI', 'MOM10', 'MOM15', 'MOM20', 'ROC5', 'ROC10', 'ROC20', 'PPO', 'RSI8',
#             'slowk', 'slowd', 'fastk', 'fastd', 'fastksr', 'fastdsr', 'ULTOSC', 'WILLR', 'ATR', 'Trange', 'TYPPRICE', 'HT_DCPERIOD', 'BETA']
# label = 'buy_or_sell'


# ## Random forest with RDD

# In[42]:


data = data.drop('date')


# In[43]:


# data.columns


# In[44]:


# Split data into training and test sets
train_data, test_data = data.randomSplit([0.7, 0.3], seed=42)


# In[45]:


# Convert data to RDDs
train_rdd = MLUtils.convertVectorColumnsToML(train_data).rdd
train_rdd = train_rdd.repartition(10)
test_rdd = MLUtils.convertVectorColumnsToML(test_data).rdd
test_rdd = test_rdd.repartition(10)


# ### Creating the labeled point for the model

# In[46]:


# train_rdd.first()


# In[48]:


header = train_rdd.first()
label_index = header.__fields__.index('buy_or_sell')


# In[53]:


labeled_points_train_rdd = train_rdd.map(lambda row: LabeledPoint(row[label_index], row[:label_index] + row[label_index+1:]))


# In[51]:


model = RandomForest.trainClassifier(labeled_points_train_rdd, numClasses=2, categoricalFeaturesInfo={}, numTrees=10, 
                                         featureSubsetStrategy="auto", impurity="gini", maxDepth=10, maxBins=32, seed=42)
    


# In[52]:


header = test_rdd.first()
label_index = header.__fields__.index('buy_or_sell')


# In[54]:


labeled_points_test_rdd = test_rdd.map(lambda row: LabeledPoint(row[label_index], row[:label_index] + row[label_index+1:]))


# In[61]:


# Make predictions on test data
predictions = model.predict(labeled_points_test_rdd.map(lambda x: x.features))


# In[62]:


# predictions.first()


# In[78]:


# Compute the area under the ROC curve
labels_and_predictions = labeled_points_test_rdd.map(lambda lp: lp.label).zip(predictions)
metrics = BinaryClassificationMetrics(labels_and_predictions)
metrics_2 = MulticlassMetrics(labels_and_predictions)
area_under_roc = metrics.areaUnderROC
accuracy = metrics_2.accuracy
f_1 = metrics_2.fMeasure(1.0)


# In[79]:


print('Random Forest RDD summary stats')
print('__________________________________')
print(f'Area under ROC: {area_under_roc}')
print(f'F1 (for label 1.0): {f_1}')
print(f'Accuracy: {accuracy}')
      

