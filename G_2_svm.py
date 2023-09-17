#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sklearn

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, to_timestamp, to_date
from pyspark.sql.window import Window

from pyspark.ml.evaluation import BinaryClassificationEvaluator,  MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.util import MLUtils
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.classification import LinearSVC
from sklearn.metrics import confusion_matrix


# In[2]:


spark = SparkSession.builder.config("spark.ui.port","4041").master("yarn").appName('Siemens_Stock_Market_Prediction').getOrCreate()
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")


# In[3]:


spark


# In[4]:


df = spark.read.option('header', 'true').csv("gs://bigdata_project_sfe/SIEMENS_minute_data_with_indicators.csv", sep = ',' , inferSchema = True)
# df = spark.read.option('header', 'true').csv("SIEMENS_minute_data_with_indicators.csv", sep = ',' , inferSchema = True).limit(1000)


# In[5]:


df.show(5)


# In[6]:


columns_list = df.columns
print(columns_list)


# ## Label (buy_or_sell) column 
# 
# ### NOTE: COMMENT WHEN NOT RUNNING ORIGINAL CSV

# In[7]:


df.createOrReplaceTempView("stock_data")
data = df


# In[8]:


# Create a new DataFrame with the 'buy_or_sell' column using Spark SQL
buy_sell_df = spark.sql("SELECT date, high, LAG(high) OVER (ORDER BY date) AS next_high FROM stock_data")
buy_sell_df = buy_sell_df.withColumn('buy_or_sell', when(buy_sell_df['next_high'] > buy_sell_df['high'], 1).otherwise(0))


# In[9]:


data = data.join(buy_sell_df.select('date', 'buy_or_sell'), on='date')


# In[10]:


print(data.columns)


# In[11]:


features = ['close', 'high', 'low', 'open', 'volume', 'sma5', 'sma10', 'sma15', 'sma20', 'ema5', 'ema10', 'ema15', 'ema20', 'upperband', 'middleband', 'lowerband', 'HT_TRENDLINE', 'KAMA10', 'KAMA20',
            'KAMA30', 'SAR', 'TRIMA5', 'TRIMA10', 'TRIMA20', 'ADX5', 'ADX10', 'ADX20', 'APO', 'macd510', 'macd520', 'macd1020', 'macd1520', 'macd1226', 'MFI', 'MOM10', 'MOM15', 'MOM20', 'ROC5', 'ROC10', 'ROC20', 'PPO', 'RSI8',
            'slowk', 'slowd', 'fastk', 'fastd', 'fastksr', 'fastdsr', 'ULTOSC', 'WILLR', 'ATR', 'Trange', 'TYPPRICE', 'HT_DCPERIOD', 'BETA']
label = 'buy_or_sell'


# ## SVM

# In[12]:


data = data.drop('date')


# In[13]:


# data.columns


# In[14]:


# data.show


# In[15]:


# print(data.printSchema()) 


# In[16]:


# Prepare the features for training by assembling them into a vector
# ( combine all feature data and separate 'label' data in a dataset, with VectorAssembler.)
assembler = VectorAssembler(inputCols= features, outputCol="features")


# In[17]:


va_df = assembler.transform(data)
va_df = va_df.select(['features', 'buy_or_sell'])
va_df.show(3)


# In[18]:


##data = data.limit(1000)


# In[19]:


# Split the data into training and testing sets
(train, test) = va_df.randomSplit([0.7, 0.3], seed=123)


# In[20]:


# Train the SVM model
svm = LinearSVC(maxIter=10, regParam=0.1, labelCol="buy_or_sell")
model = svm.fit(train)


# In[21]:


# Make predictions on the testing set
predictions = model.transform(test)
predictions.show(3)


# In[22]:


evaluator=MulticlassClassificationEvaluator(metricName="accuracy", labelCol="buy_or_sell")
acc = evaluator.evaluate(predictions)

y_pred=predictions.select("prediction").collect()
y_orig=predictions.select("buy_or_sell").collect()

cm = confusion_matrix(y_orig, y_pred)


# In[23]:


evaluator_roc = BinaryClassificationEvaluator(labelCol=label, metricName="areaUnderROC")
evaluator_acc = MulticlassClassificationEvaluator(labelCol=label, predictionCol='prediction', metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol=label, predictionCol='prediction', metricName="f1")

aucroc = evaluator_roc.evaluate(predictions)
f1 = evaluator_f1.evaluate(predictions)


# In[24]:


print('SVM summary stats')
print('__________________________________')
#print("Confusion Matrix:")
#print(cm)
print(f'Area under ROC: {aucroc}')
print(f'F1 (for label 1.0): {f1}')
print("Accuracy: ", acc)


# In[28]:


# Stop Spark session 
spark.stop()

