#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


spark = SparkSession.builder \
.config("spark.ui.port","4041") \
.config("spark.executor.instances", "4") \
.config("spark.default.parallelism", "2") \
.config("spark.executor.cores", "2") \
.master("yarn") \
.appName('Siemens_Stock_Market_Prediction') \
.getOrCreate()

spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")


# In[4]:


spark


# In[5]:


df = spark.read.option('header', 'true').csv("gs://bigdata_project_ct/SIEMENS_minute_data_with_indicators.csv", sep = ',' , inferSchema = True)
# df = spark.read.option('header', 'true').csv("SIEMENS_minute_data_with_indicators.csv", sep = ',' , inferSchema = True).limit(1000)


# In[6]:


df.show()


# In[7]:


columns_list = df.columns
print(columns_list)


# In[11]:


# Convert the 'date' column to a date (without time) and add it as a new column 'date_only'
df = df.withColumn('date_only', to_date(df['date']))
df.createOrReplaceTempView("stock_data")


# In[12]:


avg_close_df = spark.sql("SELECT date_only, AVG(close) AS avg_close, AVG(lag_close) AS avg_close_prev_day FROM (SELECT date_only,close, LAG(close) OVER (ORDER BY date_only) AS lag_close FROM stock_data) subquery GROUP BY date_only")


# In[13]:


avg_close_df.show()


# In[15]:


data = df.join(avg_close_df, on='date_only')


# ## Label (buy_or_sell) column 
# 
# ### NOTE: COMMENT WHEN NOT RUNNING ORIGINAL CSV

# In[16]:


# Create a new DataFrame with the 'buy_or_sell' column using Spark SQL
buy_sell_df = spark.sql("SELECT date, high, LAG(high) OVER (ORDER BY date) AS next_high FROM stock_data")
buy_sell_df = buy_sell_df.withColumn('buy_or_sell', when(buy_sell_df['next_high'] > buy_sell_df['high'], 1).otherwise(0))


# In[17]:


buy_sell_df.show()


# In[18]:


# Convert the DataFrame to an RDD
rdd = data.select('date_only', 'open').rdd
rdd = rdd.repartition(10)


# In[19]:


def map_func(row):
    return (row['date_only'], (row['open'], 1))

# Define a reduce function that takes two tuples (open1, count1) and (open2, count2) and returns a new tuple (open1 + open2, count1 + count2)
def reduce_func(x, y):
    return (x[0] + y[0], x[1] + y[1])

# Apply the map function to the RDD
mapped_rdd = rdd.map(map_func)

# Apply the reduce function to the mapped RDD using the reduceByKey transformation
reduced_rdd = mapped_rdd.reduceByKey(reduce_func)

# Compute the average open price for each day by mapping the reduced RDD
result_rdd = reduced_rdd.map(lambda x: (x[0], x[1][0] / x[1][1]))

# Convert the result RDD back to a DataFrame
avg_open_df = result_rdd.toDF(['date_only', 'avg_open'])

# Join the avg_open_df DataFrame with the original data DataFrame on the 'date_only' column
data = data.join(avg_open_df, on='date_only')

data = data.join(buy_sell_df.select('date', 'buy_or_sell'), on='date')


# In[20]:


print(data.columns)


# ## Moving Average Convergence Divergence (MACD): 
# This feature is derived from the MACD indicator and measures the difference between the MACD line and the signal line. It could be calculated as macd510 - macd1226.
# 
# The values 12 and 26 are commonly used as the number of periods (usually days) used in the EMA calculation for the MACD indicator. The values used in the conditions are the weight factors that determine how much weight should be given to the current close price and the previous EMA value when computing the current EMA.
# 
# For example, when computing the EMA12, the current close price has a weight factor of 2/13 and the previous EMA12 value has a weight factor of 11/13. Similarly, when computing the EMA26, the current close price has a weight factor of 2/27 and the previous EMA26 value has a weight factor of 25/27.
# 
# Finally, when computing the MACD, the value 10 is commonly used as the number of periods for the signal line. The values used in the condition (2.0 / 10) and (8.0 / 10) determine the weight given to the current MACD value and the previous MACD value when computing the current MACD.

# In[21]:


len(data.columns)


# In[22]:


rdd = data.select('date', 'macd510', 'macd1226').rdd
rdd = rdd.repartition(10)

# Define a function to compute the MACD
def macd_func(row):
    macd510 = row['macd510']
    macd1226 = row['macd1226']
    macd = macd510 - macd1226
    return (row['date'], macd)

# Apply the map function to the RDD
mapped_rdd = rdd.map(macd_func)


# Convert the result RDD back to a DataFrame
macd_df = mapped_rdd.toDF(['date', 'macd'])

macd_df.show()


# In[23]:


data = data.join(macd_df, on='date')


# In[24]:


data = data.drop('macd510', 'macd520','macd1020', 'macd1520','macd1226','ema5', 'ema10', 'ema15', 'ema20')


# In[26]:


print(data.columns)


# ## Bollinger Bands Width:
# This feature measures the width of the Bollinger Bands and could be calculated as (upperband - lowerband) / middleband.

# In[28]:


rdd = data.select('date', 'upperband', 'lowerband', 'middleband').rdd
rdd = rdd.repartition(10)

# Define a function to compute the Bollinger Bands width for a row
def bollinger_bands_width_func(row):
    upperband = row['upperband']
    lowerband = row['lowerband']
    middleband = row['middleband']
    width = (upperband - lowerband) / middleband
    return (row['date'], width)

# Apply the map function to the RDD
mapped_rdd = rdd.map(bollinger_bands_width_func)

# Convert the result RDD back to a DataFrame
bollinger_bands_width_df = mapped_rdd.toDF(['date', 'bollinger_bands_width'])

#bollinger_bands_width_df.show(1)
bollinger_bands_width_df.show()


# In[30]:


# Join the avg_open_df DataFrame with the original data DataFrame on the 'date' column
data = data.join(bollinger_bands_width_df, on='date')


# In[31]:


data = data.drop('upperband', 'lowerband', 'middleband')


# ## Commodity Channel Index (CCI) Divergence: 
# This feature measures the divergence between the CCI indicator and the close price and could be calculated as CCI5 - close.

# In[32]:


rdd = data.select('date', 'CCI5', 'close').rdd
rdd = rdd.repartition(10)


def com_channel_index(row):
    CCI5 = row['CCI5']
    close = row['close']
    diff = CCI5 - close
    return (row['date'], diff)

# Apply the map function to the RDD
mapped_rdd = rdd.map(com_channel_index)

# Convert the result RDD back to a DataFrame
com_channel_index_df = mapped_rdd.toDF(['date', 'com_channel_index'])


com_channel_index_df.show()


# In[33]:


data = data.join(com_channel_index_df, on='date')


# In[34]:


data = data.drop('CCI5','CCI10','CCI15')


# ## Relative Strength Index (RSI) Divergence:
# This feature measures the divergence between the RSI indicator and the close price and could be calculated as RSI14 - close.

# In[35]:


rdd = data.select('date', 'RSI14', 'close').rdd
rdd = rdd.repartition(10)

# Define a function to compute the Bollinger Bands width for a row
def rsi(row):
    RSI14 = row['RSI14']
    close = row['close']
    diff = RSI14 - close
    return (row['date'], diff)

# Apply the map function to the RDD
mapped_rdd = rdd.map(rsi)

# Convert the result RDD back to a DataFrame
rsi_df = mapped_rdd.toDF(['date', 'rsi'])


rsi_df.show()


# In[36]:


data = data.join(rsi_df, on='date')


# In[37]:


data = data.drop('RSI14','RSI18', 'RSI8')


# ## Momentum: 
# This feature measures the rate of change of the price and could be calculated as close - close.shift(1).

# In[39]:


rdd = data.select('date_only', 'avg_close', 'avg_close_prev_day').rdd
rdd = rdd.repartition(10)

# Define a function to compute the Bollinger Bands width for a row
def momentum(row):
    avg_close = row['avg_close']
    avg_close_prev_day = row['avg_close_prev_day']
    mom = avg_close - avg_close_prev_day
    return (row['date_only'], mom)

# Apply the map function to the RDD
mapped_rdd = rdd.map(momentum)

# Convert the result RDD back to a DataFrame
momentum_df = mapped_rdd.toDF(['date_only', 'momentum'])

momentum_df.createOrReplaceTempView("momentum_df")

momentum_df = spark.sql("SELECT date_only, AVG(momentum) as momentum FROM momentum_df GROUP BY date_only ORDER BY date_only")


momentum_df.show()


# In[40]:


data = data.join(momentum_df, on='date_only')
data = data.drop('MOM20', 'MOM15', 'MOM10')


# A Simple Moving Average (SMA) is a calculation of the arithmetic mean or average price of a financial security over a defined period of time. The defined period of time is the window size, which can be any number of time periods, such as 5, 10, 15, or 20 days.
# 
# For example, 'sma5' is the average price of the stock over the past 5 days, 'sma10' is the average price over the past 10 days, 'sma15' is the average price over the past 15 days, and 'sma20' is the average price over the past 20 days.
# 
# These indicators are used by traders and analysts to identify the direction of a trend and to determine levels of support and resistance. When the price of a stock is above its moving average, it is considered to be in an uptrend, and when the price is below its moving average, it is considered to be in a downtrend.

# In[42]:


rdd = data.select('date_only',  'sma5',  'sma10', 'sma15',  'sma20').rdd
rdd = rdd.repartition(10)

# Define a function to compute the Bollinger Bands width for a row
def sma_calc(row):
    sma5 = row['sma5']
    sma10 = row['sma10']
    sma15 = row['sma15']
    sma20 = row['sma20']
    sma = (sma5 + sma10 + sma15 + sma20) / 4
    return (row['date_only'], sma)

# Apply the map function to the RDD
mapped_rdd = rdd.map(sma_calc)

# Convert the result RDD back to a DataFrame
sma_df = mapped_rdd.toDF(['date_only', 'sma'])

sma_df.createOrReplaceTempView("sma_df")

sma_df = spark.sql("SELECT date_only, AVG(sma) as sma FROM sma_df GROUP BY date_only ORDER BY date_only")


sma_df.show()


# In[44]:


data = data.join(sma_df, on='date_only')
data = data.drop('sma5',  'sma10', 'sma15',  'sma20')


# KAMA stands for Kaufman's Adaptive Moving Average, which is a technical analysis indicator used to smooth out price fluctuations in a financial instrument's price chart. It is a type of moving average that adapts to market changes based on volatility.

# In[45]:


rdd = data.select('date_only', 'KAMA10', 'KAMA20', 'KAMA30').rdd
rdd = rdd.repartition(10)

# Define a function to compute the Bollinger Bands width for a row
def kama_calc(row):
    KAMA10 = row['KAMA10']
    KAMA20 = row['KAMA20']
    KAMA30 = row['KAMA30']
    kama = (KAMA10 + KAMA20 + KAMA30 ) / 3
    return (row['date_only'], kama)

# Apply the map function to the RDD
mapped_rdd = rdd.map(kama_calc)

# Convert the result RDD back to a DataFrame
kama_df = mapped_rdd.toDF(['date_only', 'kama'])

kama_df.createOrReplaceTempView("kama_df")

kama_df = spark.sql("SELECT date_only, AVG(kama) as kama FROM kama_df GROUP BY date_only ORDER BY date_only")


kama_df.show()


# In[46]:


data = data.join(kama_df, on='date_only')
data = data.drop('KAMA10', 'KAMA20', 'KAMA30')


# In[47]:


rdd = data.select('date_only', 'ADX5', 'ADX10', 'ADX20').rdd
rdd = rdd.repartition(10)

# Define a function to compute the Bollinger Bands width for a row
def adx_calc(row):
    ADX5 = row['ADX5']
    ADX10 = row['ADX10']
    ADX20 = row['ADX20']
    adx = (ADX5 + ADX10 + ADX20 ) / 3
    return (row['date_only'], adx)

# Apply the map function to the RDD
mapped_rdd = rdd.map(adx_calc)

# Convert the result RDD back to a DataFrame
adx_df = mapped_rdd.toDF(['date_only', 'adx'])

adx_df.createOrReplaceTempView("adx_df")

adx_df = spark.sql("SELECT date_only, AVG(adx) as adx FROM adx_df GROUP BY date_only ORDER BY date_only")


adx_df.show()


# In[48]:


data = data.join(adx_df, on='date_only')
data = data.drop('ADX5', 'ADX10', 'ADX20')


# In[49]:


rdd = data.select('date_only', 'ROC5', 'ROC10','ROC20').rdd
rdd = rdd.repartition(10)

# Define a function to compute the Bollinger Bands width for a row
def roc_calc(row):
    ROC5 = row['ROC5']
    ROC10 = row['ROC10']
    ROC20 = row['ROC20']
    roc = (ROC5 + ROC10 + ROC20) / 3
    return (row['date_only'], roc)

# Apply the map function to the RDD
mapped_rdd = rdd.map(roc_calc)

# Convert the result RDD back to a DataFrame
roc_df = mapped_rdd.toDF(['date_only', 'roc'])

roc_df.createOrReplaceTempView("roc_df")

roc_df = spark.sql("SELECT date_only, AVG(roc) as roc FROM roc_df GROUP BY date_only ORDER BY date_only")


roc_df.show()


# In[50]:


data = data.join(roc_df, on='date_only')
data = data.drop('ROC5', 'ROC10','ROC20')


# In[51]:


rdd = data.select('date_only', 'TRIMA5', 'TRIMA10', 'TRIMA20').rdd
rdd = rdd.repartition(10)

# Define a function to compute the Bollinger Bands width for a row
def trima_calc(row):
    TRIMA5 = row['TRIMA5']
    TRIMA10 = row['TRIMA10']
    TRIMA20 = row['TRIMA20']
    trima = (TRIMA5 + TRIMA10 + TRIMA20) / 3
    return (row['date_only'], trima)

# Apply the map function to the RDD
mapped_rdd = rdd.map(trima_calc)

# Convert the result RDD back to a DataFrame
trima_df = mapped_rdd.toDF(['date_only', 'trima'])

trima_df.createOrReplaceTempView("trima_df")

trima_df = spark.sql("SELECT date_only, AVG(trima) as trima FROM trima_df GROUP BY date_only ORDER BY date_only")


trima_df.show()


# In[52]:


data = data.join(trima_df, on='date_only')
data = data.drop('TRIMA5', 'TRIMA10', 'TRIMA20')


# In[54]:


data.columns


# ## Random forest with RDD

# In[56]:


data = data.drop('date', 'date_only')


# In[57]:


data.columns


# In[58]:


# Split data into training and test sets
train_data, test_data = data.randomSplit([0.7, 0.3], seed=42)


# In[59]:


# Convert data to RDDs
train_rdd = MLUtils.convertVectorColumnsToML(train_data).rdd
train_rdd = train_rdd.repartition(10)
test_rdd = MLUtils.convertVectorColumnsToML(test_data).rdd
test_rdd = test_rdd.repartition(10)


# ### Creating the labeled point for the model

# In[60]:


train_rdd.first()


# In[61]:


header = train_rdd.first()
label_index = header.__fields__.index('buy_or_sell')


# In[62]:


label_index


# In[63]:


labeled_points_train_rdd = train_rdd.map(lambda row: LabeledPoint(row[label_index], row[:label_index] + row[label_index+1:]))


# In[65]:


labeled_points_train_rdd.first()


# In[66]:


model = RandomForest.trainClassifier(labeled_points_train_rdd, numClasses=2, categoricalFeaturesInfo={}, numTrees=10, 
                                         featureSubsetStrategy="auto", impurity="gini", maxDepth=10, maxBins=32, seed=42)
    


# In[67]:


header = test_rdd.first()
label_index = header.__fields__.index('buy_or_sell')


# In[68]:


labeled_points_test_rdd = test_rdd.map(lambda row: LabeledPoint(row[label_index], row[:label_index] + row[label_index+1:]))


# In[69]:


# Make predictions on test data
predictions = model.predict(labeled_points_test_rdd.map(lambda x: x.features))


# In[70]:


predictions.first()


# In[71]:


# Compute the area under the ROC curve
labels_and_predictions = labeled_points_test_rdd.map(lambda lp: lp.label).zip(predictions)
metrics = BinaryClassificationMetrics(labels_and_predictions)
metrics_2 = MulticlassMetrics(labels_and_predictions)
area_under_roc = metrics.areaUnderROC
accuracy = metrics_2.accuracy
f_1 = metrics_2.fMeasure(1.0)


# In[72]:


print('Random Forest RDD feat eng summary stats')
print('__________________________________')
print(f'Area under ROC: {area_under_roc}')
print(f'F1 (for label 1.0): {f_1}')
print(f'Accuracy: {accuracy}')
      


# In[ ]:




