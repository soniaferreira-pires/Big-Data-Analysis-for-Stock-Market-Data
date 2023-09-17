#!/usr/bin/env python
# coding: utf-8

# In[10]:


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
from pyspark.ml.feature import VectorAssembler, PCA, StandardScaler
from pyspark.mllib.util import MLUtils
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


# In[3]:


spark = SparkSession.builder.config("spark.ui.port","4041").master("yarn").appName('Siemens_Stock_Market_Prediction').getOrCreate()
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")


# In[4]:


spark


# In[13]:


df = spark.read.option('header', 'true').csv("gs://bigdataproject-europe2work/SIEMENS_minute_data_with_indicators.csv", sep = ',' , inferSchema = True)
# df = spark.read.option('header', 'true').csv("SIEMENS_minute_data_with_indicators.csv", sep = ',' , inferSchema = True).limit(1000)


# In[14]:


df.show()


# In[15]:


columns_list = df.columns
print(columns_list)


# ## Feature normalization

# In[16]:


all_numeric = ['close','high','low','open','volume','sma5','sma10','sma15','sma20','ema5','ema10','ema15','ema20','upperband','middleband','lowerband','HT_TRENDLINE','KAMA10','KAMA20','KAMA30','SAR','TRIMA5','TRIMA10','TRIMA20','ADX5','ADX10','ADX20','APO','CCI5','CCI10','CCI15','macd510','macd520','macd1020','macd1520','macd1226','MFI','MOM10','MOM15','MOM20','ROC5','ROC10','ROC20','PPO','RSI14','RSI8','slowk','slowd','fastk','fastd','fastksr','fastdsr','ULTOSC','WILLR','ATR','Trange','TYPPRICE','HT_DCPERIOD','BETA']
correlated = ['close', 'high', 'low', 'open', 'sma5', 'sma10', 'sma15', 'sma20', 'ema5', 'ema10', 'ema15', 'ema20', 'upperband','middleband', 'lowerband', 'HT_TRENDLINE', 'KAMA10', 'KAMA20', 'KAMA30', 'TRIMA5', 'TRIMA10', 'TRIMA20']
non_correlated = ['volume','SAR','ADX5','ADX10','ADX20','APO','CCI5','CCI10','CCI15','macd510','macd520','macd1020','macd1520','macd1226','MFI','MOM10','MOM15','MOM20','ROC5','ROC10','ROC20','PPO','RSI14','RSI8','slowk','slowd','fastk','fastd','fastksr','fastdsr','ULTOSC','WILLR','ATR','Trange','TYPPRICE','HT_DCPERIOD','BETA']

assembler_all = VectorAssembler(inputCols = all_numeric, outputCol = 'all_features')
df = assembler_all.transform(df)
all_scaler = StandardScaler(inputCol = 'all_features', outputCol = 'scaled_all_features').fit(df)
df = all_scaler.transform(df)

assembler = VectorAssembler(inputCols = correlated, outputCol = 'corr_features')
df = assembler.transform(df)
corr_scaler = StandardScaler(inputCol = 'corr_features', outputCol = 'scaled_corr_features').fit(df)
df = corr_scaler.transform(df)

assembler = VectorAssembler(inputCols = non_correlated, outputCol = 'non_corr_features')
df = assembler.transform(df)
non_corr_scaler = StandardScaler(inputCol = 'non_corr_features', outputCol = 'scaled_non_corr_features').fit(df)
df = non_corr_scaler.transform(df)


# ## Pricipal Component Analysis (PCA)

# In[17]:


# PCA (Pyspark - all data)

n_components = len(all_numeric)
pca_eval_all = PCA(
    k = n_components, 
    inputCol = 'scaled_all_features', 
    outputCol = 'pca_all_features'
).fit(df)

#importance/removal of features
temp = pca_eval_all.transform(df)
print('Explained Variance Ratio', pca_eval_all.explainedVariance.toArray())


# In[18]:


# PCA (Pyspark - all data) 
# choosing the first 12 components(component %): 1st: 0.40; 2nd: 0.25; 3rd: 0.10; 4th: 0.04; 5th: 0.026; 6th: 0.021; 7th: 0.02; 8th: 0.017;
# 9th: 0.016; 10th: 0.015; 11th: 0.013; 12th: 0.010)

pca_eval_all = PCA(
    k = 12, 
    inputCol = 'scaled_all_features', 
    outputCol = 'pca_all_features'
).fit(df)

df = pca_eval_all.transform(df)
print('Explained Variance Ratio', pca_eval_all.explainedVariance.toArray())


# In[19]:


# PCA (Pyspark - correlated)

pca_eval = PCA(
    k = 1, 
    inputCol = 'scaled_corr_features', 
    outputCol = 'pca_corr_features'
).fit(df)

# choosing the first component (k = 1) (component %: 1st: 0.99999)
df = pca_eval.transform(df)
print('Explained Variance Ratio', pca_eval.explainedVariance.toArray())


# ## Label (buy_or_sell) column 
# 

# In[20]:


df.createOrReplaceTempView("stock_data")
data = df



# Convert the 'date' column to a date (without time) and add it as a new column 'date_only'
data = data.withColumn('date_only', to_date(data['date']))

# Register the DataFrame as a temporary table
data.createOrReplaceTempView("stock_data")


# Create a new DataFrame with the 'buy_or_sell' column using Spark SQL
buy_sell_df = spark.sql("SELECT date, high, LAG(high) OVER (ORDER BY date) AS next_high FROM stock_data")
buy_sell_df = buy_sell_df.withColumn('buy_or_sell', when(buy_sell_df['next_high'] > buy_sell_df['high'], 1).otherwise(0))

# Join the buy_sell_df DataFrame with the original data DataFrame on the 'date' column
data = data.join(buy_sell_df.select('date', 'buy_or_sell'), on='date')

# Define features and label
features = ['close', 'high', 'low', 'open', 'volume', 'sma5', 'sma10', 'sma15', 'sma20',\
 'ema5', 'ema10', 'ema15', 'ema20', 'upperband', 'middleband', \
  'lowerband', 'HT_TRENDLINE', 'KAMA10', 'KAMA20', 'KAMA30', 'SAR', 'TRIMA5', 'TRIMA10', \
   'TRIMA20', 'ADX5', 'ADX10', 'ADX20', 'APO', 'CCI5', 'CCI10', 'CCI15', 'macd510', 'macd520', \
   'macd1020', 'macd1520' , 'macd1226' , 'MOM10' , 'MOM15' , 'MOM20' , 'ROC5' , \
 'ROC10' , 'ROC20' , 'PPO' , 'RSI14' , 'RSI8' , 'slowk' , 'slowd' , 'fastk' , \
'fastd' , 'fastksr' , 'fastdsr' , 'ULTOSC' , 'WILLR' , 'ATR' , 'Trange', 'TYPPRICE', 'HT_DCPERIOD', 'BETA'
]
label = 'buy_or_sell'

# Assemble features into a single vector
assembler = VectorAssembler(inputCols=features, outputCol='features')

# Define the classifier
rf = RandomForestClassifier(labelCol=label, featuresCol='features')

# Build the pipeline
pipeline = Pipeline(stages=[assembler, rf])

# Split data into training and test sets
train_data, test_data = data.randomSplit([0.7, 0.3])

# Fit the model
model = pipeline.fit(train_data)

# Make predictions on test data
predictions = model.transform(test_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol=label)
accuracy = evaluator.evaluate(predictions)
print(f'Accuracy: {accuracy}')
