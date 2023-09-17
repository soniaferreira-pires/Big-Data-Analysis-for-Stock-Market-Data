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
from pyspark.ml.feature import VectorAssembler, PCA, StandardScaler
from pyspark.mllib.util import MLUtils
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


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


# ## Feature normalization

# In[7]:


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

# In[8]:


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


# In[9]:


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


# In[14]:


features = ['close', 'high', 'low', 'open', 'volume', 'sma5', 'sma10', 'sma15', 'sma20', 'ema5', 'ema10', 'ema15', 'ema20', 'upperband', 'middleband', 'lowerband', 'HT_TRENDLINE', 'KAMA10', 'KAMA20',
            'KAMA30', 'SAR', 'TRIMA5', 'TRIMA10', 'TRIMA20', 'ADX5', 'ADX10', 'ADX20', 'APO', 'macd510', 'macd520', 'macd1020', 'macd1520', 'macd1226', 'MFI', 'MOM10', 'MOM15', 'MOM20', 'ROC5', 'ROC10', 'ROC20', 'PPO', 'RSI8',
            'slowk', 'slowd', 'fastk', 'fastd', 'fastksr', 'fastdsr', 'ULTOSC', 'WILLR', 'ATR', 'Trange', 'TYPPRICE', 'HT_DCPERIOD', 'BETA']
label = 'buy_or_sell'


# ## SVM

# In[15]:


data = data.drop('date')


# In[16]:


# data.columns


# In[17]:


data.show


# In[18]:


# print(data.printSchema()) 


# In[19]:


# Prepare the features for training by assembling them into a vector
# ( combine all feature data and separate 'label' data in a dataset, with VectorAssembler.)
assembler = VectorAssembler(inputCols= features, outputCol="features")


# In[20]:


va_df = assembler.transform(data)
va_df = va_df.select(['features', 'buy_or_sell'])
va_df.show(3)


# In[21]:


##data = data.limit(1000)


# In[22]:


# Split the data into training and testing sets
(train, test) = va_df.randomSplit([0.7, 0.3], seed=123)


# In[23]:


# Train the SVM model
svm = LinearSVC(maxIter=10, regParam=0.1, labelCol="buy_or_sell")
model = svm.fit(train)


# In[24]:


# Make predictions on the testing set
predictions = model.transform(test)
predictions.show(3)


# In[25]:


evaluator=MulticlassClassificationEvaluator(metricName="accuracy", labelCol="buy_or_sell")
acc = evaluator.evaluate(predictions)

y_pred=predictions.select("prediction").collect()
y_orig=predictions.select("buy_or_sell").collect()

cm = confusion_matrix(y_orig, y_pred)


# In[26]:


evaluator_roc = BinaryClassificationEvaluator(labelCol=label, metricName="areaUnderROC")
evaluator_acc = MulticlassClassificationEvaluator(labelCol=label, predictionCol='prediction', metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol=label, predictionCol='prediction', metricName="f1")

aucroc = evaluator_roc.evaluate(predictions)
f1 = evaluator_f1.evaluate(predictions)


# In[27]:


print('SVM PCA corr features summary stats')
print('__________________________________')
#print("Confusion Matrix:")
#print(cm)
print(f'Area under ROC: {aucroc}')
print(f'F1 (for label 1.0): {f1}')
print("Accuracy: ", acc)


# In[28]:


# Stop Spark session 
spark.stop()

