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


df = spark.read.option('header', 'true').csv("gs://bigdata_project_ct/SIEMENS_minute_data_with_indicators.csv", sep = ',' , inferSchema = True)
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


# In[22]:


# Create a new DataFrame with the 'buy_or_sell' column using Spark SQL
buy_sell_df = spark.sql("SELECT date, high, LAG(high) OVER (ORDER BY date) AS next_high FROM stock_data")
buy_sell_df = buy_sell_df.withColumn('buy_or_sell', when(buy_sell_df['next_high'] > buy_sell_df['high'], 1).otherwise(0))


# In[23]:


data = data.join(buy_sell_df.select('date', 'buy_or_sell'), on='date')


# In[24]:


print(data.columns)


# In[40]:


data = data.drop('date', 'close', 'high', 'low', 'open', 'volume', 'sma5', 'sma10', 'sma15', 'sma20', 'ema5', 'ema10', 'ema15', 'ema20', 'upperband', 'middleband', 
                 'lowerband', 'HT_TRENDLINE', 'KAMA10', 'KAMA20', 'KAMA30', 'SAR', 'TRIMA5', 'TRIMA10', 'TRIMA20', 'ADX5', 'ADX10', 'ADX20', 'APO', 'CCI5', 'CCI10', 
                 'CCI15', 'macd510', 'macd520', 'macd1020', 'macd1520', 'macd1226', 'MFI', 'MOM10', 'MOM15', 'MOM20', 'ROC5', 'ROC10', 'ROC20', 'PPO', 'RSI14', 'RSI8', 
                 'slowk', 'slowd', 'fastk', 'fastd', 'fastksr', 'fastdsr', 'ULTOSC', 'WILLR', 'ATR', 'Trange', 'TYPPRICE', 'HT_DCPERIOD', 'BETA','all_features', 'scaled_all_features', 'corr_features', 'scaled_corr_features', 
                 'non_corr_features', 'scaled_non_corr_features' )


# ## Random forest with RDD

# In[36]:


data.first()


# In[42]:


data_all = data.drop('pca_corr_features')


# In[43]:


data_corr = data.drop('pca_all_features')


# In[44]:


data_all.show()


# ### Split data

# In[93]:


# Split data_all into training and test sets
train_data_all, test_data_all = data_all.randomSplit([0.7, 0.3], seed=42)


# In[47]:


# Split data_corr into training and test sets
train_data_corr, test_data_corr = data_corr.randomSplit([0.7, 0.3], seed=42)


# ### Convert train test data to RDD

# In[94]:


# Convert data_all to RDDs
train_all_rdd = MLUtils.convertVectorColumnsToML(train_data_all).rdd
train_all_rdd = train_all_rdd.repartition(10)
test_all_rdd = MLUtils.convertVectorColumnsToML(test_data_all).rdd
test_all_rdd = test_all_rdd.repartition(10)


# In[52]:


# Convert data_corr to RDDs
train_corr_rdd = MLUtils.convertVectorColumnsToML(train_data_corr).rdd
train_corr_rdd = train_corr_rdd.repartition(10)
test_corr_rdd = MLUtils.convertVectorColumnsToML(test_data_corr).rdd
test_corr_rdd = test_corr_rdd.repartition(10)


# In[143]:


data_corr.show()


# ### Creating the labeled point for the model

# ### data_all

# In[95]:


train_all_rdd.first()


# In[96]:


header_all = train_all_rdd.first()
label_index_all = header_all.__fields__.index('buy_or_sell')


# In[97]:


label_index_all


# In[116]:


train_all_rdd.first()[0].toArray().tolist()


# In[117]:


labeled_points_train_all_rdd = train_all_rdd.map(lambda row: LabeledPoint(row[label_index_all], row[0].toArray().tolist()))


# In[119]:


labeled_points_train_all_rdd.first()


# ### data corr

# In[144]:


train_corr_rdd.first()


# In[145]:


header_corr = train_corr_rdd.first()
label_index_corr = header_corr.__fields__.index('buy_or_sell')


# In[146]:


label_index_corr


# In[147]:


labeled_points_train_corr_rdd = train_corr_rdd.map(lambda row: LabeledPoint(row[label_index_corr], row[0].toArray().tolist()))


# In[148]:


labeled_points_train_corr_rdd.first()


# ### Train and test model 

# ### data_all

# In[120]:


model_all = RandomForest.trainClassifier(labeled_points_train_all_rdd, numClasses=2, categoricalFeaturesInfo={}, numTrees=10, 
                                         featureSubsetStrategy="auto", impurity="gini", maxDepth=10, maxBins=32, seed=42)   


# In[121]:


header_all = test_all_rdd.first()
label_index_all = header_all.__fields__.index('buy_or_sell')


# In[125]:


test_all_rdd.first


# In[126]:


labeled_points_test_all_rdd = test_all_rdd.map(lambda row: LabeledPoint(row[label_index_all], row[0].toArray().tolist()))


# In[127]:


# Make predictions on test data_all
predictions_all = model_all.predict(labeled_points_test_all_rdd.map(lambda x: x.features))


# In[128]:


predictions_all.first()


# In[62]:


# predictions.first()


# ### data_corr

# In[149]:


model_corr = RandomForest.trainClassifier(labeled_points_train_corr_rdd, numClasses=2, categoricalFeaturesInfo={}, numTrees=10, 
                                         featureSubsetStrategy="auto", impurity="gini", maxDepth=10, maxBins=32, seed=42)   


# In[150]:


header_corr = test_corr_rdd.first()
label_index_corr = header_corr.__fields__.index('buy_or_sell')


# In[151]:


labeled_points_test_corr_rdd = test_corr_rdd.map(lambda row: LabeledPoint(row[label_index_corr], row[0].tolist()))


# In[152]:


# Make predictions on test data_corr
predictions_corr = model_corr.predict(labeled_points_test_corr_rdd.map(lambda x: x.features))


# In[159]:


print(predictions_corr.take(100))


# ### Metrics

# ### data_all

# In[129]:


# Compute the area under the ROC curve for data_all
labels_and_predictions_all = labeled_points_test_all_rdd.map(lambda lp: lp.label).zip(predictions_all)
metrics_all = BinaryClassificationMetrics(labels_and_predictions_all)
metrics_2_all = MulticlassMetrics(labels_and_predictions_all)
area_under_roc_all = metrics_all.areaUnderROC
accuracy_all = metrics_2_all.accuracy
f_1_all = metrics_2_all.fMeasure(1.0)


# In[130]:


print('Random Forest RDD PCA all features summary stats')
print('__________________________________')
print(f'Area under ROC: {area_under_roc_all}')
print(f'F1 (for label 1.0): {f_1_all}')
print(f'Accuracy: {accuracy_all}')
      


# ### data_corr

# In[160]:


# Compute the area under the ROC curve for data_corr
labels_and_predictions_corr = labeled_points_test_corr_rdd.map(lambda lp: lp.label).zip(predictions_corr)
metrics_corr = BinaryClassificationMetrics(labels_and_predictions_corr)
metrics_2_corr = MulticlassMetrics(labels_and_predictions_corr)
area_under_roc_corr = metrics_corr.areaUnderROC
accuracy_corr = metrics_2_corr.accuracy
f_1_corr = metrics_2_corr.fMeasure(0.0)


# In[142]:


print('Random Forest RDD PCA corr features summary stats')
print('__________________________________')
print(f'Area under ROC: {area_under_roc_corr}')
print(f'F1 (for label 0.0): {f_1_corr}')
print(f'Accuracy: {accuracy_corr}')

