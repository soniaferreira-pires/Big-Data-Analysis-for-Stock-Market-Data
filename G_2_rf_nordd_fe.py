import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

import sklearn
from sklearn.decomposition import PCA 

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, col, skewness, kurtosis, lead, when
from pyspark.sql.window import Window

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import to_date


spark = SparkSession.builder \
.config("spark.ui.port","4041") \
.config("spark.driver.memory","8g") \
.config("spark.executor.memory","4g") \
.config("spark.executor.instances", "12") \
.config("spark.default.parallelism", "2") \
.config("spark.executor.cores", "2") \
.master("yarn") \
.appName('Siemens_Stock_Market_Prediction') \
.getOrCreate()

spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

#.config(“spark.executor.memory”,”4g”) \

df = spark.read.option('header', 'true').csv("gs://bigdataprojectnn4/SIEMENS_minute_data_with_indicators.csv", sep = ',' , inferSchema = True)

# Convert the 'date' column to a date (without time) and add it as a new column 'date_only'
df = df.withColumn('date_only', to_date(df['date']))

# Register the DataFrame as a temporary table
df.createOrReplaceTempView("stock_data")

columns_list = df.columns

avg_close_df = spark.sql("SELECT date_only, AVG(close) AS avg_close, AVG(lag_close) AS avg_close_prev_day FROM (SELECT date_only,close, LAG(close) OVER (ORDER BY date_only) AS lag_close FROM stock_data) subquery GROUP BY date_only")

data = df.join(avg_close_df, on='date_only')

# Create a new DataFrame with the 'buy_or_sell' column using Spark SQL
buy_sell_df = spark.sql("SELECT date, high, LAG(high) OVER (ORDER BY date) AS next_high FROM stock_data")
buy_sell_df = buy_sell_df.withColumn('buy_or_sell', when(buy_sell_df['next_high'] > buy_sell_df['high'], 1).otherwise(0))

# Convert the DataFrame to an RDD
rdd = data.select('date_only', 'open').rdd
rdd = rdd.repartition(10)

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

columns_list = data.columns


#This one is not working yet

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

data = data.join(macd_df, on='date')

data = data.drop('macd510', 'macd520','macd1020', 'macd1520','macd1226','ema5', 'ema10', 'ema15', 'ema20')

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

# Join the avg_open_df DataFrame with the original data DataFrame on the 'date' column
data = data.join(bollinger_bands_width_df, on='date')

data = data.drop('upperband', 'lowerband', 'middleband')

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

data = data.join(com_channel_index_df, on='date')

data = data.drop('CCI5','CCI10','CCI15')

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

data = data.join(rsi_df, on='date')

data = data.drop('RSI14','RSI18', 'RSI8')

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

data = data.join(momentum_df, on='date_only')
data = data.drop('MOM20', 'MOM15', 'MOM10')

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

data = data.join(sma_df, on='date_only')
data = data.drop('sma5',  'sma10', 'sma15',  'sma20')

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

data = data.join(kama_df, on='date_only')
data = data.drop('KAMA10', 'KAMA20', 'KAMA30')

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

data = data.join(adx_df, on='date_only')
data = data.drop('ADX5', 'ADX10', 'ADX20')

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

data = data.join(roc_df, on='date_only')
data = data.drop('ROC5', 'ROC10','ROC20')

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

data = data.join(trima_df, on='date_only')
data = data.drop('TRIMA5', 'TRIMA10', 'TRIMA20')

# Define features and label
features = [
 'close',
 'high',
 'low',
 'open',
 'volume',
 'HT_TRENDLINE',
 'SAR',
 'APO',
 'MFI',
 'PPO',
 'slowk',
 'slowd',
 'fastk',
 'fastd',
 'fastksr',
 'fastdsr',
 'ULTOSC',
 'WILLR',
 'ATR',
 'Trange',
 'TYPPRICE',
 'HT_DCPERIOD',
 'BETA',
 'avg_close',
 'avg_close_prev_day',
 'avg_open',
 'macd',
 'bollinger_bands_width',
 'com_channel_index',
 'rsi',
 'momentum',
 'sma',
 'kama',
 'adx',
 'roc',
 'trima']
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
