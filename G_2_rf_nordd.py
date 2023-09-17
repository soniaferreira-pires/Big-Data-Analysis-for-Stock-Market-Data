from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import to_date
from pyspark.sql import SparkSession
from pyspark.sql.functions import when


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

# Load data
data = spark.read.csv('gs://bigdataprojectnn4/SIEMENS_minute_data_with_indicators.csv', header=True, inferSchema=True)



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