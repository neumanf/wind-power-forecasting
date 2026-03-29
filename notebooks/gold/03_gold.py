# Databricks notebook source

from pyspark.sql.functions import *
from pyspark.sql.window import Window

# COMMAND ----------

dbutils.widgets.text("catalog", "dev")
dbutils.widgets.text("silver_schema", "silver")
dbutils.widgets.text("gold_schema", "gold")


# COMMAND ----------


catalog = dbutils.widgets.get("catalog")
silver_schema = dbutils.widgets.get("silver_schema")
gold_schema = dbutils.widgets.get("gold_schema")

silver_table = f'{catalog}.{silver_schema}.wind_power'
gold_table = f'{catalog}.{gold_schema}.wind_power'

# COMMAND ----------

spark.sql(f'CREATE SCHEMA IF NOT EXISTS {catalog}.{gold_schema};')

# COMMAND ----------

df = spark.table(silver_table)

# COMMAND ----------

df_hourly = (
    df
    .withColumn("hour_timestamp", date_trunc("hour", col("timestamp")))
    .groupBy("WTG", "hour_timestamp")
    .agg(
        avg("WindSpeed").alias("avg_wind_speed"),
        avg("ActivePower").alias("avg_power"),
        avg("RotorRPM").alias("avg_rotor_rpm"),
        avg("AmbientTemperature").alias("avg_temp"),
        stddev("WindSpeed").alias("wind_speed_std"),
        max("WindSpeed").alias("max_wind_speed")
    )
)

# COMMAND ----------

df_features = (
    df_hourly
    .withColumn("hour", hour("hour_timestamp"))
    .withColumn("day_of_week", dayofweek("hour_timestamp"))
    .withColumn("month", month("hour_timestamp"))
)

# COMMAND ----------

window_spec = Window.partitionBy("WTG").orderBy("hour_timestamp")

df_features = (
    df_features
    .withColumn("lag_1h_wind", lag("avg_wind_speed", 1).over(window_spec))
    .withColumn("lag_2h_wind", lag("avg_wind_speed", 2).over(window_spec))
    .withColumn("lag_1h_power", lag("avg_power", 1).over(window_spec))
)

# COMMAND ----------

rolling_window = Window.partitionBy("WTG").orderBy("hour_timestamp").rowsBetween(-3, 0)

df_features = (
    df_features
    .withColumn("rolling_mean_wind_3h", avg("avg_wind_speed").over(rolling_window))
    .withColumn("rolling_mean_power_3h", avg("avg_power").over(rolling_window))
)

# COMMAND ----------

df_gold = df_features.dropna()

# COMMAND ----------

df_gold = df_gold.withColumnRenamed("avg_power", "target_power")

# COMMAND ----------

(
    df_gold.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(gold_table)
)

# COMMAND ----------

display(spark.table(gold_table))