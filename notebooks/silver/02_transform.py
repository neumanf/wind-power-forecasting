# Databricks notebook source

from pyspark.sql.functions import *
from pyspark.sql.window import Window

# COMMAND ----------

dbutils.widgets.text("catalog", "dev")
dbutils.widgets.text("silver_schema", "silver")
dbutils.widgets.text("bronze_schema", "bronze")

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
bronze_schema = dbutils.widgets.get("bronze_schema")
silver_schema = dbutils.widgets.get("silver_schema")

bronze_table = f'{catalog}.{bronze_schema}.wind_power'
silver_table = f'{catalog}.{silver_schema}.wind_power'

# COMMAND ----------

spark.sql(f'CREATE SCHEMA IF NOT EXISTS {catalog}.{silver_schema};')

# COMMAND ----------

df = spark.table(bronze_table)

# COMMAND ----------

df = df.withColumnRenamed("AmbientTemperatue", "AmbientTemperature")

# COMMAND ----------

df = df.filter(col("timestamp").isNotNull())

# COMMAND ----------

df = df.dropna(subset=["WindSpeed", "ActivePower", "RotorRPM"])

# COMMAND ----------

df = df.filter(
    (col("WindSpeed") >= 0) &
    (col("WindSpeed") <= 40) &
    (col("ActivePower") >= 0) &
    (col("RotorRPM") >= 0)
)

# COMMAND ----------

df = df.filter(col("TurbineStatus").isNotNull())

# COMMAND ----------

window_spec = Window.partitionBy("timestamp", "WTG").orderBy(col("ingestion_time").desc())

df = (
    df
    .withColumn("row_num", row_number().over(window_spec))
    .filter(col("row_num") == 1)
    .drop("row_num")
)

# COMMAND ----------

df = (
    df
    .withColumn("year", year("timestamp"))
    .withColumn("month", month("timestamp"))
    .withColumn("day", dayofmonth("timestamp"))
    .withColumn("hour", hour("timestamp"))
)

# COMMAND ----------

df = df.fillna({
    "WindDirection": 0,
    "NacellePosition": 0
})

# COMMAND ----------

(
    df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(silver_table)
)

# COMMAND ----------

display(spark.table(silver_table))