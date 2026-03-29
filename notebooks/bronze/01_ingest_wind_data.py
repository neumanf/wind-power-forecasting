# Databricks notebook source

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *

# COMMAND ----------

dbutils.widgets.text("catalog", "dev")
dbutils.widgets.text("bronze_schema", "bronze")
dbutils.widgets.text("dataset_path", "/Volumes/dev/bronze/files/Turbine_Data.csv")

# COMMAND ----------

dataset_path = dbutils.widgets.get("dataset_path")
catalog = dbutils.widgets.get("catalog")
bronze_schema = dbutils.widgets.get("bronze_schema")
wind_power_table = f'{catalog}.{bronze_schema}.wind_power'

# COMMAND ----------

spark.sql(f'CREATE SCHEMA IF NOT EXISTS {catalog}.{bronze_schema};')

# COMMAND ----------

schema = StructType([
    StructField("timestamp", TimestampType(), True),
    StructField("ActivePower", DoubleType(), True),
    StructField("AmbientTemperatue", DoubleType(), True),
    StructField("BearingShaftTemperature", DoubleType(), True),
    StructField("Blade1PitchAngle", DoubleType(), True),
    StructField("Blade2PitchAngle", DoubleType(), True),
    StructField("Blade3PitchAngle", DoubleType(), True),
    StructField("ControlBoxTemperature", DoubleType(), True),
    StructField("GearboxBearingTemperature", DoubleType(), True),
    StructField("GearboxOilTemperature", DoubleType(), True),
    StructField("GeneratorRPM", DoubleType(), True),
    StructField("GeneratorWinding1Temperature", DoubleType(), True),
    StructField("GeneratorWinding2Temperature", DoubleType(), True),
    StructField("HubTemperature", DoubleType(), True),
    StructField("MainBoxTemperature", DoubleType(), True),
    StructField("NacellePosition", DoubleType(), True),
    StructField("ReactivePower", DoubleType(), True),
    StructField("RotorRPM", DoubleType(), True),
    StructField("TurbineStatus", StringType(), True),
    StructField("WTG", StringType(), True),
    StructField("WindDirection", DoubleType(), True),
    StructField("WindSpeed", DoubleType(), True)
])

# COMMAND ----------

df_raw = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "false")
    .schema(schema)
    .csv(dataset_path)
)


df = df_raw.withColumnRenamed("_c0", "timestamp")

df_bronze = (
    df
    .withColumn("ingestion_time", current_timestamp())
)

# COMMAND ----------

(
    df_bronze.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(wind_power_table)
)

# COMMAND ----------

display(spark.table(wind_power_table))
