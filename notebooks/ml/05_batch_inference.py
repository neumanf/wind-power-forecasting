# Databricks notebook source

# COMMAND ----------

import sys
sys.path.append('/Workspace/Users/fabricionewman@gmail.com/.bundle/wind_power_forecasting/dev/files')

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd

from src.features.feature_engineering import select_features
from pyspark.sql.functions import *

# COMMAND ----------

dbutils.widgets.text("catalog", "dev")
dbutils.widgets.text("gold_schema", "gold")
dbutils.widgets.text("output_table", "dev.gold.wind_power_predictions")
dbutils.widgets.text("experiment_path", "/Shared/wind_power_forecasting")

catalog = dbutils.widgets.get("catalog")
gold_schema = dbutils.widgets.get("gold_schema")
experiment_path = dbutils.widgets.get("experiment_path")

gold_table = f'{catalog}.{gold_schema}.wind_power'
output_table = f'{catalog}.{gold_schema}.wind_power_predictions'

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

client = mlflow.tracking.MlflowClient()

experiment = client.get_experiment_by_name(experiment_path)
experiment_id = experiment.experiment_id

runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=1
)

run_id = runs[0].info.run_id

model_uri = f"runs:/{run_id}/model"

# COMMAND ----------

model = mlflow.sklearn.load_model(model_uri)

# COMMAND ----------

df = spark.table(gold_table)

pdf = df.toPandas()

# COMMAND ----------

features, X, y = select_features(pdf)

# COMMAND ----------

pdf["predicted_power"] = model.predict(X)

# COMMAND ----------

pdf["prediction_time"] = pd.Timestamp.now()
pdf["model_run_id"] = run_id

# COMMAND ----------

pred_df = spark.createDataFrame(pdf)

# COMMAND ----------

(
    pred_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(output_table)
)

# COMMAND ----------

display(spark.table(output_table))
