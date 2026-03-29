# Databricks notebook source

# COMMAND ----------

import sys
sys.path.append('/Workspace/Users/fabricionewman@gmail.com/.bundle/wind_power_forecasting/dev/files')

# COMMAND ----------

import mlflow
import pandas as pd
from datetime import timedelta

import pyspark.sql.functions as F
from src.features.feature_engineering import select_features

# COMMAND ----------

dbutils.widgets.text("catalog", "dev")
dbutils.widgets.text("gold_schema", "gold")
dbutils.widgets.text("experiment_path", "/Shared/wind_power_forecasting")
dbutils.widgets.text("forecast_hours", "24")

catalog = dbutils.widgets.get("catalog")
gold_schema = dbutils.widgets.get("gold_schema")
experiment_path = dbutils.widgets.get("experiment_path")
forecast_hours = int(dbutils.widgets.get("forecast_hours"))

gold_table = f'{catalog}.{gold_schema}.wind_power'
output_table = f'{catalog}.{gold_schema}.wind_power_forecast'

# COMMAND ----------

client = mlflow.tracking.MlflowClient()

experiment = client.get_experiment_by_name(experiment_path)
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=1
)

run_id = runs[0].info.run_id
model_uri = f"runs:/{run_id}/model"

model = mlflow.sklearn.load_model(model_uri)

# COMMAND ----------

df = spark.table(gold_table)

latest_row = (
    df.orderBy(F.col("hour_timestamp").desc())
    .limit(1)
    .toPandas()
)

current = latest_row.copy()

# COMMAND ----------

features = [
    "avg_wind_speed",
    "avg_rotor_rpm",
    "avg_temp",
    "wind_speed_std",
    "hour",
    "day_of_week",
    "month",
    "lag_1h_wind",
    "lag_2h_wind",
    "lag_1h_power",
    "rolling_mean_wind_3h",
    "rolling_mean_power_3h"
]

# COMMAND ----------

def forecast_recursive(model, current_row, steps):

    forecasts = []
    history_power = [current_row.iloc[0]["lag_1h_power"]]

    current = current_row.copy()

    for step in range(steps):
        X = current.loc[:, features]

        pred = model.predict(X)[0]

        current_time = pd.to_datetime(current.iloc[0]["hour_timestamp"])
        next_time = current_time + pd.Timedelta(hours=1)

        new_row = current.copy()

        new_row.loc[0, "hour_timestamp"] = next_time
        new_row.loc[0, "hour"] = next_time.hour
        new_row.loc[0, "day_of_week"] = next_time.dayofweek + 1
        new_row.loc[0, "month"] = next_time.month

        new_row.loc[0, "lag_1h_power"] = pred
        new_row.loc[0, "lag_2h_wind"] = current.iloc[0]["lag_1h_wind"]
        new_row.loc[0, "lag_1h_wind"] = current.iloc[0]["avg_wind_speed"]

        history_power.append(pred)
        new_row.loc[0, "rolling_mean_power_3h"] = sum(history_power[-3:]) / min(len(history_power), 3)

        forecasts.append({
            "timestamp": next_time,
            "predicted_power": pred
        })

        current = new_row

    return pd.DataFrame(forecasts)

# COMMAND ----------

forecast_pdf = forecast_recursive(model, current, forecast_hours)

# COMMAND ----------

forecast_df = spark.createDataFrame(forecast_pdf)

forecast_df = forecast_df.withColumn("generated_at", F.current_timestamp())

# COMMAND ----------

(
    forecast_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(output_table)
)

# COMMAND ----------

display(spark.table(output_table))
