# Databricks notebook source

# COMMAND ----------

import sys
sys.path.append('/Workspace/Users/fabricionewman@gmail.com/.bundle/wind_power_forecasting/dev/files')

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models.signature import infer_signature
from src.models.train import train_model
from src.models.predict import predict
from src.models.evaluate import evaluate
from src.features.feature_engineering import select_features

# COMMAND ----------

dbutils.widgets.text("catalog", "dev")
dbutils.widgets.text("gold_schema", "gold")
dbutils.widgets.text("model_schema", "ml")
dbutils.widgets.text("model_name", "wind_power_model")

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
gold_schema = dbutils.widgets.get("gold_schema")
model_schema = dbutils.widgets.get("model_schema")
model_name = dbutils.widgets.get("model_name")

gold_table = f'{catalog}.{gold_schema}.wind_power'

# COMMAND ----------

spark.sql(f'CREATE SCHEMA IF NOT EXISTS {catalog}.{model_schema};')

# COMMAND ----------

df = spark.table(gold_table).toPandas()

# COMMAND ----------

features, X, y = select_features(df)

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

# COMMAND ----------

params = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
}

mlflow.tracking._model_registry.utils._get_registry_uri_from_spark_session = lambda: "databricks"
mlflow.set_experiment("/Shared/wind_power_forecasting")

with mlflow.start_run():
    model = train_model(X_train, y_train, params)

    preds = predict(model, X_test)

    metrics = evaluate(y_test, preds)

    mlflow.log_params(params)

    mlflow.log_metric("mae", metrics["mae"])
    mlflow.log_metric("r2", metrics["r2"])

    mlflow.log_param("features", ",".join(features))

    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X_train.head(5)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )

