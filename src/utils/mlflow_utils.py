import mlflow

def log_model(model, params, metrics, model_name):
    for k, v in params.items():
        mlflow.log_param(k, v)

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)