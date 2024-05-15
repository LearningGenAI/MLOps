import os
import yaml
import pandas as pd
import numpy as np
import argparse
from get_data import read_params
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
import joblib
import json
import mlflow
from urllib.parse import urlparse
from mlflow.tracking import MlflowClient
from pprint import pprint


def best_prod_model(config_path):
    config = read_params(config_path)
    mlflow_config = config["mlflow_config"]  # Corrected this line
    model_name = mlflow_config["registered_model_name"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    mlflow.set_tracking_uri(remote_server_uri)

    run = mlflow.search_runs(experiment_ids="3")
    lowest = run["metrics.rmse"].sort_values(ascending=True).iloc[0]
    lowest_run_id = run[run["metrics.rmse"] == lowest]["run_id"].iloc[0]

    client = MlflowClient()
    logged_model = None

    for mv in client.search_model_versions(f"name='{model_name}'"):  # Fixed the missing quote
        mv = dict(mv)
        if mv["run_id"] == lowest_run_id:
            current_version = mv["version"]
            logged_model = mv["source"]
            pprint(mv, indent=4)

            client.transition_model_version_stage(name=model_name, version=current_version, stage="Production")
        else:
            current_version = mv["version"]  # Fixed the typo in "version"
            client.transition_model_version_stage(name=model_name, version=current_version, stage="Staging")

    if logged_model is not None:
        loaded_model = mlflow.pyfunc.load_model(logged_model)
        model_dir = config["model_dirs"]["model_dir"]  # Fixed the key access
        joblib.dump(loaded_model, model_dir)
    else:
        print("No logged model found.")


if __name__ == "main":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    best_prod_model(config_path=parsed_args.config)