import mlflow
import numpy as np
import pathlib
import pickle
import pandas as pd
import scipy
import xgboost as xgb
from datetime import date
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error

from prefect import flow, task
from prefect_aws import S3Bucket
from prefect.artifacts import create_markdown_artifact

DATA_PATH = "../../data/"
DATASET = "green"
TARGET = "duration"


@task(retries=3, retry_delay_seconds=10)
def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

@task
def add_features(dftr, dfvl):
    dftr['PU_DO'] = dftr['PULocationID'] + '_' + dftr['DOLocationID']
    dfvl['PU_DO'] = dfvl['PULocationID'] + '_' + dfvl['DOLocationID']

    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dv = DictVectorizer()

    train_dicts = dftr[categorical + numerical].to_dict(orient='records')
    val_dicts = dfvl[categorical + numerical].to_dict(orient='records')

    xtr = dv.fit_transform(train_dicts)
    xvl = dv.transform(val_dicts)

    ytr = dftr[TARGET].values
    yvl = dfvl[TARGET].values

    return xtr, xvl, ytr, yvl, dv

@task(log_prints=True)
def train_best_model(xtr, xvl, ytr, yvl, dv):
    with mlflow.start_run():
        train = xgb.DMatrix(xtr, label=ytr)
        valid = xgb.DMatrix(xvl, label=yvl)

        best_params = {
            "learning_rate": 0.09585355369315604,
            "max_depth": 30,
            "min_child_weight": 1.060597050922164,
            "objective": "reg:linear",
            "reg_alpha": 0.018060244040060163,
            "reg_lambda": 0.011658731377413597,
            "seed": 42,
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, "valid")],
            early_stopping_rounds=50
        )
        ypr = booster.predict(valid)
        rmse = mean_squared_error(yvl, ypr, squared=False)
        mlflow.log_metric("rmse", rmse)

        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
        
        markdown__rmse_report = f"""# RMSE Report
        ## Summary

        Duration Prediction 

        ## RMSE XGBoost Model

        | Region    | RMSE |
        |:----------|-------:|
        | {date.today()} | {rmse:.2f} |
        """

        create_markdown_artifact(
            key="duration-model-report", 
            markdown=markdown__rmse_report
        )

    return None

@flow
def main_flow_s3():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment")

    if False:
        s3_bucket_block = S3Bucket.load("zoomcamps_bucket")
        s3_bucket_block.download_folder_to_path(
            from_folder = "data",
            to_folder = "date"
        )

    if True:
        dftr = read_dataframe(f"{DATA_PATH}{DATASET}_tripdata_2022-01.parquet")
        dfvl = read_dataframe(f"{DATA_PATH}{DATASET}_tripdata_2022-02.parquet")

    xtr, xvl, ytr, yvl, dv = add_features(dftr, dfvl)
    train_best_model(xtr, xvl, ytr, yvl, dv)

if __name__ == "__main__":
    main_flow_s3()

