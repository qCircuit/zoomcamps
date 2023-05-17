import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import pickle
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from prefect import flow, task

def read_data(filename):
    df = pd.read_parquet(filename)
    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds()
    df = df[df["duration"] < 60*60*12].reset_index(drop=True)
    df = df[df["duration"] > 0].reset_index(drop=True)

    categoricals = ["PULocationID", "DOLocationID"]
    numericals = ["trip_distance"]

    df[categoricals] = df[categoricals].astype(str)

    x = df[categoricals + numericals].values
    y = df["duration"].values
 
    return x, y

@task
def add_features():
    xtr, ytr = read_data("../duration_prediction/data/yellow_tripdata_2022-01.parquet")
    xts, yts = read_data("../duration_prediction/data/yellow_tripdata_2023-01.parquet")
    xtr, xvl, ytr, yvl = train_test_split(xtr, ytr, test_size=0.2, random_state=42)
    print(xtr.shape, xvl.shape, ytr.shape, yvl.shape, xts.shape, yts.shape)

    return xtr, xvl, ytr, yvl, xts, yts

@task
def fit_model(xtr, xvl, ytr, yvl, xts, yts):
    modelName = "xgboossRegressor"
    with mlflow.start_run() as run:
        
        regressor = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric = mean_squared_error
        )

        regressor.fit(xtr, ytr, eval_set=[(xvl, yvl)])
        ypr = regressor.predict(xts)
        rmse = mean_squared_error(yts, ypr, squared=False)

        mlflow.xgboost.log_model(regressor, artifact_path="xgb_model")
        mlflow.xgboost.autolog()

        print(rmse,"model fit completed")

    return regressor

@flow
def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("duration_prediction")
    xtr, xvl, ytr, yvl, xts, yts = add_features()
    model = fit_model(xtr, xvl, ytr, yvl, xts, yts)

main() 