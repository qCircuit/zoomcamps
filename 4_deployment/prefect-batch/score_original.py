import mlflow
import pandas as pd
import pickle
import os
import sys
import uuid

from datetime import datetime
from dateutil.relativedelta import relativedelta
from prefect import task, flow, get_run_logger
from prefect.context import get_run_context

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline


def generate_uuids(n):
    return [str(uuid.uuid4()) for i in range(n)]

def read_dataframe(filename):
    df = pd.read_parquet(filename)
    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df["ride_id"] = generate_uuids(len(df))
    
    return df

def prepare_dictionaries(df):
    categorical = ["PULocationID", "DOLocationID"]
    numerical = ["trip_distance"]
    df[categorical] = df[categorical].astype(str)
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]
    categorical = ["PU_DO"]
    dicts = df[categorical + numerical].to_dict(orient="records")
    return dicts

def load_model(RUN_ID):
    model = f'runs:/{RUN_ID}/model'
    model = mlflow.pyfunc.load_model(model)

    return model

def save_results(df, ypr, RUN_ID, output_file):
    df_result = pd.DataFrame()
    df_result["ride_id"] = df["ride_id"]
    df_result["lpep_pickup_datetime"] = df["lpep_pickup_datetime"]
    df_result["PULocationID"] = df["PULocationID"]
    df_result["DOLocationID"] = df["DOLocationID"]
    df_result["actual_duration"] = df["duration"]
    df_result["predicted_duration"] = ypr
    df_result["diff"] = df_result["actual_duration"] - df_result["predicted_duration"]
    df_result["model_version"] = RUN_ID
    
    print(f"writing file to {output_file}")
    df_result.to_parquet(output_file, index=False)

@task
def apply_model(input_file, RUN_ID, output_file):
    logger = get_run_logger()

    logger.info(f"reading file {input_file}")
    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)

    logger.info(f"loading model {RUN_ID}")
    model = load_model(RUN_ID)

    logger.info(f"applying model...")
    ypr = model.predict(dicts)

    logger.info(f"saving results to {output_file}")
    save_results(df, ypr, RUN_ID, output_file)

def get_paths(run_date, TAXI_TYPE, RUN_ID):
    prev_month = run_date - relativedelta(months=1)
    year = prev_month.year
    month = prev_month.month
    
    input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{TAXI_TYPE}_tripdata_{year:04d}-{month:02d}.parquet"
    output_file = f"output/{TAXI_TYPE}_taxi.parquet"

    return input_file, output_file

@flow
def ride_duration_prediction(TAXI_TYPE, RUN_ID, run_date=None):
    if run_date is None:
        ctx = get_run_context()
        run_date = ctx.flow_run.expected_start_time

    input_file, output_file = get_paths(run_date, TAXI_TYPE, RUN_ID)
    apply_model(input_file, RUN_ID, output_file)

def run():
    TAXI_TYPE = sys.argv[1]
    year = int(sys.argv[2])
    month = int(sys.argv[3])
    RUN_ID = sys.argv[4] # ac7d9665277c494da7929f136c28faf9

    ride_duration_prediction(
        TAXI_TYPE, RUN_ID, 
        run_date = datetime(year=year, month=month, day=1)
    )

if __name__ == "__main__":
    run()