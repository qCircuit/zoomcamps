import pickle
import mlflow
import pandas as pd
import uuid
import sys

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline


RUN_ID = "e8b684464b874c4699d1a86d628dc9f2"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def read_dataframe(filename):
    df = pd.read_parquet(filename)
    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    # lista dels identificadors unics
    ride_ids = [str(uuid.uuid4()) for i in range(len(df))]
    df["ride_id"] = ride_ids

    return df

def prepare_dictionaries(df):
    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]
    categorical = ["PU_DO"]
    
    numerical = ["trip_distance"]
    dicts = df[categorical + numerical].to_dict(orient="records")

    return dicts

def load_model(RUN_ID):
    model = f'runs:/{RUN_ID}/model'
    model = mlflow.pyfunc.load_model(model)

    return model

def apply_model(INPUT_FILE, OUTPUT_FILE):
    print(f"reading file from {INPUT_FILE}")
    df = read_dataframe(INPUT_FILE)
    dicts = prepare_dictionaries(df)

    print(f"loading file of the run: {RUN_ID}")
    model = load_model(RUN_ID)
    ypr = model.predict(dicts)


    df_result = pd.DataFrame()
    df_result["ride_id"] = df["ride_id"]
    df_result["lpep_pickup_datetime"] = df["lpep_pickup_datetime"]
    df_result["PULocationID"] = df["PULocationID"]
    df_result["DOLocationID"] = df["DOLocationID"]
    df_result["actual_duration"] = df["duration"]
    df_result["predicted_duration"] = ypr
    df_result["diff"] = df_result["actual_duration"] - df_result["predicted_duration"]
    df_result["model_version"] = RUN_ID
    
    print(f"writing file to {OUTPUT_FILE}")
    df_result.to_parquet(OUTPUT_FILE, index=False)

    return df_result

def run():

    TAXI_TYPE = sys.argv[1] #green

    INPUT_FILE = f"../../data/{TAXI_TYPE}_tripdata_2022-01.parquet"
    OUTPUT_FILE = f"output/{TAXI_TYPE}_output.parquet"
    
    apply_model(INPUT_FILE, OUTPUT_FILE)

if __name__ == "__main__":
    run()