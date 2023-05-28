import click
import pandas as pd
import pickle
import os
from sklearn.feature_extraction import DictVectorizer

import configura

def dump_pickle(obj, filename: str):
    with open(filename, 'wb') as f_out:
        pickle.dump(obj, f_out)

def read_df(filename: str):
    df = pd.read_parquet(filename)
    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df

def preprocess(df: pd.DataFrame, dv:DictVectorizer, fit_dv: bool = False):
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ["PU_DO"]
    numerical = ["trip_distance"]
    dicts = df[categorical+numerical].to_dict(orient='records')
    
    if fit_dv:
        x = dv.fit_transform(dicts)
    else:
        x = dv.transform(dicts)

    return x, dv

@click.command()
@click.option(
    "--raw_data_path",
    help="Location where the raw NYC taxi trip data was saved",
    default=configura.raw_data_path
)
@click.option(
    "--dest_path",
    help="Location where the preprocessed NYC taxi trip data should be saved"
)

def run_data_prep(raw_data_path: str, dest_path: str, dataset: str = "green"):
    df_train = read_df(
        os.path.join(raw_data_path, f"{dataset}_tripdata_2022-01.parquet")
    )
    df_val = read_df(
        os.path.join(raw_data_path, f"{dataset}_tripdata_2022-02.parquet")
    )
    df_test = read_df(
        os.path.join(raw_data_path, f"{dataset}_tripdata_2022-03.parquet")
    )

    target = 'tip_amount'
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values

    dv = DictVectorizer()
    xtr, dv = preprocess(df_train, dv, fit_dv=True)
    xvl, _ = preprocess(df_val, dv, fit_dv=False)
    xts, _ = preprocess(df_test, dv, fit_dv=False)

    os.makedirs(configura.dest_path, exist_ok=True)
    dump_pickle((xtr, y_train), os.path.join(configura.dest_path, "train.pkl"))

    dump_pickle(dv, os.path.join(configura.dest_path, "dv.pkl"))
    dump_pickle((xtr, y_train), os.path.join(configura.dest_path, "train.pkl"))
    dump_pickle((xvl, y_val), os.path.join(configura.dest_path, "val.pkl"))
    dump_pickle((xts, y_test), os.path.join(configura.dest_path, "test.pkl"))

if __name__ == '__main__':
    run_data_prep()