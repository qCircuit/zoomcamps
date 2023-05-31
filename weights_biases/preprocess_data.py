import click
import os
import pandas as pd
import pickle

import wandb

from sklearn.feature_extraction import DictVectorizer

def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)
    
def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda v: v.total_seconds() / 60)

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df

def preprocess(df, dv, fit_dv=False):
    df["PU_DO"] = df.PULocationID + "_" + df.DOLocationID
    categorical = ["PU_DO"]
    numerical = ["trip_distance"]
    dicts = df[categorical + numerical].to_dict(orient="records")
    if fit_dv:
        x = dv.fit_transform(dicts)
    else:
        x = dv.transform(dicts)

    return x, dv

@click.command()
@click.option("--project", help="el nombre de proyecto")
@click.option("--entity", help="el nombre de entidad")
@click.option("--data_path", help="el path donde se encuentra el dataset")
@click.option("--dest_path", help="el path donde se guardan los archivos")
def run_data_prep(
    project: str,
    entity: str,
    data_path: str,
    dest_path: str,
    dataset: str = "green",
):
    wandb.init(project=project, entity=entity, job_type="preprocess")

    dftr = read_dataframe(
        os.path.join(data_path, f"{dataset}_tripdata_2022-01.parquet")
    )
    dfvl = read_dataframe(
        os.path.join(data_path, f"{dataset}_tripdata_2022-02.parquet")
    )
    dfts = read_dataframe(
        os.path.join(data_path, f"{dataset}_tripdata_2022-03.parquet")
    )

    target = "tip_amount"
    ytr = dftr[target].values
    yvl = dfvl[target].values
    yts = dfts[target].values

    dv = DictVectorizer()
    xtr, dv = preprocess(dftr, dv, True)
    xvl, _ = preprocess(dfvl, dv, False)
    xts, _ = preprocess(dfts, dv, False)

    os.makedirs(dest_path, exist_ok=True)
    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((xtr, ytr), os.path.join(dest_path, "train.pkl"))
    dump_pickle((xvl, yvl), os.path.join(dest_path, "val.pkl"))
    dump_pickle((xts, yts), os.path.join(dest_path, "test.pkl"))

    artifact = wandb.Artifact("Taxi", type="preprocessed_dataset")
    artifact.add_dir(dest_path)
    wandb.log_artifact(artifact)

if __name__ == "__main__":
    run_data_prep()