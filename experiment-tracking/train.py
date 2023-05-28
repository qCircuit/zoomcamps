import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import configura

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("experiment-tracking")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def dump_pickle(obj, filename: str):
    with open(filename, 'wb') as f_out:
        pickle.dump(obj, f_out)

@click.command()
@click.option(
    "--dest_path",
    default=configura.dest_path,
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(dest_path: str):
    with mlflow.start_run():
        xtr, ytr = load_pickle(os.path.join(dest_path, "train.pkl"))
        xvl, yvl = load_pickle(os.path.join(dest_path, "val.pkl"))

        mlflow.autolog()

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(xtr, ytr)
        ypr = rf.predict(xvl)

        dump_pickle(rf, "output/model.pkl")

        rmse = mean_squared_error(yvl, ypr, squared=False)

        mlflow.set_tag("model", "random forest")
        mlflow.sklearn.log_model(rf, "output/model.pkl")

if __name__ == '__main__':
    
        run_train()
        