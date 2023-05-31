import click
import os
import pickle

import wandb

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
    

@click.command()
@click.option("--project", help="el nombre de proyecto")
@click.option("--entity", help="el nombre de entidad")
@click.option("--data_artifact", help="el nombre del artifacto de datos")
@click.option("--random_state", default=0)
@click.option("--max_depth", default=10)
def run_train(
    project,
    entity,
    data_artifact,
    random_state,
    max_depth
):
    wandb.init(
        project=project,
        entity=entity,
        job_type="train",
        config={"max_depth": max_depth, "random_state": random_state}      
    )

    SWEEP_MODE = True

    artifact = wandb.use_artifact(data_artifact, type="preprocessed_dataset")
    data_path = artifact.download()

    xtr, ytr = load_pickle(os.path.join(data_path, "train.pkl"))
    xvl, yvl = load_pickle(os.path.join(data_path, "val.pkl"))

    if SWEEP_MODE:
        config = wandb.config

    if SWEEP_MODE:
        rf = RandomForestRegressor(
            max_depth=config.max_depth, 
            random_state=config.random_state
        )
    else:
        rf = RandomForestRegressor(
            max_depth=max_depth, 
            random_state=random_state
        )
    rf.fit(xtr, ytr)
    ypr = rf.predict(xvl)
    wandb.config = rf.get_params()

    mse = mean_squared_error(yvl, ypr, squared=False)
    wandb.log({"MSE": mse})

    with open("regressor.pkl", "wb") as f_out:
        pickle.dump(rf, f_out)
    artifact = wandb.Artifact("regressor", type="model")
    artifact.add_file("regressor.pkl")
    wandb.log_artifact(artifact)

if __name__ == "__main__":
    run_train()