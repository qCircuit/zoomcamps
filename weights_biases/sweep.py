import click
import os
import pickle
import wandb

from functools import partial
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def run_train(data_artifact):
    wandb.init()
    config = wandb.config

    artifact = wandb.use_artifact(data_artifact, type="preprocessed_dataset")
    data_path = artifact.download()

    xtr, ytr = load_pickle(os.path.join(data_path, "train.pkl"))
    xvl, yvl = load_pickle(os.path.join(data_path, "val.pkl"))

    rf = RandomForestRegressor(
        max_depth=config.max_depth,
        random_state=0
    )
    rf.fit(xtr, ytr)
    ypr = rf.predict(xvl)

    mse = mean_squared_error(yvl, ypr, squared=False)
    wandb.log({"MSE": mse})

    with open("regressor.pkl", "wb") as f_out:
        pickle.dump(rf, f_out)

    artifact = wandb.Artifact(f"{wandb.run.id}-model", type="model")
    artifact.add_file("regressor.pkl")
    wandb.log_artifact(artifact)

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {
        "name": "MSE",
        "goal": "minimize"
    },
    "parameters": {
        "max_depth": {
            "distribution": "int_uniform",
            "min": 1,
            "max": 20,
    },
        "n_estimators": {
            "distribution": "int_uniform",
            "min": 10,
            "max": 50,
        },
        "min_samples_split": {
            "distribution": "int_uniform",
            "min": 2,
            "max": 10,
        },
        "min_samples_leaf": {
            "distribution": "int_uniform",
            "min": 1,
            "max": 4,
        }    
    }
}

@click.command()
@click.option("--project", help="el nombre de proyecto")
@click.option("--entity", help="el nombre de entidad")
@click.option("--data_artifact", help="el nombre del artifacto de datos")
@click.option("--count", default=5, help="el numero de ejecuciones")
def run_sweep(project, entity, data_artifact, count):
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=project, entity=entity)
    run = wandb.agent(sweep_id, partial(run_train, data_artifact), count=count)

if __name__ == "__main__":
    run_sweep()