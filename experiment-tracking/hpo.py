import os
import pickle
import click
import mlflow
import optuna

from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import configura

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("random-forest-hyperopt")

def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
    
@click.command()
@click.option(
    "--dest-path", 
    default=configura.dest_path, 
    help="the location of the processed data file"
)
@click.option(
    "--num_trials",
    default=10,
     help="The number of parameter evaluations for the optimizer to explore"
)

def run_optimization(dest_path: str, num_trials: int):
    xtr, ytr = load_pickle(os.path.join(dest_path, "train.pkl"))
    xvl, yvl = load_pickle(os.path.join(dest_path, "val.pkl"))

    def objective(trial):
        with mlflow.start_run():
            with mlflow.start_run(nested=True):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 10, 50, 1),
                    'max_depth': trial.suggest_int('max_depth', 1, 20, 1),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, 1),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4, 1),
                    'random_state': 42,
                    'n_jobs': -1
                }

                rf = RandomForestRegressor(**params)
                rf.fit(xtr, ytr)
                y_pred = rf.predict(xvl)
                rmse = mean_squared_error(yvl, y_pred, squared=False)

                for key, value in params.items():
                    mlflow.log_param(key, value)
                mlflow.log_metric("rmse", rmse)

        return rmse
    

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)
        

if __name__ == '__main__':
    run_optimization()
