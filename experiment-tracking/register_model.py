import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

import configura

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state', 'n_jobs']

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("random-forest-hyperopt")
mlflow.sklearn.autolog()

def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
    
def train_and_log_model(data_path, params):
    xtr, ytr = load_pickle(os.path.join(data_path, "train.pkl"))
    xvl, yvl = load_pickle(os.path.join(data_path, "val.pkl"))
    xts, yts = load_pickle(os.path.join(data_path, "test.pkl"))
    
    with mlflow.start_run():
        for param in RF_PARAMS:
            params[param] = int(params[param])

        rf = RandomForestRegressor(**params)
        rf.fit(xtr, ytr)
        
        val_rmse = mean_squared_error(yvl, rf.predict(xvl), squared=False)
        mlflow.log_metric("val_rmse", val_rmse)
        test_rmse = mean_squared_error(yts, rf.predict(xts), squared=False) 
        mlflow.log_metric("test_rmse", test_rmse)

@click.command()
@click.option(
    "--data_path",
    default=configura.dest_path,
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path, top_n):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )
    for run in runs:
        train_and_log_model(data_path, run.data.params)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    best_run = runs[0]
    print(best_run.data.metrics)
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    registered_model = mlflow.register_model(model_uri, "BestModel")
    print(f"Registered model: {registered_model.name} (version {registered_model.version})")

if __name__ == '__main__':
    run_register_model()