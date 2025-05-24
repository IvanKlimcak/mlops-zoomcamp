import os
import pickle
import mlflow
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from utils import load_pickle, dump_pickle
from config import Location, Training
from prefect import flow
from pathlib import Path


@flow
def train(training_sample: Path, dv_output: Path):

    X, y = load_pickle(training_sample)

    with mlflow.start_run():
        dv = DictVectorizer()
        X_train = dv.fit_transform(X)
        dump_pickle(dv, dv_output)

        lr = LinearRegression()
        lr.fit(X_train, y)


if __name__ == "__main__":
    mlflow.set_tracking_uri(uri=Training().mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name=Training().mlflow_experiment_name)
    mlflow.mlflow.sklearn.autolog()

    train(
        training_sample=os.path.join(Location().output_location, "train.pkl"),
        dv_output=os.path.join(Location().output_location, "dv.pkl"),
    )
