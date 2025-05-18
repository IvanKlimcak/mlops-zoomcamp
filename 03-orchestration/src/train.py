import os
import pickle
import mlflow
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from utils import load_pickle
from config import Location, Training
from prefect import flow


@flow
def train(path_to_file: str):

    X, y = load_pickle(path_to_file)

    with mlflow.start_run():
        dv = DictVectorizer()
        X_train = dv.fit_transform(X)

        lr = LinearRegression()
        lr.fit(X_train, y)


if __name__ == "__main__":
    mlflow.set_tracking_uri(uri=Training().mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name=Training().mlflow_experiment_name)
    mlflow.mlflow.sklearn.autolog()

    train(path_to_file=os.path.join(Location().output_location, "train.pkl"))
