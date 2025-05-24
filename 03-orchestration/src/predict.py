import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import mlflow
from utils import load_pickle
from pathlib import Path
from typing import Any
from src.config import Location, Training


def main(data_path: Path, mlflow_model_run: str, dict_vectorizer_path: Path) -> None:

    X, y = load_pickle(data_path)
    dv = load_pickle(dict_vectorizer_path)

    mlflow.set_tracking_uri(mlflow_tracking_uri)


def predict(X: Any, model_run: str, dv: DictVectorizer):

    X = dv.transform(X)
    loaded_model = mlflow.pyfunc.load_model(f"runs:/{model_run}/model")
    return loaded_model.predict(X)


if __name__ == "__main__":

    data_path: Path = sys.argv[1]
    mlflow_model_run: str = sys.argv[2]
    dv_path: Path = sys.argv[3]
    #    output_path = sys.argv[4]

    mlflow.set_tracking_uri(uri=Training().mlflow_tracking_uri)

    preds = predict(
        X=load_pickle(filename=data_path)[0],
        model_run=mlflow_model_run,
        dv=load_pickle(filename=dv_path),
    )

    # Save predictions
    print(preds[:50])
