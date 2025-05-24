from pathlib import Path
import pandas as pd
from prefect import flow, task
from config import Location, Preprocess
import os
from utils import dump_pickle


@task
def read_file(file_path: str) -> pd.DataFrame:
    return pd.read_parquet(file_path)


@task
def calculate_duration(df: pd.DataFrame) -> pd.DataFrame:
    return (df.lpep_dropoff_datetime - df.lpep_pickup_datetime).dt.total_seconds() / 60


@task
def filter_valid_durations(df: pd.DataFrame, target_var: str) -> pd.DataFrame:
    return df.loc[df[target_var].between(1, 60)]


@task
def convert_categorical(df: pd.DataFrame, categorical_vars) -> pd.DataFrame:
    df[categorical_vars] = df[categorical_vars].fillna(-1).astype("int").astype("str")
    return df


@flow
def preprocess_data(file_path: str, categorical_vars: list, target_var: str) -> tuple:
    df = (
        read_file(file_path)
        .assign(duration=calculate_duration)
        .pipe(filter_valid_durations, target_var)
        .pipe(convert_categorical, categorical_vars)
    )
    return (df[categorical_vars].to_dict(orient="records"), df[target_var].values)


if __name__ == "__main__":

    train = preprocess_data(
        file_path=os.path.join(Location().input_location, Location().training_sample),
        categorical_vars=Preprocess().categorical,
        target_var=Preprocess().target,
    )

    dump_pickle(
        obj=train, filename=os.path.join(Location().output_location, "train.pkl")
    )

    valid = preprocess_data(
        file_path=os.path.join(Location().input_location, Location().validation_sample),
        categorical_vars=Preprocess().categorical,
        target_var=Preprocess().target,
    )

    dump_pickle(
        obj=valid, filename=os.path.join(Location().output_location, "valid.pkl")
    )

    test = preprocess_data(
        file_path=os.path.join(Location().input_location, Location().test_sample),
        categorical_vars=Preprocess().categorical,
        target_var=Preprocess().target,
    )

    dump_pickle(obj=test, filename=os.path.join(Location().output_location, "test.pkl"))
