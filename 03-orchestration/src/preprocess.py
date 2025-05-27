from pathlib import Path
import pandas as pd
from prefect import flow, task
from .config import Location, Preprocess
import os
from .utils import dump_pickle, get_file
import argparse
import logging


@task
def read_file(file_path: str) -> pd.DataFrame:
    return pd.read_parquet(file_path)


@task
def calculate_duration(df: pd.DataFrame) -> pd.DataFrame:
    return (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60


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

    parser = argparse.ArgumentParser(
        description="Preprocess data for training, validation, and testing."
    )

    parser.add_argument(
        "--taxi_tp",
        type=str,
        required=True,
        help="Type of taxi data.",
    )
    parser.add_argument(
        "--year", type=int, required=True, help="Year of the data to preprocess."
    )
    parser.add_argument(
        "--month", type=int, required=True, help="Month of the data to preprocess."
    )

    args = parser.parse_args()

    processed_data = preprocess_data(
        file_path=os.path.join(
            Location().input_location,
            get_file(taxi_tp=args.taxi_tp, year=args.year, month=args.month),
        ),
        categorical_vars=Preprocess().categorical,
        target_var=Preprocess().target,
    )

    dump_pickle(
        obj=processed_data,
        filename=os.path.join(Location().output_location, "output.pkl"),
    )
