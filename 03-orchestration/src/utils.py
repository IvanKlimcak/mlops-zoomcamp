import pickle
from prefect import task


@task
def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


@task
def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def get_file(taxi_tp: str, year: int, month: int) -> str:
    return f"{taxi_tp}_tripdata_{year:04d}-{month:02d}.parquet"
