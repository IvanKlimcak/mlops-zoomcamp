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
