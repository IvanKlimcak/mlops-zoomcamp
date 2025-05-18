from pydantic import BaseModel


class Location(BaseModel):
    input_location: str = "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    training_sample: str = "green_tripdata_2023-01.parquet"
    validation_sample: str = "green_tripdata_2023-02.parquet"
    test_sample: str = "green_tripdata_2023-03.parquet"
    output_location: str = "/home/ubuntu/mlops-zoomcamp/03-orchestration/output"


class Preprocess(BaseModel):
    categorical: list[str] = ["PULocationID", "DOLocationID"]
    target: list[str] = ["duration"]
    numerical: list[str] = ["trip_distance"]


class Training(BaseModel):
    mlflow_tracking_uri: str = "http://0.0.0.0:5000/"
    mlflow_experiment_name: str = "lr"
