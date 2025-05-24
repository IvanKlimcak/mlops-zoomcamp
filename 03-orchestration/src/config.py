from pydantic import BaseModel


class Location(BaseModel):
    input_location: str = "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    output_location: str = "/home/ubuntu/mlops-zoomcamp/03-orchestration/output"
    training_sample: str = "green_tripdata_2023-01.parquet"
    validation_sample: str = "green_tripdata_2023-02.parquet"
    test_sample: str = "green_tripdata_2023-03.parquet"


class Preprocess(BaseModel):
    categorical: list[str] = ["PULocationID", "DOLocationID"]
    target: str = "duration"


class Training(BaseModel):
    mlflow_tracking_uri: str = "http://0.0.0.0:5000/"
    mlflow_experiment_name: str = "lr"
