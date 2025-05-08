import pandas as pd 
from typing import Union
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error as rmse
from functools import cached_property
import logging
import os 

TARGET = 'duration'
CATEGORICAL = ['PULocationID', 'DOLocationID']
DATA_PATH = "https://d37ci6vzurychx.cloudfront.net/trip-data/"
TRAINING_FILE = "yellow_tripdata_2023-01.parquet"
VALDATION_FILE = "yellow_tripdata_2023-02.parquet"

class TaxiRidePrediction:
    
    def __init__(self, path, file, target, vars):
        self.path = path
        self.file = file
        self.target = target
        self.vars = vars 

    @property
    def full_path(self):
        return os.path.join(self.path, self.file)
    
    @cached_property
    def raw_data(self) -> pd.DataFrame:
        return pd.read_parquet(self.full_path)
    
    @staticmethod
    def calculate_duration(df:pd.DataFrame) -> pd.DataFrame:
        df[TARGET] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds()/60
        return df       

    @staticmethod
    def filter_outliers(df:pd.DataFrame) -> pd.DataFrame:
        return df.loc[df[TARGET].between(1,60,inclusive='both')]
    
    @cached_property
    def processed_data(self) -> pd.DataFrame:
        return self.raw_data.pipe(self.calculate_duration).pipe(self.filter_outliers)
    
    @cached_property
    def regression_matrices(self) -> tuple:
        
        return (
            self.processed_data[self.vars].astype(str).to_dict(orient='records'),
            self.processed_data[self.target].values
        )

def main():

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    # Creating training pipeline class
    train = TaxiRidePrediction(
        path = DATA_PATH,
        file = TRAINING_FILE,
        target = TARGET,
        vars = CATEGORICAL
    )

    logger.info(f"Loading data: {TRAINING_FILE}")
    
    # Fit vectorizer
    dv = DictVectorizer()
    X_train = dv.fit_transform(X = train.regression_matrices[0])
    y_train = train.regression_matrices[1]
    
    # Fit regression 
    lr = LinearRegression()
    lr.fit(X = X_train, y = y_train)
    y_train_pred = lr.predict(X = X_train)
    rmse_train = rmse(y_train, y_train_pred)

    logger.info(f"RMSE for training sample: {rmse_train}")

    # Creating validation pipeline     
    valid = TaxiRidePrediction(
    path = DATA_PATH,
    file = VALDATION_FILE,
    target = TARGET,
    vars = CATEGORICAL
    )

    logger.info(f"Loading data: {VALDATION_FILE}")
    
    # Fitting vectorizer 
    X_valid = dv.transform(valid.regression_matrices[0])
    y_valid = valid.regression_matrices[1]
    
    # Predicting using trained LR model 
    y_valid_pred = lr.predict(X_valid)
    rmse_valid = rmse(y_valid, y_valid_pred)

    logger.info(f"RMSE for validation sample: {rmse_valid}")

if __name__ == "__main__":
    main()