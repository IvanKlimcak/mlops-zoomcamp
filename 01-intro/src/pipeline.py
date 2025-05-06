import pandas as pd 
from typing import Union
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error as rmse
from functools import cached_property

TARGET = 'duration'
CATEGORICAL = ['PULocationID', 'DOLocationID']
TRAINING_PATH = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
VALIDATION_PATH = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet",

class TaxiRidePrediction:
    
    def __init__(self, path, target, vars):
        self.path = path
        self.target = target
        self.vars = vars 

    @cached_property
    def raw_data(self) -> pd.DataFrame:
        return pd.read_parquet(self.path)
    
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

    # Creating training pipeline class
    train = TaxiRidePrediction(
        path = TRAINING_PATH[0],
        target = TARGET,
        vars = CATEGORICAL
    )

    # Fit vectorizer
    dv = DictVectorizer()
    X_train = dv.fit_transform(X = train.regression_matrices[0])
    y_train = train.regression_matrices[1]
    
    # Fit regression 
    lr = LinearRegression()
    lr.fit(X = X_train, y = y_train)
    y_train_pred = lr.predict(X = X_train)
    
    # Creating validation pipeline     
    valid = TaxiRidePrediction(
    path = VALIDATION_PATH[0],
    target = TARGET,
    vars = CATEGORICAL
    )
    
    # Applying vectorizer
    X_valid = dv.transform(valid.regression_matrices[0])
    y_valid = valid.regression_matrices[1]
    
    # Applyinh regression 
    y_valid_pred = lr.predict(X_valid)

    return{
        'rmse_training_sample': rmse(y_train, y_train_pred),
        'rmse_validation_sample': rmse(y_valid, y_valid_pred)
    }
    
if __name__ == "__main__":
    main()