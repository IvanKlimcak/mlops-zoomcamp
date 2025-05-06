import pandas as pd 
from pathlib import Path 
from typing import Union
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error as rmse

TARGET = 'duration'
CATEGORICAL = ['PULocationID', 'DOLocationID']

def load_data(path:Path) -> pd.DataFrame:
    return pd.read_parquet(path)

def calculate_duration(df:pd.DataFrame) -> pd.DataFrame:
    df[TARGET] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds()/60
    return df 

def filter_outliers(df:pd.DataFrame) -> pd.DataFrame:
    return df.loc[df[TARGET].between(1,60,inclusive='both')]

def transform_raw_data(df:pd.DataFrame) -> pd.DataFrame:
    return (
        df.pipe(calculate_duration)
          .pipe(filter_outliers)
    )

def get_matrices(df:pd.DataFrame) -> tuple:
    X = df[CATEGORICAL].astype(str).to_dict(orient='records')
    y = df[TARGET].values
    
    return (X, y)

def vectorize_matrix(X:dict, dv:Union[DictVectorizer|None]=None) -> tuple:
    if dv is None:
        dv = DictVectorizer()
        X_out = dv.fit_transform(X) 
    else:
        X_out = dv.transform(X)
    
    return (X_out, dv) 

def predict_evaluate(X, y, lr:Union[LinearRegression|None]=None) -> tuple:
    if lr is None:
        lr = LinearRegression()
        lr.fit(X, y)
        
    y_pred = lr.predict(X)
    
    return rmse(y, y_pred)



