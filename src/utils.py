import pandas as pd

def read_series(filename):
    df = pd.read_csv(filename, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.astype(float)
    return df
