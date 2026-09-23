import pandas as pd

def load_splits(data_dir):
    train=pd.read_csv(data_dir/'train.csv')
    val=pd.read_csv(data_dir/'val.csv')
    test=pd.read_csv(data_dir/'test.csv')

    return train, val, test
    