import pandas as pd

def load_dataset(filepath):
    df = pd.read_csv(filepath)
    return df