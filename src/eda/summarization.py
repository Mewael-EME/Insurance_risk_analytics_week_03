# src/eda/summarization.py

import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded data with shape: {df.shape}")
    return df

def describe_data(df):
    print("\n Descriptive Statistics:")
    print(df.describe(include='all'))

def check_data_types(df):
    print("\n Column Data Types:")
    print(df.dtypes)
