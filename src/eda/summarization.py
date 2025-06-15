# src/eda/summarization.py

import pandas as pd

def load_data(data_path):
    #data_path = "data/insurance_data.txt"
    df = pd.read_csv(data_path, sep='|')
    print(f"Loaded data with shape: {df.shape}")
    return df

def describe_data(df):
    print("\n Descriptive Statistics:")
    print(df.describe(include='all'))

def check_data_types(df):
    print("\n Column Data Types:")
    print(df.dtypes)
