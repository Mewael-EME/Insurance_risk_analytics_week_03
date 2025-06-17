import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler  
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_filter(df_path):
    """Load dataset and filter rows with claims > 0."""
    df = pd.read_csv(df_path)

    if 'TotalClaims' not in df.columns:
        raise KeyError("Column 'TotalClaims' not found in the dataset.")

    df = df[df['TotalClaims'] > 0].reset_index(drop=True)
    return df

def preprocess(df, target_col='TotalClaims'):
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in the DataFrame.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_feats = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_feats = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Ensure all categoricals are strings
    X[categorical_feats] = X[categorical_feats].astype(str)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())  # ✅ Fix
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_feats),
        ('cat', categorical_transformer, categorical_feats)
    ])

    X_proc = preprocessor.fit_transform(X)
    return X_proc, y, preprocessor

def split_data(X, y, test_size=0.3, random_state=42):
    """Split data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
