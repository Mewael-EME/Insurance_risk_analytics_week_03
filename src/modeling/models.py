from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_models(X_train, y_train):
    """
    Train multiple regression models on training data.

    Returns a dict of trained models.
    """
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42, objective='reg:squarederror')
    }
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        models[name] = model
    return models

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a single trained model using RMSE and R2 metrics.
    """
    import numpy as np
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)  
    rmse = np.sqrt(mse)                      
    r2 = r2_score(y_test, preds)
    return {'RMSE': rmse, 'R2': r2}

def evaluate_all(models, X_test, y_test):
    """
    Evaluate all models and return a dictionary with scores.
    """
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test, y_test)
    return results
