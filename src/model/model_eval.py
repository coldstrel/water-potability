import numpy as np
import pandas as pd
import os
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data: {filepath}: {e}")
    
def prepare_data(data:pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns=['Potability'], axis=1)
        y = data['Potability']
        return X, y
    except Exception as e:
        raise Exception(f"Error preparing data {data}: {e}")

def load_model(filepath:str):
    try:
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {filepath}: {e}")

def evaluation_model(model, X_test: pd.DataFrame, y_test:pd.Series) -> dict:
    try:
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        metrics_dict = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error evaluating model {model} : {e}")

def save_metrics(metrics_dict: dict, filepath:str) -> None:
    try: 
        with open(filepath, "w") as f:
            json.dump(metrics_dict, f, indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics {filepath} : {e }")

def main():
    try:
        test_data_path = "data/processed/test_processed_mean.csv"
        model_path = "models/model.pkl"
        metrics_path = "reports/metrics.json"
        
        test_data = load_data(test_data_path)
        X_test, y_test = prepare_data(test_data)
        model = load_model(model_path)
        metrics = evaluation_model(model, X_test, y_test)
        save_metrics(metrics, metrics_path)
    except Exception as e:
        raise Exception(F"Error {e}")
    
    
if __name__ == "__main__":
    main()
