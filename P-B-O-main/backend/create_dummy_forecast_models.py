import pickle
import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
import os

warnings.filterwarnings('ignore')

def create_models():
    print("Loading drug names from test002.csv...")
    try:
        df = pd.read_csv("data/test002.csv")
        drug_names = [str(x).strip().upper() for x in df['drug_name'].dropna().unique()]
    except Exception as e:
        print(f"Could not load data: {e}")
        return

    # To save time, let's create models for just 50 common drugs in the dataset
    drug_models = {}
    target_drugs = drug_names[:50]
    
    print(f"Training ARIMA models on dummy historical data for {len(target_drugs)} drugs...")
    
    for _id, drug in enumerate(target_drugs):
        base_cost = np.random.uniform(20, 500)
        # 36 months of historical cost with a slight upward trend and noise
        historical_data = base_cost + np.linspace(0, base_cost * 0.2, 36) + np.random.normal(0, base_cost * 0.05, 36)
        
        model = ARIMA(historical_data, order=(1, 1, 0))
        result = model.fit()
        drug_models[drug] = result
        
    os.makedirs("data", exist_ok=True)
    with open("data/time_series_forecast_drugs.pkl", "wb") as f:
        pickle.dump(drug_models, f)
        
    print("✅ Successfully generated 'data/time_series_forecast_drugs.pkl'.")
    print("\nHere are a few valid drug names you can use for testing:")
    for drug in target_drugs[:5]:
        print(f" - {drug}")

if __name__ == "__main__":
    create_models()
