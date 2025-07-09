# utils/preprocessing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['store_id', 'sku_id', 'date'])

    # Encode features
    feature_cols = ['sales_qty', 'day_of_week', 'is_weekend', 'is_holiday',
                    'temperature', 'rainy', 'inventory_level', 'supplier_delay']
    
    scalers = {}
    for col in feature_cols:
        scaler = MinMaxScaler()
        df[col] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler

    return df[feature_cols].values, df
