import pandas as pd
import numpy as np
import joblib
import os

# --- CONFIGURATION ---
MODEL_PATH = 'models/store_demand_models.joblib'
DATA_PATH = 'data/raw/retail_hourly_synthetic.csv'
OUTPUT_PATH = 'outputs/store_replenishment_recs.csv'

def generate_store_recommendations():
    print("🚀 STARTING STORE INFERENCE PIPELINE...")
    
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model not found. Run store_forecast_model.py first.")
        return

    # 1. Load Models and Data
    combined_models = joblib.load(MODEL_PATH)
    cls_model = combined_models['classifier']
    reg_model = combined_models['regressor']
    features = combined_models['features']
    
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
    
    # 2. Data Preparation (Must match training features exactly)
    target = 'pos_sales_units'
    df['sales_volume'] = df[target]
    
    df['stockout_flag_int'] = df['stockout_flag'].apply(lambda x: 1 if x == 'Y' else 0)
    df = df.sort_values(by=['store_id', 'sku_id', 'timestamp'])
    
    # Lags
    df['lag_1h'] = df.groupby(['store_id', 'sku_id'])['sales_volume'].shift(1)
    df['lag_24h'] = df.groupby(['store_id', 'sku_id'])['sales_volume'].shift(24)
    
    # Time/Operational Features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # --- REAL USE CASE FIX: Impute Lags with 0 ---
    lag_columns = ['lag_1h', 'lag_24h']
    df[lag_columns] = df[lag_columns].fillna(0.0)

    # Drop any remaining NaNs from other potential features (if they exist)
    df.dropna(inplace=True)
    
    # 3. Select Live Batch (Last 24 Hours for ALL Stores/SKUs)
    max_date = df['timestamp'].max()
    cutoff_date = max_date - pd.Timedelta(hours=24)
    live_batch = df[df['timestamp'] > cutoff_date].copy()
    
    # 4. Two-Stage Prediction
    X_live = live_batch[features]
    
    # Stage 1: Predict Probability of Sale
    proba_sale = cls_model.predict_proba(X_live)[:, 1] 
    
    # Stage 2: Predict Volume (if sale occurs)
    pred_volume = reg_model.predict(X_live)
    
    # Final Forecast: P(Sale) * Volume(If Sale)
    live_batch['predicted_demand'] = proba_sale * pred_volume
    
    # 5. Replenishment Logic (Order Recommendation)
    def decision_logic(row):
        on_shelf = row['on_shelf_units']
        
        # RULE 1: HIGH Expiry Risk (Using waste_units from the raw CSV)
        waste_threshold = 5 
        if row['waste_units'] > waste_threshold: 
            # Order to clear the excess expiring stock AND stock up to capacity
            order_qty = row['planogram_capacity_units'] - on_shelf + row['waste_units']
            return 'ORDER_EXPIRY_PULL', max(0, np.ceil(order_qty))
        
        # RULE 2: Forecasted demand exceeds immediate on-shelf stock
        elif row['predicted_demand'] > on_shelf:
            # Recommend stocking up to planogram capacity based on forecast
            order_qty = row['planogram_capacity_units'] - on_shelf
            return 'ORDER_TO_CAPACITY', max(0, np.ceil(order_qty))
            
        return 'NO_ACTION', 0

    live_batch[['Recommendation_Type', 'Recommended_Qty']] = live_batch.apply(
        lambda x: pd.Series(decision_logic(x)), axis=1
    )
    
    # 6. Final Output Table
    output_cols = [
        'timestamp', 'store_id', 'sku_id', 
        'on_shelf_units', 'planogram_capacity_units',
        'predicted_demand', 'Recommended_Qty', 'Recommendation_Type',
        'promo_flag', 'footfall_count'
    ]
    
    final_df = live_batch[live_batch['Recommendation_Type'] != 'NO_ACTION'][output_cols]
    
    os.makedirs('outputs', exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"✅ Recommendations Generated for {final_df['store_id'].nunique()} stores.")
    print(f"📄 Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_store_recommendations()