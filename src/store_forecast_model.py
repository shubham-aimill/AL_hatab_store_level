import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.metrics import mean_absolute_error, f1_score
import os

# --- CONFIGURATION ---
INPUT_PATH = 'data/raw/retail_hourly_synthetic.csv'
SINGLE_MODEL_PATH = 'models/store_demand_models.joblib' 
TARGET = 'pos_sales_units'

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (Used for Zero-Robust Accuracy)"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (Calculated ONLY on non-zero true values)"""
    non_zero_mask = y_true > 0
    y_true_nz = y_true[non_zero_mask]
    y_pred_nz = y_pred[non_zero_mask]
    
    if len(y_true_nz) == 0:
        return np.nan
        
    return np.mean(np.abs((y_true_nz - y_pred_nz) / y_true_nz)) * 100

def train_store_pipeline():
    print("🚀 STARTING STORE FORECASTING PIPELINE (Two-Stage LightGBM - Reverting Filter)...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(INPUT_PATH, parse_dates=['timestamp'])
    except FileNotFoundError:
        print(f"❌ Error: {INPUT_PATH} not found.")
        return

    # 2. Data Preparation & Stockout Correction 
    print("   1/3: Engineering Features (Lags, Promos, Stockout Correction)...")
    
    # Target Variables
    df['sales_volume'] = df[TARGET]
    df['sale_occurrence'] = (df['sales_volume'] > 0).astype(int) 

    # --- STOCKOUT CORRECTION & Feature Engineering ---
    df['stockout_flag_int'] = df['stockout_flag'].apply(lambda x: 1 if x == 'Y' else 0)
    df = df.sort_values(by=['store_id', 'sku_id', 'timestamp'])
    
    # Lags [1h, 24h] 
    df['lag_1h'] = df.groupby(['store_id', 'sku_id'])['sales_volume'].shift(1)
    df['lag_24h'] = df.groupby(['store_id', 'sku_id'])['sales_volume'].shift(24)
    
    # Time/Operational Features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # --- Impute Lags with 0 ---
    lag_columns = ['lag_1h', 'lag_24h']
    df[lag_columns] = df[lag_columns].fillna(0.0)
    
    # Drop any rows where other critical features (like stock/promo) are missing
    df.dropna(inplace=True) 
    
    features = [
        'lag_1h', 'lag_24h', 
        'on_shelf_units', 'planogram_capacity_units',
        'promo_flag', 'footfall_count', 
        'stockout_flag_int',
        'hour', 'day_of_week'
    ]

    # --- Training Split ---
    split = int(len(df) * 0.8)
    X_train, X_test = df[features].iloc[:split], df[features].iloc[split:]
    y_reg_all = df['sales_volume']
    
    # 3. Stage 1: Classifier Training (Probability of Sale)
    print(f"   2/3: Training Stage 1 (Classifier - Probability) on {len(X_train)} records...")
    y_cls_train = df['sale_occurrence'].iloc[:split]
    y_cls_test = df['sale_occurrence'].iloc[split:]
    
    cls_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    cls_model.fit(X_train, y_cls_train)
    
    # 4. Stage 2: Regressor Training (Volume, trained ONLY on non-zero sales, NO VOLUME THRESHOLD)
    print("   3/3: Training Stage 2 (Regressor - Volume, Reverted to all Sales > 0)...")
    
    # *** FIX: Reverting the mask to train on all sales > 0 ***
    non_zero_train_mask = (y_reg_all.iloc[:split] > 0)
    X_reg_train = X_train[non_zero_train_mask]
    y_reg_train = y_reg_all.iloc[:split][non_zero_train_mask]
    
    reg_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    reg_model.fit(X_reg_train, y_reg_train)

    # --- Evaluation ---
    cls_preds = cls_model.predict(X_test)
    reg_preds = reg_model.predict(X_test)
    
    cls_proba = cls_model.predict_proba(X_test)[:, 1] 
    final_forecast = cls_proba * reg_preds
    
    y_true = df['sales_volume'].iloc[split:]

    # Calculate Metrics
    final_smape = smape(y_true, final_forecast)
    final_mape = mape(y_true, final_forecast)
    final_mae = mean_absolute_error(y_true, final_forecast)
    f1_score_cls = f1_score(y_cls_test, cls_preds)
    
    print("\n" + "="*50)
    print("📊 STORE MODEL PERFORMANCE (Two-Stage)")
    print("="*50)
    print(f"   Final sMAPE (Accuracy): {final_smape:.2f}%")
    print(f"   Final MAPE (Error):     {final_mape:.2f}%")
    print(f"   Final MAE (Error in units): {final_mae:.2f} units")
    print(f"   Classifier F1 Score (Predicting Sale): {f1_score_cls:.4f}")
    print("="*50)

    # Top Features
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': reg_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\n🏆 TOP 5 DRIVERS (Volume Prediction):")
    print(importance.head(5).to_string(index=False))

    # 5. Save Models 
    os.makedirs('models', exist_ok=True)
    combined_models = {
        'classifier': cls_model,
        'regressor': reg_model,
        'features': features
    }
    joblib.dump(combined_models, SINGLE_MODEL_PATH)
    print(f"\n💾 Both models saved successfully to a single file: {SINGLE_MODEL_PATH}")

if __name__ == "__main__":
    train_store_pipeline()