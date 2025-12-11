import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Al Hatab Store Forecasting Dashboard", layout="wide")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    path = 'outputs/store_replenishment_recs.csv'
    if not os.path.exists(path):
        st.error("❌ Output file not found! Please run 'python src/store_inference.py' first.")
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate utilization
    df['shelf_utilization'] = df['on_shelf_units'] / df['planogram_capacity_units']
    df['stockout_risk'] = df['predicted_demand'] / (df['on_shelf_units'] + 1) # Simple risk proxy
    
    return df

all_df = load_data()

if not all_df.empty:
    # --- SIDEBAR FILTERS ---
    st.sidebar.title("📍 Store Manager View")
    st.sidebar.divider()
    
    # 1. Store Selector
    available_stores = all_df['store_id'].unique()
    selected_store = st.sidebar.selectbox("Select Store", available_stores)
    
    # 2. SKU Selector
    available_skus = all_df['sku_id'].unique()
    selected_sku = st.sidebar.multiselect("Filter SKU", options=available_skus, default=available_skus[:3])

    # FILTER DATA
    df = all_df[
        (all_df['store_id'] == selected_store) & 
        (all_df['sku_id'].isin(selected_sku))
    ]

    # --- MAIN HEADER ---
    st.title(f"🍞 {selected_store} - Replenishment Planner")
    st.markdown("Hourly Actionable Insights for On-Shelf Inventory")
    st.divider()
    
    if df.empty:
        st.warning("No actionable alerts found for the selected SKUs in this time frame.")
    else:
        # --- ROW 1: KPIS (Inventory Health) ---
        c1, c2, c3, c4 = st.columns(4)
        
        avg_util = df['shelf_utilization'].mean() * 100
        total_actions = len(df)
        expiry_actions = len(df[df['Recommendation_Type'] == 'ORDER_EXPIRY_PULL'])
        
        with c1: st.metric("🎯 Average Shelf Utilization", f"{avg_util:.1f}%")
        with c2: st.metric("🚨 Total Actionable Alerts", f"{total_actions}")
        with c3: st.metric("🗑️ Expiry Pull Alerts", f"{expiry_actions}", delta_color="inverse")
        with c4: st.metric("🛒 Avg Predicted Demand", f"{df['predicted_demand'].mean():.2f} units/hr")

        st.divider()
        
        # --- ROW 2: VISUALIZATION (Demand vs. Stock) ---
        st.subheader("Demand & Stock Trend")
        
        plot_df = df.melt(id_vars=['timestamp', 'sku_id'], 
                          value_vars=['predicted_demand', 'on_shelf_units', 'planogram_capacity_units'],
                          var_name='Metric', value_name='Units')
        
        fig = px.line(plot_df, x='timestamp', y='Units', color='Metric', line_dash='Metric',
                      title='Forecasted Demand vs. Current Shelf Stock (Last 24 Hours)',
                      color_discrete_map={
                          'predicted_demand': 'blue', 
                          'on_shelf_units': 'green',
                          'planogram_capacity_units': 'red'
                      })
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # --- ROW 3: ACTIONABLE TABLE ---
        st.subheader("📋 Final Replenishment Recommendations")
        
        display_cols = ['timestamp', 'sku_id', 'on_shelf_units', 'planogram_capacity_units', 
                        'predicted_demand', 'Recommended_Qty', 'Recommendation_Type', 'promo_flag', 'footfall_count']
        
        def highlight_logic(row):
            if row['Recommendation_Type'] == 'ORDER_EXPIRY_PULL':
                return ['background-color: #ffcccc'] * len(row)
            elif row['Recommendation_Type'] == 'ORDER_TO_CAPACITY':
                return ['background-color: #ccffcc'] * len(row)
            return [''] * len(row)

        st.dataframe(df[display_cols].style.apply(highlight_logic, axis=1), use_container_width=True)

else:
    st.info("Waiting for data... Run 'python src/store_inference.py'")