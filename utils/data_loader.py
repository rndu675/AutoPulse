import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import json
from datetime import datetime

@st.cache_resource
def load_model(model_path='models/xgboost_model_artifacts.joblib'):
    """
    Loads and caches the machine learning model artifacts.
    """
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.warning(f"Failed to load model artifacts from {model_path}: {e}")
            return None
    return None

@st.cache_data
def load_geojson(geojson_path='data/provinces.geojson'):
    """
    Loads and caches the GeoJSON file for Sri Lanka provinces.
    """
    if os.path.exists(geojson_path):
        with open(geojson_path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_data(file_path='data/car_price_dataset.csv'):
    """
    Loads and preprocesses the vehicle dataset.
    """
    if not os.path.exists(file_path):
        st.error(f"Dataset not found at {file_path}")
        return pd.DataFrame()

    df = pd.read_csv(file_path, index_col=0)
    
    # --- Preprocessing logic from dataset. ---
    
    # --- 1. Clean column names to snake_case. ---
    def clean_column_name(col_name):
        col_name = str(col_name).lower()
        col_name = col_name.replace('(cc)', '_cc')
        col_name = col_name.replace('(km)', '_km')
        col_name = col_name.replace(' ', '_')
        col_name = col_name.replace('-', '_')
        col_name = col_name.replace('__', '_')
        return col_name.strip('_')
    
    df.columns = [clean_column_name(col) for col in df.columns]
    
    # --- 2. Drop duplicates. ---
    df.drop_duplicates(ignore_index=True, inplace=True)
    
    # --- 3. Filter for USED condition. ---
    if 'condition' in df.columns:
        df = df[df['condition'] == 'USED'].copy()
    
    # --- 4. Feature Engineering. ---
    # Calculate vehicle age dynamically.
    if 'yom' in df.columns:
        df['age'] = datetime.now().year - df['yom']
        
        # Classify year category based on manufacture year.
        def classify_year(year):
            if year < 2010: return "Old"
            elif year < 2018: return "Intermediate"
            else: return "Modern"
        df['year_category'] = df['yom'].apply(classify_year)
    
    # Define engine segment & apply log transform.
    if 'engine_cc' in df.columns:
        df['log_engine_cc'] = np.log1p(df['engine_cc'])
        def classify_engine(cc):
            if pd.isna(cc): return "Unknown"
            cc = float(cc)
            if cc < 800: return "Micro (<800cc)"
            elif cc <= 1200: return "Compact (800-1200cc)"
            elif cc <= 1600: return "Mid-Range (1200-1600cc)"
            else: return "Large (>1600cc)"
        df['engine_segment'] = df['engine_cc'].apply(classify_engine)
        
    # Group rare and common brands.
    if 'brand' in df.columns:
        brand_counts = df['brand'].value_counts()
        def group_brand(x):
            count = brand_counts.get(x, 0)
            if count > 100: return x
            elif count >= 10: return 'OTHER'
            else: return 'RARE'
        df['brand_grouped'] = df['brand'].apply(group_brand)
        
    # Clean leasing information (matching model binary logic).
    if 'leasing' in df.columns:
        df['leasing'] = df['leasing'].apply(lambda x: True if 'lease' in str(x).lower() and 'no' not in str(x).lower() else False)

    # Convert binary feature values to boolean.
    binary_cols = ['air_condition', 'power_steering', 'power_mirror', 'power_window']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: True if str(x).lower() == 'available' else False)

    # Map towns to provinces.
    if 'town' in df.columns:
        town_to_province = {
            "Colombo": "Western", "Gampaha": "Western", "Negombo": "Western", "Kalutara": "Western",
            "Panadura": "Western", "Moratuwa": "Western", "Dehiwala-Mount-Lavinia": "Western",
            "Maharagama": "Western", "Kotte": "Western", "Wattala": "Western", "Ja-Ela": "Western",
            "Kelaniya": "Western", "Kadawatha": "Western", "Nugegoda": "Western", "Piliyandala": "Western",
            "Boralesgamuwa": "Western", "Kandy": "Central", "Matale": "Central", "Nuwara-Eliya": "Central",
            "Gampola": "Central", "Nawalapitiya": "Central", "Hatton": "Central", "Galle": "Southern",
            "Matara": "Southern", "Hambantota": "Southern", "Weligama": "Southern", "Tangalle": "Southern",
            "Hikkaduwa": "Southern", "Ambalangoda": "Southern", "Jaffna": "Northern", "Vavuniya": "Northern",
            "Kilinochchi": "Northern", "Mullaitivu": "Northern", "Batticaloa": "Eastern", "Trincomalee": "Eastern",
            "Ampara": "Eastern", "Kalmunai": "Eastern", "Kurunegala": "North Western", "Puttalam": "North Western",
            "Kuliyapitiya": "North Western", "Chilaw": "North Western", "Anuradapura": "North Central",
            "Polonnaruwa": "North Central", "Badulla": "Uva", "Bandarawela": "Uva", "Haputale": "Uva",
            "Welimada": "Uva", "Ratnapura": "Sabaragamuwa", "Kegalle": "Sabaragamuwa", "Balangoda": "Sabaragamuwa"
        }
        df["province"] = df["town"].map(town_to_province).fillna("Other")

    return df
