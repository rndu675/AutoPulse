import streamlit as st
import pandas as pd
import numpy as np
import os
import shap
import plotly.express as px
from datetime import datetime
from utils.data_loader import load_model
from utils.style_loader import clean_html

# --- Configuration Constants ---
CONTINUOUS_COLS = ['log_engine_cc', 'age']
CATEGORICAL_COLS = ['gear', 'fuel_type', 'province', 'brand_grouped']
BINARY_COLS = ['leasing', 'air_condition', 'power_steering', 'power_mirror', 'power_window']

@st.cache_data
def get_shap_analysis(_model, _df, _model_features):
    """
    Computes aggregated SHAP values on a downsampled subset of the dataset.
    """
    # Sample a fixed safe size (150 rows) to prevent memory issues and speed up SHAP
    sample_size = min(150, len(_df))
    sample_df = _df.sample(n=sample_size, random_state=42)
    
    # Feature engineering on sample (ensuring consistency with data loader)
    X_sample = sample_df[['log_engine_cc', 'age', 'gear', 'fuel_type', 'province', 'brand_grouped', 
                          'leasing', 'air_condition', 'power_steering', 'power_mirror', 'power_window']].copy()
    
    X_enc_sample = pd.get_dummies(X_sample, columns=CATEGORICAL_COLS + BINARY_COLS)
    
    # Align columns
    for col in _model_features:
        if col not in X_enc_sample.columns:
            X_enc_sample[col] = 0
    X_final_sample = X_enc_sample[_model_features]
    
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X_final_sample)
    
    # Aggregate SHAP values
    shap_df = pd.DataFrame(shap_values, columns=_model_features)
    agg_shap = {}
    for orig in CONTINUOUS_COLS:
        if orig in shap_df.columns:
            agg_shap[orig] = np.mean(np.abs(shap_df[orig]))
    
    for orig in (CATEGORICAL_COLS + BINARY_COLS):
        ohe_cols = [c for c in shap_df.columns if c.startswith(orig + '_')]
        if ohe_cols:
            agg_shap[orig] = np.sum([np.mean(np.abs(shap_df[c])) for c in ohe_cols])
        elif orig in shap_df.columns:
            agg_shap[orig] = np.mean(np.abs(shap_df[orig]))
    
    return agg_shap

def app(df):
    """
    Renders the Price Prediction page of the AutoPulse.AI web application.
    Builds a specifications configuration interface, runs the XGBoost machine learning model,
    and displays individual SHAP impact scores, forced attribution charts, and attribution tables.

    Parameters:
        df (pd.DataFrame): The preprocessed vehicle listings dataset.
    """
    st.title("Price Prediction (XGBoost)")

    model_artifacts = load_model()
    
    if not model_artifacts:
        st.error("**Model artifacts not found or could not be loaded!**")
        st.info("Currently viewing a simulation of the prediction interface.")
        model = None
        model_features = None
    else:
        model = model_artifacts['model']
        model_features = model_artifacts['features']

    st.subheader(":material/settings: Vehicle Specifications")
    
    # Pre-populate options from the dataset.
    brands = sorted(df['brand'].unique())
    gears = sorted(df['gear'].unique())
    fuels = sorted(df['fuel_type'].unique())
    provinces = sorted(df['province'].unique())

    col1, col2 = st.columns(2)

    with col1:
        selected_brand = st.selectbox("Brand", brands)

    with col2:
        province = st.selectbox("Province", provinces)

    with col1:
        fuel_type = st.selectbox("Fuel Type", fuels)
        current_year = datetime.now().year
        yom = st.slider("Year of Manufacture (YOM)", min_value=1950, max_value=current_year, value=2015)
        
        is_electric = (fuel_type == "Electric")
        engine_cc = st.slider("Engine Capacity (cc)", min_value=0, max_value=8000, value=0 if is_electric else 1300, step=100, disabled=is_electric)
        millage = st.slider("Millage (km)", min_value=0, max_value=500000, value=50000, step=1000)
        
        default_gear_idx = gears.index("Automatic") if "Automatic" in gears else 0
        gear = st.selectbox("Transmission", gears, index=default_gear_idx if is_electric else 0, disabled=is_electric)

    with col2:
        st.markdown("**:material/checklist: Features & Options**")
        leasing = st.checkbox("Ongoing Leasing")
        air_condition = st.checkbox("Air Conditioning", value=True)
        power_steering = st.checkbox("Power Steering", value=True)
        power_mirror = st.checkbox("Power Mirror", value=True)
        power_window = st.checkbox("Power Window", value=True)

    st.markdown("---")
    
    # --- 3. Model prediction execution ---
    if st.button("Predict Vehicle Price", type="primary"):
        # Map brand to grouping categories.
        brand_counts = df['brand'].value_counts()
        count = brand_counts.get(selected_brand, 0)
        if count > 100: brand_grouped = selected_brand
        elif count >= 10: brand_grouped = 'OTHER'
        else: brand_grouped = 'RARE'

        # Override values for Electric vehicles to conform to the training dataset schema.
        final_engine_cc = 1500 if fuel_type == "Electric" else engine_cc
        final_gear = "Automatic" if fuel_type == "Electric" else gear

        # Prepare feature dictionary.
        input_data = {
            'log_engine_cc': np.log1p(final_engine_cc),
            'age': datetime.now().year - yom,
            'gear': final_gear,
            'fuel_type': fuel_type,
            'province': province,
            'brand_grouped': brand_grouped,
            'leasing': leasing,
            'air_condition': air_condition,
            'power_steering': power_steering,
            'power_mirror': power_mirror,
            'power_window': power_window
        }

        if model_artifacts:
            # Create DataFrame.
            X_input = pd.DataFrame([input_data])
            X_encoded = pd.get_dummies(X_input, columns=CATEGORICAL_COLS + BINARY_COLS)
            for col in model_features:
                if col not in X_encoded.columns:
                    X_encoded[col] = 0
            X_final = X_encoded[model_features]
            
            # Predict vehicle price.
            pred_log = model.predict(X_final)
            pred_price = np.exp(pred_log)[0] # Use np.exp instead of expm1
            
            # Calculate SHAP for this specific prediction.
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_final)
            
            # Store everything in session state for persistence.
            st.session_state['pred_results'] = {
                'price': pred_price,
                'log_price': pred_log[0],
                'X_final': X_final,
                'shap_values': shap_values,
                'expected_value': explainer.expected_value,
                'input_data': input_data,
                'original_brand': selected_brand,
                'yom': yom
            }
        else:
            # Setup simulation session state.
            st.session_state['pred_results'] = {
                'price': 34.50,
                'is_sim': True
            }

    # --- 4. Render valuation results ---
    if 'pred_results' in st.session_state:
        res = st.session_state['pred_results']
        st.markdown("---")
        
        if res.get('is_sim'):
            st.success(f"### :material/payments: Estimated Market Value: {res['price']:,.2f} Lakhs (LKR)")
            st.caption("Disclaimer: This is a placeholder prediction.")
        else:
            st.success(f"### :material/payments: Estimated Market Value: {res['price']:,.2f} Lakhs (LKR)")
            st.info(f"The model predicted a log-price of {res['log_price']:.4f}, which converts to ≈ {res['price']:,.2f} Lakhs LKR.")

            # --- Individual SHAP Force Plot ---
            st.subheader(":material/visibility: Prediction Breakdown (SHAP Force Plot)")
            st.write("This plot shows how specific features influenced the **Log-Price** estimate.")
            
            try:
                # --- 1. SHAP aggregation logic ---
                
                # Map the raw SHAP values (one-hot encoded) back to the clean category names.
                agg_shap_single = {}
                for i, col in enumerate(res['X_final'].columns):
                    val = res['shap_values'][0][i]
                    found = False
                    for orig in (CATEGORICAL_COLS + BINARY_COLS):
                        if col.startswith(orig + '_') or col == orig:
                            agg_shap_single[orig] = agg_shap_single.get(orig, 0) + val
                            found = True
                            break
                    if not found:
                        agg_shap_single[col] = agg_shap_single.get(col, 0) + val

                # --- 2. Reconstruct human-readable values for plot ---
                # Ensure the plot labels look like "Brand = TOYOTA" instead of one-hot encoded names.
                clean_names = []
                clean_values = []
                clean_shaps = []
                
                # Re-derive input values for clean display on the plot.
                # Use the display-ready names for simplicity.
                for feat, impact in agg_shap_single.items():
                    display_name = feat.replace('_grouped', '').replace('_', ' ').title()
                    clean_names.append(display_name)
                    clean_shaps.append(impact)
                    
                    # Find the value of the feature (Age, Gear, etc.)
                    # Access from stored 'res' to avoid UnboundLocalError during reruns.
                    if 'age' in feat.lower(): val = f"{res['input_data']['age']}y"
                    elif 'engine' in feat.lower(): val = f"{int(np.expm1(res['input_data']['log_engine_cc']))}cc"
                    elif 'brand' in feat.lower(): val = res['original_brand']
                    elif 'gear' in feat.lower(): val = res['input_data']['gear']
                    elif 'fuel' in feat.lower(): val = res['input_data']['fuel_type']
                    elif 'province' in feat.lower(): val = res['input_data']['province']
                    elif feat in BINARY_COLS: val = "Yes" if res['input_data'][feat] else "No"
                    else: val = ""
                    clean_values.append(val)

                # --- 3. Render force plot ---
                def st_shap(plot, height=None):
                    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
                    st.iframe(shap_html, height=height)

                force_p = shap.force_plot(
                    res['expected_value'], 
                    np.array(clean_shaps), 
                    pd.Series(clean_values, index=clean_names), 
                    matplotlib=False,
                    link='identity'
                )
                st_shap(force_p, height=150)
                
                st.caption("🔴 Red bars push the price higher | 🔵 Blue bars push the price lower")
                st.markdown(f"**Base Log-Price:** `{res['expected_value']:.2f}` → **Predicted Log-Price:** `{res['log_price']:.2f}`")
                
                # --- Aggregated price attribution table ---
                st.markdown("#### :material/leaderboard: Feature Impact Estimates (Lakhs LKR)")
                st.write("Below are the estimated contributions of each feature group to the final price:")
                
                shap_contribs = []
                for i, feat in enumerate(clean_names):
                    log_val = clean_shaps[i]
                    if abs(log_val) > 0.0001:
                        # Multiplicative factor calculated as exp(log_val) for log-models.
                        perc_impact = (np.exp(log_val) - 1) * 100
                        # Calculation of Lakhs impact:
                        # Formula: Price - (Price / exp(log_val)).
                        # Shows how much the price would change if the feature was neutral.
                        lakhs_impact = res['price'] * (1 - 1/np.exp(log_val))
                        
                        shap_contribs.append({
                            "Feature Group": feat,
                            "Log Impact": round(log_val, 4),
                            "Est. Price Change (%)": f"{perc_impact:+.1f}%",
                            "Est. Impact (Lakhs)": f"{lakhs_impact:+.2f}"
                        })
                
                st.table(pd.DataFrame(shap_contribs).sort_values(by="Log Impact", ascending=False))
                
            except Exception as e:
                st.error(f"Error generating insights: {e}")

    # --- 5. Model transparency & insights ---
    if st.checkbox("Show Model Insights"):
        if model_artifacts:
            st.markdown("### :material/insights: Model Insights & Transparency")
            st.markdown(clean_html("""
            This implementation uses **XGBoost Regressor** to predict car prices. It handles mixed data types by one-hot encoding categorical and binary features, such as `brand_grouped` and `leasing`. 
            
            To maximize performance, the code employs **RandomizedSearchCV**, executing 300 fits to find the optimal balance of hyperparameters like `learning_rate` and `max_depth`. The final model achieved an **R-squared of 0.8942**, indicating it explains approximately 89% of the variance in the log-transformed price. L1 and L2 regularization were applied to prevent overfitting, resulting in a robust, high accuracy predictive tool for the secondary market.
            """))
            
            # --- 1. Aggregated feature importance ---
            st.subheader(":material/bar_chart: Aggregated Feature Importance")
            try:
                feature_importances = model.feature_importances_
                importance_df = pd.DataFrame({
                    'feature': model_features,
                    'importance': feature_importances
                })

                # Aggregation logic from notebook
                aggregated_importances = {}
                for _, row in importance_df.iterrows():
                    f_name = row['feature']
                    imp = row['importance']
                    found = False
                    for orig in (CATEGORICAL_COLS + BINARY_COLS):
                        if f_name.startswith(orig + '_') or f_name == orig:
                            aggregated_importances[orig] = aggregated_importances.get(orig, 0) + imp
                            found = True
                            break
                    if not found:
                        aggregated_importances[f_name] = aggregated_importances.get(f_name, 0) + imp

                agg_imp_df = pd.DataFrame(list(aggregated_importances.items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=True)
                # Clean names for display
                agg_imp_df['Feature'] = agg_imp_df['Feature'].apply(lambda x: x.replace('_grouped', '').replace('_', ' ').title())
                
                # Plot with Plotly for interactivity
                fig_imp = px.bar(agg_imp_df, x='Importance', y='Feature', orientation='h',
                                 title="Feature Contribution to Model (Aggregated)",
                                 color='Importance', color_continuous_scale='Viridis')
                
                # Ensure longest is at the top
                fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}, title_x=0.5, title_xanchor='center')
                st.plotly_chart(fig_imp, width="stretch")
                st.info("This chart identifies the core drivers the XGBoost model prioritized during training to minimize overall predictive error.")

            except Exception as e:
                st.error(f"Error calculating feature importance: {e}")

            # --- 2. Aggregated SHAP analysis ---
            st.subheader(":material/analytics: Aggregated SHAP Analysis (Mean Absolute Values)")
            st.info("Calculating SHAP values using an optimized subset of the dataset...")
            
            try:
                aggregated_shap = get_shap_analysis(model, df, model_features)
                agg_shap_df = pd.DataFrame(list(aggregated_shap.items()), columns=['Feature', 'Mean Absolute SHAP Value']).sort_values(by='Mean Absolute SHAP Value', ascending=True)
                # Clean names for display
                agg_shap_df['Feature'] = agg_shap_df['Feature'].apply(lambda x: x.replace('_grouped', '').replace('_', ' ').title())
                
                fig_shap = px.bar(agg_shap_df, x='Mean Absolute SHAP Value', y='Feature', orientation='h',
                                 title="SHAP Feature Importance (Aggregated Impact)",
                                 color='Mean Absolute SHAP Value', color_continuous_scale='Plasma')
                
                # Ensure longest is at the top
                fig_shap.update_layout(yaxis={'categoryorder':'total ascending'}, title_x=0.5, title_xanchor='center')
                st.plotly_chart(fig_shap, width="stretch")
                st.info("SHAP values quantify the average magnitude of a feature's impact on individual price predictions across the entire dataset.")
                
            except Exception as e:
                st.error(f"Error calculating SHAP values: {e}")
        else:
            st.warning("Please load the model artifacts to see insights.")
