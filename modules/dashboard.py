import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import folium
import branca.colormap as cm
import copy
from utils.data_loader import load_geojson
from utils.style_loader import load_svg, clean_html

def app(df):
    """
    Renders the Market Insights Dashboard page of the AutoPulse.AI web application.
    Builds dynamic filtering widgets and displays five strategic tabs of market perspectives
    including price distributions, depreciation curves, engine specifications, comfort feature
    premiums, and a regional choropleth map.

    Parameters:
        df (pd.DataFrame): The preprocessed vehicle listings dataset.
    """
    st.title("Market Insights Dashboard")
    st.markdown("""
        Explore the dynamics of the Sri Lankan vehicle market through five strategic phases 
        of data analysis. Use the **Global Filters** below to customize the visualizations 
        based on your specific budget and preferences.
    """)

    remove_outliers = st.checkbox("Remove price outliers", value=False, key="dash_outliers")
    if remove_outliers:
        df = df[df['price'] <= df['price'].quantile(0.95)]

    # --- Global dashboard filters ---
    def reset_filters():
        st.session_state.p_range = (float(df['price'].min()), float(df['price'].max()))
        st.session_state.y_range = (int(df['yom'].min()), int(df['yom'].max()))
        st.session_state.m_range = (int(df['millage_km'].min()), int(df['millage_km'].max()))

    if 'p_range' not in st.session_state:
        reset_filters()

    with st.expander(":material/filter_alt: Global Dashboard Filters", expanded=True):
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            price_range = st.slider("Price Range (Lakhs LKR)", 
                                    float(df['price'].min()), float(df['price'].max()), 
                                    key="p_range")
        
        with col_f2:
            year_range = st.slider("Year of Manufacture", 
                                   int(df['yom'].min()), int(df['yom'].max()), 
                                   key="y_range")
            
        with col_f3:
            mileage_range = st.slider("Mileage Range (km)", 
                                      int(df['millage_km'].min()), int(df['millage_km'].max()), 
                                      key="m_range")
        
        st.button("Reset Filters", type="secondary", on_click=reset_filters)

    # Apply global filters
    filtered_df = df[
        (df['price'] >= price_range[0]) & (df['price'] <= price_range[1]) &
        (df['yom'] >= year_range[0]) & (df['yom'] <= year_range[1]) &
        (df['millage_km'] >= mileage_range[0]) & (df['millage_km'] <= mileage_range[1])
    ]

    # --- Market strategy metrics ---
    total_count = len(df)
    filtered_count = len(filtered_df)
    perc_filtered = (filtered_count / total_count) * 100 if total_count > 0 else 0
    
    if not filtered_df.empty:
        median_p = filtered_df['price'].median()
        median_p_str = f"{median_p:.1f}L"
        avg_age = filtered_df['age'].mean()
        avg_age_str = f"{avg_age:.1f} yrs"
        median_m = filtered_df['millage_km'].median()
        median_m_str = f"{median_m/1000:.0f}k km"
        top_brand = filtered_df['brand'].mode()[0]
        top_fuel = filtered_df['fuel_type'].mode()[0]
    else:
        median_p_str = "N/A"
        avg_age_str = "N/A"
        median_m_str = "N/A"
        top_brand = "N/A"
        top_fuel = "N/A"

    svg_car = load_svg("car.svg")
    svg_price = load_svg("price.svg")
    svg_calendar = load_svg("calendar.svg")
    svg_mileage = load_svg("mileage.svg")
    svg_brand = load_svg("brand.svg")
    svg_fuel = load_svg("fuel.svg")

    st.markdown(clean_html(f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div>
                    <div class="metric-card-header">
                        {svg_car}
                        <div class="metric-card-label">Listings Found</div>
                    </div>
                    <div class="metric-card-value">{filtered_count:,}</div>
                </div>
                <div class="metric-card-trend-badge">↑ {perc_filtered:.1f}% Market</div>
            </div>
            <div class="metric-card">
                <div>
                    <div class="metric-card-header">
                        {svg_price}
                        <div class="metric-card-label">Median Price</div>
                    </div>
                    <div class="metric-card-value">{median_p_str}</div>
                </div>
            </div>
            <div class="metric-card">
                <div>
                    <div class="metric-card-header">
                        {svg_calendar}
                        <div class="metric-card-label">Avg Age</div>
                    </div>
                    <div class="metric-card-value">{avg_age_str}</div>
                </div>
            </div>
            <div class="metric-card">
                <div>
                    <div class="metric-card-header">
                        {svg_mileage}
                        <div class="metric-card-label">Median Mileage</div>
                    </div>
                    <div class="metric-card-value">{median_m_str}</div>
                </div>
            </div>
            <div class="metric-card">
                <div>
                    <div class="metric-card-header">
                        {svg_brand}
                        <div class="metric-card-label">Top Brand</div>
                    </div>
                    <div class="metric-card-value">{top_brand}</div>
                </div>
            </div>
            <div class="metric-card">
                <div>
                    <div class="metric-card-header">
                        {svg_fuel}
                        <div class="metric-card-label">Top Fuel</div>
                    </div>
                    <div class="metric-card-value">{top_fuel}</div>
                </div>
            </div>
        </div>
    """), unsafe_allow_html=True)

    st.markdown("---")

    if filtered_df.empty:
        st.warning("No vehicles match the selected criteria. Please adjust your filters.")
        return

    # Define tabs for the 5 strategic perspectives
    dash_tabs = st.tabs([
        ":material/payments: Budget & Availability",
        ":material/verified_user: Value Retention & Fuel Types",
        ":material/settings: Transmission & Engine Specs",
        ":material/grid_view: Features & Comfort",
        ":material/map: Regional Price Distribution"
    ])

    # --- Perspective 1: Budget & Availability ---
    with dash_tabs[0]:
        st.header(":material/payments: Budget & Availability")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Price Distribution (Lakhs LKR)")
            
            # --- 1. Base histogram ---
            fig1 = px.histogram(filtered_df, x='price', nbins=50, 
                                title="Market Volume (with Smooth Density Line)",
                                labels={'price': 'Price (Lakhs LKR)'},
                                color_discrete_sequence=['#1f77b4'],
                                marginal="box")
            
            # --- 2. Manual KDE calculation for the "smooth line" ---
            try:
                prices = filtered_df['price'].dropna().values
                if len(prices) > 1:
                    kde = gaussian_kde(prices)
                    x_range = np.linspace(prices.min(), prices.max(), 100)
                    y_kde = kde(x_range)
                    
                    # Normalize the KDE and scale by (nb_samples * bin_width)
                    bin_width = (prices.max() - prices.min()) / 50
                    y_kde_scaled = y_kde * len(prices) * bin_width
                    
                    fig1.add_trace(go.Scatter(x=x_range, y=y_kde_scaled, 
                                              mode='lines', name='Smooth Line',
                                              line=dict(color='orange', width=3)))
            except Exception:
                pass # Skip if KDE fails due to too few points
                
            fig1.update_layout(title_x=0.5, title_xanchor='center')
            st.plotly_chart(fig1, width="stretch")
            
            # --- Dynamic Interpretation: Phase 1 Histogram ---
            global_median = df['price'].median()
            filtered_median = filtered_df['price'].median()
            diff_perc = ((filtered_median - global_median) / global_median) * 100
            direction = "above" if diff_perc > 0 else "below"
            
            st.markdown(clean_html(f"""
                <div style="background-color: var(--secondary-background-color); color: var(--text-color); padding: 10px; border-radius: 5px; border-left: 5px solid #1f77b4; font-size: 0.9rem;">
                    <b>Market Insight:</b> Your current selection has a median price of <b>{filtered_median:.1f}L</b>, 
                    which is <b>{abs(diff_perc):.1f}% {direction}</b> the overall market average of {global_median:.1f}L.
                </div>
            """), unsafe_allow_html=True)

        with col2:
            st.subheader("Market Share by Brand (Top 10 Brands)")
            brand_counts = filtered_df['brand'].value_counts().head(10).reset_index()
            fig2 = px.pie(brand_counts, values='count', names='brand', hole=0.5,
                          title="Listing Volume by Brand",
                          color_discrete_sequence=px.colors.qualitative.Pastel)
            fig2.update_layout(title_x=0.5, title_xanchor='center')
            st.plotly_chart(fig2, width="stretch")
            
            # --- Dynamic Interpretation: Phase 1 Pie ---
            if not brand_counts.empty:
                top_brand = brand_counts.iloc[0]['brand']
                top_share = (brand_counts.iloc[0]['count'] / brand_counts['count'].sum()) * 100
                st.markdown(clean_html(f"""
                    <div style="background-color: var(--secondary-background-color); color: var(--text-color); padding: 10px; border-radius: 5px; border-left: 5px solid #ff7f0e; font-size: 0.9rem;">
                        <b>Brand Dominance:</b> <b>{top_brand}</b> leads this segment with <b>{top_share:.1f}%</b> of your current filter volume.
                    </div>
                """), unsafe_allow_html=True)

        st.subheader("Price vs. Mileage (Value Decay)")
        try:
            # 1. Compute Pearson correlation dynamically for each category and overall
            cat_definitions = {
                'Old': 'YOM < 2010',
                'Intermediate': '2010 ≤ YOM ≤ 2017',
                'Modern': 'YOM ≥ 2018'
            }
            
            category_labels = {}
            valid_corrs = {}
            
            for cat_name, cat_def in cat_definitions.items():
                cat_data = filtered_df[filtered_df['year_category'] == cat_name]
                if len(cat_data) > 1 and cat_data['millage_km'].nunique() > 1 and cat_data['price'].nunique() > 1:
                    r = cat_data['millage_km'].corr(cat_data['price'], method='pearson')
                    if not pd.isna(r):
                        valid_corrs[cat_name] = r
                        category_labels[cat_name] = f"{cat_name} ({cat_def}, r = {r:.2f})"
                        continue
                category_labels[cat_name] = f"{cat_name} ({cat_def}, r = N/A)"
            
            # Map the categories in a copy of the filtered dataframe for plotting
            plot_df = filtered_df.copy()
            plot_df['year_category_legend'] = plot_df['year_category'].map(category_labels)
            
            # Category orders to enforce Old, Intermediate, Modern ordering in the legend
            cat_order = [category_labels[c] for c in ['Old', 'Intermediate', 'Modern']]
            
            # Try to plot with OLS trendline
            try:
                fig3 = px.scatter(plot_df, x='millage_km', y='price', color='year_category_legend',
                                  trendline="ols", title="How Mileage Affects Asking Price",
                                  labels={'millage_km': 'Mileage (km)', 'price': 'Price (Lakhs LKR)', 'year_category_legend': 'Year Category'},
                                  category_orders={'year_category_legend': cat_order},
                                  hover_data=['brand', 'yom'])
            except Exception:
                # Fall back to plotting without trendline if OLS fails (e.g. not enough data points in one category)
                fig3 = px.scatter(plot_df, x='millage_km', y='price', color='year_category_legend',
                                  title="How Mileage Affects Asking Price",
                                  labels={'millage_km': 'Mileage (km)', 'price': 'Price (Lakhs LKR)', 'year_category_legend': 'Year Category'},
                                  category_orders={'year_category_legend': cat_order},
                                  hover_data=['brand', 'yom'])
                                  
            fig3.update_layout(title_x=0.5, title_xanchor='center')
            st.plotly_chart(fig3, width="stretch")
            
            # Calculate overall Pearson correlation
            if len(filtered_df) > 1 and filtered_df['millage_km'].nunique() > 1 and filtered_df['price'].nunique() > 1:
                overall_r = filtered_df['millage_km'].corr(filtered_df['price'], method='pearson')
            else:
                overall_r = np.nan
                
            # Generate the dynamic summary
            if valid_corrs:
                strongest_cat = max(valid_corrs.keys(), key=lambda k: abs(valid_corrs[k]))
                strongest_r = valid_corrs[strongest_cat]
                weakest_cat = min(valid_corrs.keys(), key=lambda k: abs(valid_corrs[k]))
                weakest_r = valid_corrs[weakest_cat]
                
                # Check relationship direction for strongest and weakest
                if 'Old' in valid_corrs and 'Modern' in valid_corrs:
                    old_abs = abs(valid_corrs['Old'])
                    modern_abs = abs(valid_corrs['Modern'])
                    if old_abs > modern_abs:
                        intro = "Mileage is a stronger predictor of price for <b>older</b> vehicles than for <b>newer</b> vehicles."
                    elif old_abs < modern_abs:
                        intro = "Mileage is a stronger predictor of price for <b>newer</b> vehicles than for <b>older</b> vehicles."
                    else:
                        intro = "Mileage has a similar predictive strength for price on older and newer vehicles."
                else:
                    intro = f"Mileage is a stronger predictor of price in the <b>{strongest_cat}</b> category than in other categories."
                    
                body = f"The strongest relationship is observed in the <b>{strongest_cat}</b> category (r = <b>{strongest_r:.2f}</b>), while the <b>{weakest_cat}</b> category shows the weakest relationship (r = <b>{weakest_r:.2f}</b>)."
            else:
                intro = "There is insufficient data to evaluate the relationship between mileage and price across categories."
                body = ""
                
            overall_str = ""
            if not pd.isna(overall_r):
                direction = "lower" if overall_r < 0 else "higher"
                overall_str = f"Overall, higher mileage is associated with <b>{direction}</b> asking prices (overall r = <b>{overall_r:.2f}</b>)."
            
            insight_text = f"{intro} {body} {overall_str}".strip()
            
            st.markdown(clean_html(f"""
                <div style="background-color: var(--secondary-background-color); color: var(--text-color); padding: 10px; border-radius: 5px; border-left: 5px solid #2ca02c; font-size: 0.9rem;">
                    <b>Value Decay:</b> {insight_text}
                    <div style="margin-top: 6px; font-size: 0.78rem; color: #a1a1aa; font-style: italic;">Note: Pearson correlation (r) is used here to align with the OLS trendlines shown above.</div>
                </div>
            """), unsafe_allow_html=True)
            
        except Exception as e:
            fig3 = px.scatter(filtered_df, x='millage_km', y='price', color='year_category',
                              title="How Mileage Affects Asking Price",
                              labels={'millage_km': 'Mileage (km)', 'price': 'Price (Lakhs LKR)'},
                              hover_data=['brand', 'yom'])
            fig3.update_layout(title_x=0.5, title_xanchor='center')
            st.plotly_chart(fig3, width="stretch")

    # --- Perspective 2: Reliability & Value Retention ---
    with dash_tabs[1]:
        st.header(":material/verified_user: Value Retention & Fuel Types")
        
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Median Depreciation by Top Brands")
            top_5_brands = filtered_df['brand'].value_counts().head(5).index
            if not top_5_brands.empty:
                dep_df = filtered_df[filtered_df['brand'].isin(top_5_brands)].groupby(['brand', 'age'])['price'].median().reset_index()
                fig4 = px.line(dep_df, x='age', y='price', color='brand',
                               title="Median Price Retention by Vehicle Age",
                               labels={'age': 'Years Since Manufacture', 'price': 'Median Price (Lakhs)'})
                fig4.update_layout(title_x=0.5, title_xanchor='center')
                st.plotly_chart(fig4, width="stretch")
                
                # --- Dynamic Interpretation: Phase 2 Line ---
                try:
                    age_diff = dep_df['age'].max() - dep_df['age'].min()
                    if age_diff > 0:
                        avg_yearly_drop = (dep_df.groupby('brand')['price'].first() - dep_df.groupby('brand')['price'].last()).mean() / age_diff
                        st.markdown(clean_html(f"""
                            <div style="background-color: var(--secondary-background-color); color: var(--text-color); padding: 10px; border-radius: 5px; border-left: 5px solid #d62728; font-size: 0.9rem;">
                                <b>Depreciation:</b> On average, vehicles in this selection lose approx. <b>{abs(avg_yearly_drop):.2f}L</b> in listing value per year of age.
                            </div>
                        """), unsafe_allow_html=True)
                    else:
                        st.markdown(clean_html(f"""
                            <div style="background-color: var(--secondary-background-color); color: var(--text-color); padding: 10px; border-radius: 5px; border-left: 5px solid #d62728; font-size: 0.9rem;">
                                <b>Depreciation:</b> Insufficient age span to calculate average yearly depreciation.
                            </div>
                        """), unsafe_allow_html=True)
                except Exception:
                    pass
            else:
                st.write("Not enough brand data to show depreciation.")

        with col4:
            st.subheader("Median Price by Fuel Type")
            fuel_df = filtered_df.groupby('fuel_type')['price'].median().sort_values(ascending=False).reset_index()
            fig5 = px.bar(fuel_df, x='fuel_type', y='price', color='fuel_type',
                          title="Median Price for Fuel Categories",
                          labels={'fuel_type': 'Fuel Type', 'price': 'Median Price (Lakhs)'})
            fig5.update_layout(title_x=0.5, title_xanchor='center')
            st.plotly_chart(fig5, width="stretch")
            
            # --- Dynamic Interpretation: Phase 2 Fuel ---
            if len(fuel_df) > 1:
                most_exp = fuel_df.iloc[0]
                least_exp = fuel_df.iloc[-1]
                ratio = most_exp['price'] / least_exp['price'] if least_exp['price'] > 0 else 0
                st.markdown(clean_html(f"""
                    <div style="background-color: var(--secondary-background-color); color: var(--text-color); padding: 10px; border-radius: 5px; border-left: 5px solid #9467bd; font-size: 0.9rem;">
                        <b>Fuel Variance:</b> <b>{most_exp['fuel_type']}</b> vehicles are currently listed <b>{ratio:.1f}x</b> higher than <b>{least_exp['fuel_type']}</b> options.
                    </div>
                """), unsafe_allow_html=True)

    # --- Perspective 3: Performance & Technical Specs ---
    with dash_tabs[2]:
        st.header(":material/settings: Transmission & Engine Specs")
        
        col5, col6 = st.columns(2)

        with col5:
            st.subheader("Transmission Price Premium")
            fig6 = px.box(filtered_df, x='gear', y='price', color='gear',
                          title="Manual vs. Automatic Price Spread",
                          labels={'gear': 'Transmission Type', 'price': 'Price (Lakhs LKR)'})
            fig6.update_layout(title_x=0.5, title_xanchor='center')
            st.plotly_chart(fig6, width="stretch")
            
            # --- Dynamic Interpretation: Phase 3 Transmission ---
            try:
                auto_m = filtered_df[filtered_df['gear'] == 'Automatic']['price'].median()
                manual_m = filtered_df[filtered_df['gear'] == 'Manual']['price'].median()
                if not pd.isna(auto_m) and not pd.isna(manual_m):
                    auto_premium = auto_m - manual_m
                    st.markdown(clean_html(f"""
                        <div style="background-color: var(--secondary-background-color); color: var(--text-color); padding: 10px; border-radius: 5px; border-left: 5px solid #8c564b; font-size: 0.9rem;">
                            <b>Auto Premium:</b> Automatic transmissions currently carry a <b>{auto_premium:+.1f}L</b> median premium over manuals here.
                        </div>
                    """), unsafe_allow_html=True)
            except Exception:
                pass

        with col6:
            st.subheader("Engine Capacity vs. Price")
            try:
                seg_names = [
                    "Micro (<800cc)",
                    "Compact (800-1200cc)",
                    "Mid-Range (1200-1600cc)",
                    "Large (>1600cc)"
                ]
                
                segment_labels = {}
                valid_corrs_cc = {}
                
                for seg in seg_names:
                    seg_data = filtered_df[filtered_df['engine_segment'] == seg]
                    if len(seg_data) > 1 and seg_data['engine_cc'].nunique() > 1 and seg_data['price'].nunique() > 1:
                        r = seg_data['engine_cc'].corr(seg_data['price'], method='pearson')
                        if not pd.isna(r):
                            valid_corrs_cc[seg] = r
                            segment_labels[seg] = f"{seg[:-1]}, r = {r:.2f})"
                            continue
                    segment_labels[seg] = f"{seg[:-1]}, r = N/A)"
                
                # Copy for plotting
                plot_df_cc = filtered_df.copy()
                plot_df_cc['engine_segment_legend'] = plot_df_cc['engine_segment'].map(segment_labels)
                
                # Ordering of engine segments in legend
                seg_order = [segment_labels[s] for s in seg_names]
                
                try:
                    fig7 = px.scatter(plot_df_cc, x='engine_cc', y='price', color='engine_segment_legend',
                                      trendline="ols", title="Engine Size Impact on Valuation",
                                      labels={'engine_cc': 'Engine Size (cc)', 'price': 'Price (Lakhs LKR)', 'engine_segment_legend': 'Engine Segment'},
                                      category_orders={'engine_segment_legend': seg_order})
                except Exception:
                    fig7 = px.scatter(plot_df_cc, x='engine_cc', y='price', color='engine_segment_legend',
                                      title="Engine Size Impact on Valuation",
                                      labels={'engine_cc': 'Engine Size (cc)', 'price': 'Price (Lakhs LKR)', 'engine_segment_legend': 'Engine Segment'},
                                      category_orders={'engine_segment_legend': seg_order})
                                      
                fig7.update_layout(title_x=0.5, title_xanchor='center')
                st.plotly_chart(fig7, width="stretch")
                
                # Calculate overall correlation
                if len(filtered_df) > 1 and filtered_df['engine_cc'].nunique() > 1 and filtered_df['price'].nunique() > 1:
                    overall_r_cc = filtered_df['engine_cc'].corr(filtered_df['price'], method='pearson')
                else:
                    overall_r_cc = np.nan
                
                # Dynamic summary generator
                if valid_corrs_cc:
                    strongest_seg = max(valid_corrs_cc.keys(), key=lambda k: abs(valid_corrs_cc[k]))
                    strongest_r_cc = valid_corrs_cc[strongest_seg]
                    weakest_seg = min(valid_corrs_cc.keys(), key=lambda k: abs(valid_corrs_cc[k]))
                    weakest_r_cc = valid_corrs_cc[weakest_seg]
                    
                    short_names = {
                        "Micro (<800cc)": "Micro",
                        "Compact (800-1200cc)": "Compact",
                        "Mid-Range (1200-1600cc)": "Mid-Range",
                        "Large (>1600cc)": "Large-engine"
                    }
                    
                    str_name = short_names.get(strongest_seg, strongest_seg)
                    wk_name = short_names.get(weakest_seg, weakest_seg)
                    
                    if abs(weakest_r_cc) < 0.1:
                        weakest_desc = f"show almost no relationship between engine size and price (r = <b>{weakest_r_cc:.2f}</b>)"
                    else:
                        weakest_desc = f"show the weakest relationship between engine size and price (r = <b>{weakest_r_cc:.2f}</b>)"
                        
                    body = f"The strongest association is observed in <b>{str_name}</b> vehicles (r = <b>{strongest_r_cc:.2f}</b>), while <b>{wk_name}</b> vehicles {weakest_desc}."
                else:
                    body = "There is insufficient data to compare relationship strengths across engine segments."
                
                overall_desc = ""
                if not pd.isna(overall_r_cc):
                    abs_r = abs(overall_r_cc)
                    strength = "strong" if abs_r > 0.6 else "moderate" if abs_r > 0.3 else "weak"
                    overall_desc = f"Overall, engine size alone is a <b>{strength}</b> predictor of vehicle price (r = <b>{overall_r_cc:.2f}</b>)."
                
                insight_text_cc = f"The relationship between engine size and price varies across engine segments. {body} {overall_desc}".strip()
                
                st.markdown(clean_html(f"""
                    <div style="background-color: var(--secondary-background-color); color: var(--text-color); padding: 10px; border-radius: 5px; border-left: 5px solid #e377c2; font-size: 0.9rem;">
                        <b>Engine Power:</b> {insight_text_cc}
                        <div style="margin-top: 6px; font-size: 0.78rem; color: #a1a1aa; font-style: italic;">Note: Pearson correlation (r) is used here to align with the OLS trendlines shown above.</div>
                    </div>
                """), unsafe_allow_html=True)
                
            except Exception as e:
                fig7 = px.scatter(filtered_df, x='engine_cc', y='price', color='engine_segment',
                                  title="Engine Size Impact on Valuation",
                                  labels={'engine_cc': 'Engine Size (cc)', 'price': 'Price (Lakhs LKR)'})
                fig7.update_layout(title_x=0.5, title_xanchor='center')
                st.plotly_chart(fig7, width="stretch")

    # --- Perspective 4: Features & Comfort ---
    with dash_tabs[3]:
        st.header(":material/grid_view: Features & Comfort")
        
        col7, col8 = st.columns(2)

        with col7:
            st.subheader("Feature Premium Analysis")
            features = ['air_condition', 'power_steering', 'power_mirror', 'power_window']
            feature_impact = []
            for feature in features:
                if feature in filtered_df.columns:
                    with_f = filtered_df[filtered_df[feature] == True]['price'].median()
                    without_f = filtered_df[filtered_df[feature] == False]['price'].median()
                    f_label = feature.replace('_', ' ').title()
                    if not pd.isna(with_f) or not pd.isna(without_f):
                        feature_impact.append({'Feature': f_label, 'Status': 'With', 'Median Price': np.nan_to_num(with_f)})
                        feature_impact.append({'Feature': f_label, 'Status': 'Without', 'Median Price': np.nan_to_num(without_f)})
            
            if feature_impact:
                fig8 = px.bar(pd.DataFrame(feature_impact), x='Feature', y='Median Price', color='Status', barmode='group',
                              title="Impact of Standard Features on Price",
                              labels={'Median Price': 'Median Price (Lakhs LKR)'})
                fig8.update_layout(title_x=0.5, title_xanchor='center')
                st.plotly_chart(fig8, width="stretch")
                
                # --- Dynamic Interpretation: Phase 4 Lift ---
                f_df = pd.DataFrame(feature_impact)
                f_df['Diff'] = f_df.groupby('Feature')['Median Price'].transform(lambda x: x.max() - x.min())
                sorted_f_df = f_df.sort_values(by='Diff', ascending=False)
                top_f = sorted_f_df.iloc[0]['Feature']
                top_v = sorted_f_df.iloc[0]['Diff']
                st.markdown(clean_html(f"""
                    <div style="background-color: var(--secondary-background-color); color: var(--text-color); padding: 10px; border-radius: 5px; border-left: 5px solid #bcbd22; font-size: 0.9rem;">
                        <b>Feature Lift:</b> The presence of <b>{top_f}</b> currently adds the highest median value (approx. <b>+{top_v:.1f}L</b>) to these listings.
                    </div>
                """), unsafe_allow_html=True)

        with col8:
            st.subheader("Modernity vs. Features (Availability)")
            try:
                heatmap_data = []
                available_years = sorted(filtered_df['yom'].unique())
                plot_years = available_years[-15:] if len(available_years) > 15 else available_years
                
                for year in plot_years:
                    year_df = filtered_df[filtered_df['yom'] == year]
                    row = {'Year': str(int(year))}
                    for feature in features:
                        if feature in filtered_df.columns:
                            penetration = (year_df[feature].sum() / len(year_df)) * 100 if len(year_df) > 0 else 0
                            row[feature.replace('_', ' ').title()] = penetration
                    heatmap_data.append(row)
                
                if heatmap_data:
                    fig9 = px.imshow(pd.DataFrame(heatmap_data).set_index('Year').T,
                                     labels=dict(x="Year of Manufacture", y="Feature", color="% Available"),
                                     title="Feature Standardisation Over Time",
                                     color_continuous_scale="Viridis")
                    fig9.update_layout(title_x=0.5, title_xanchor='center')
                    st.plotly_chart(fig9, width="stretch")
                    
                    # --- Dynamic Interpretation: Phase 4 Heatmap ---
                    h_df = pd.DataFrame(heatmap_data)
                    latest_y = h_df['Year'].max()
                    earliest_y = h_df['Year'].min()
                    st.markdown(clean_html(f"""
                        <div style="background-color: var(--secondary-background-color); color: var(--text-color); padding: 10px; border-radius: 5px; border-left: 5px solid #17becf; font-size: 0.9rem;">
                            <b>Standardization:</b> We see a clear move towards feature inclusion (Yellow) from <b>{earliest_y}</b> to <b>{latest_y}</b> in this selection.
                        </div>
                    """), unsafe_allow_html=True)
            except Exception:
                pass

    # --- Perspective 5: Geographic Price Volatility ---
    with dash_tabs[4]:
        st.header(":material/map: Regional Price Distribution")
        
        st.subheader("Median Price by Province")
        if not filtered_df.empty:
            province_df = filtered_df.groupby('province')['price'].agg(['median', 'count']).reset_index().sort_values('median', ascending=False)
            
            # --- Load GeoJSON for Sri Lanka provinces ---
            geojson_data = load_geojson()
            
            # --- Map data province names to GeoJSON shapeName ---
            province_name_map = {
                'Western':      'Western Province',
                'Eastern':      'Eastern Province',
                'Northern':     'Northern Province',
                'Southern':     'Southern Province',
                'Central':      'Central Province',
                'North Western': 'North Western Province',
                'North Central': 'North Central Province',
                'Uva':          'Uva Province',
                'Sabaragamuwa': 'Sabaragamuwa Province',
                'Other':        None,  # Skip unmapped
            }
            
            province_df['geojson_name'] = province_df['province'].map(province_name_map)
            map_df = province_df[province_df['geojson_name'].notna()].copy()
            
            # Create a dictionary for quick lookup during geojson rendering
            province_data = map_df.set_index("geojson_name").to_dict(orient="index")
            
            min_val = map_df['median'].min() if not map_df.empty else 0
            max_val = map_df['median'].max() if not map_df.empty else 100
            
            # Define a beautiful, premium color scale (OrRd: Orange to Red gradient)
            colormap = cm.LinearColormap(
                colors=['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15'],
                vmin=min_val,
                vmax=max_val,
                caption="Median Price (Lakhs LKR)"
            )

            def style_function(feature):
                shape_name = feature['properties']['shapeName']
                data = province_data.get(shape_name, {})
                median_val = data.get('median', np.nan)
                return {
                    'fillColor': colormap(median_val) if not pd.isna(median_val) else '#ffffff',
                    'color': '#ffffff',
                    'weight': 1.5,
                    'fillOpacity': 0.75
                }

            def highlight_function(feature):
                return {
                    'fillColor': '#4f6ef7',
                    'color': '#ffffff',
                    'weight': 2.5,
                    'fillOpacity': 0.85
                }

            # Inject calculated stats directly into GeoJSON properties for the tooltip
            geojson_data_copy = copy.deepcopy(geojson_data)
            for feature in geojson_data_copy['features']:
                shape_name = feature['properties']['shapeName']
                data = province_data.get(shape_name, {})
                feature['properties']['median_price'] = f"{data.get('median', 0.0):.1f} Lakhs LKR" if 'median' in data else "No Listings"
                feature['properties']['listings_count'] = f"{data.get('count', 0):,}" if 'count' in data else "0"

            # Create Folium Map with clean cartodbpositron tiles locked to Sri Lanka bounds
            m = folium.Map(
                location=[7.8731, 80.7718],
                zoom_start=6.8,
                tiles="cartodbpositron",
                width="100%",
                height="100%",
                zoom_control=True,
                min_zoom=6,
                max_zoom=11,
                max_bounds=[[5.0, 78.0], [10.5, 83.5]]
            )

            # Add colormap legend to the map
            colormap.add_to(m)

            # Add GeoJson layer with custom styling & hover tooltips
            folium.GeoJson(
                geojson_data_copy,
                style_function=style_function,
                highlight_function=highlight_function,
                tooltip=folium.GeoJsonTooltip(
                    fields=['shapeName', 'median_price', 'listings_count'],
                    aliases=['Province:', 'Median Price:', 'Listings:'],
                    localize=True,
                    sticky=True,
                    style="""
                        background-color: #1e1e24;
                        border: 1px solid #3f3f46;
                        border-radius: 8px;
                        padding: 10px 14px;
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                        font-size: 13px;
                        color: #f4f4f5;
                        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
                    """
                )
            ).add_to(m)

            # Wrap in custom css class for rounded corners and render in a centered column
            col_space_l, col_map_body, col_space_r = st.columns([2, 4, 2])
            with col_map_body:
                st.markdown('<div class="rounded-map-container">', unsafe_allow_html=True)
                st.iframe(m.get_root().render(), height=500)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # --- Dynamic Interpretation: Phase 5 Regional ---
            if not province_df.empty:
                top_p = province_df.iloc[0]['province']
                top_v = province_df.iloc[0]['median']
                low_p = province_df.iloc[-1]['province']
                low_v = province_df.iloc[-1]['median']
                diff = top_v - low_v
                st.markdown(clean_html(f"""
                    <div style="background-color: var(--secondary-background-color); color: var(--text-color); padding: 10px; border-radius: 5px; border-left: 5px solid #d62728; font-size: 0.9rem;">
                        <b>Provincial Variation:</b> <b>{top_p}</b> leads as the highest-priced region (median <b>{top_v:.1f}L</b>), 
                        while <b>{low_p}</b> is the most affordable at <b>{low_v:.1f}L</b> with a spread of <b>{diff:.1f}L</b> across provinces.
                    </div>
                """), unsafe_allow_html=True)

    # Global summary footer
    st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
    st.info(f"Analysis focusing on {len(filtered_df):,} vehicles based on your current filter settings.")

