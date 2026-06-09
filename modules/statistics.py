import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px

from utils.style_loader import clean_html
def app(df):
    """
    Renders the Statistical Validation page of the AutoPulse.AI web application.
    Executes Kruskal-Wallis, Mann-Whitney U, Spearman rank correlations, Levene's risk checks,
    and Cramér's V associations to validate Sri Lankan vehicle market data attributes.

    Parameters:
        df (pd.DataFrame): The preprocessed vehicle listings dataset.
    """
    st.title("Statistical Tests & Validation")
    st.markdown("""
        Moving beyond simple averages, this module uses rigorous statistical tests to 
        validate market "common knowledge" and identify true value drivers.
    """)

    remove_outliers = st.checkbox("Remove price outliers", value=False, key="stat_outliers")
    if remove_outliers:
        df = df[df['price'] <= df['price'].quantile(0.95)]

    tabs = st.tabs([
        ":material/foundation: Brand Value & Depreciation", 
        ":material/settings: Engine & Fuel Volatility", 
        ":material/grid_view: Features & Association"
    ])

    # --- Brand Value & Depreciation ---
    with tabs[0]:
        st.header("Brand Value & Depreciation")
        
        # --- 1. Price distributions by brand ---
        st.subheader("1. Price Distributions Differ by Brand")
        st.caption("Methodology: **Kruskal-Wallis H-Test** (Non-parametric Variance Analysis)")
        top_10_brands = df['brand'].value_counts().head(10).index
        brand_groups = [df[df['brand'] == b]['price'].values for b in top_10_brands]
        h_stat, p_val = stats.kruskal(*brand_groups)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("**Kruskal-Wallis H-Test**")
            render_stat_card("Brand Price Variance", h_stat, p_val)
            st.markdown('<div style="margin-top: -1rem;"></div>', unsafe_allow_html=True)
            st.info("**User Value:** Confirms that brand choice is a primary driver of price in Sri Lanka, as distributions are significantly varied.")
        
        with col2:
            fig_kv = px.box(df[df['brand'].isin(top_10_brands)], x='brand', y='price', color='brand',
                               points="all", title="Price Spread across Top 10 Brands")
            fig_kv.update_layout(title_x=0.5, title_xanchor='center')
            st.plotly_chart(fig_kv, width="stretch")

        st.divider()

        # --- 2. Interactive brand benchmarking ---
        st.subheader("2. Brand Benchmarking")
        st.caption("Methodology: **Mann-Whitney U Test** (Independent Sample Comparison)")
        st.write("Compare the price distributions of any two brands.")
        
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            brand_a = st.selectbox("Select Benchmark Brand", sorted(df['brand'].unique()), index=list(sorted(df['brand'].unique())).index('TOYOTA'))
        with b_col2:
            brand_b = st.selectbox("Select Comparison Brand", sorted(df['brand'].unique()), index=list(sorted(df['brand'].unique())).index('SUZUKI'))
        
        group_a = df[df['brand'] == brand_a]['price']
        group_b = df[df['brand'] == brand_b]['price']
        
        if not group_a.empty and not group_b.empty:
            u_stat, p_u = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
            diff = group_a.median() - group_b.median()
            
            m_col1, m_col2 = st.columns([1, 2])
            with m_col1:
                render_stat_card(f"{brand_a} vs {brand_b}", u_stat, p_u)
                st.markdown('<div style="margin-top: -1rem;"></div>', unsafe_allow_html=True)
                st.write(f"**Median Difference:** {diff:+.2f} Lakhs")
                if p_u < 0.05:
                    st.success(f"**Result:** {brand_a} is priced significantly differently than {brand_b}.")
                else:
                    st.warning(f"**Result:** No statistically significant price difference found between these two.")
            
            with m_col2:
                fig_comp = px.histogram(df[df['brand'].isin([brand_a, brand_b])], x='price', color='brand', barmode='overlay',
                                       marginal='box', title=f"{brand_a} vs {brand_b}")
                fig_comp.update_layout(title_x=0.5, title_xanchor='center')
                st.plotly_chart(fig_comp, width="stretch")

        st.divider()

        # --- 3. Age & performance correlation ---
        st.subheader("3. True Impact of Age & Performance")
        st.write("Spearman Correlation ($ρ$) measures how strictly two variables move together. Filter by brand to see specific depreciation trends.")
        
        selected_brands = st.multiselect("Filter Brands for Analysis", sorted(df['brand'].unique()), default=['TOYOTA', 'SUZUKI', 'NISSAN'])
        
        stat_df = df[df['brand'].isin(selected_brands)] if selected_brands else df
        
        if not stat_df.empty:
            if stat_df['age'].nunique() > 1 and stat_df['price'].nunique() > 1:
                corr_age, p_age = stats.spearmanr(stat_df['age'], stat_df['price'])
                
                c_col1, c_col2 = st.columns([1, 2])
                with c_col1:
                    st.metric("Price vs. Age Correlation", f"{corr_age:.3f}", 
                              delta="Strong Negative" if corr_age < -0.7 else "Moderate Negative", 
                              help="Value loss over time for selected brands")
                    st.info(f"**Interpretation:** For the selected brands, vehicles lose value at a rate of {abs(corr_age):.1%} consistency relative to their age.")
                
                with c_col2:
                    fig_age_scatter = px.scatter(stat_df, x='age', y='price', color='brand',
                                               trendline="ols", title="Price vs. Age (Depreciation Curve)",
                                               labels={'age': 'Vehicle Age (Years)', 'price': 'Price (Lakhs LKR)'})
                    fig_age_scatter.update_layout(title_x=0.5, title_xanchor='center')
                    st.plotly_chart(fig_age_scatter, width="stretch")
            else:
                st.warning("Insufficient variance in vehicle age or price to calculate Spearman correlation.")
        else:
            st.warning("Please select at least one brand to view the depreciation analysis.")

    # --- Engine & Fuel Volatility ---
    with tabs[1]:
        st.header("Engine & Fuel Volatility")
        st.markdown("Discover how engine size and fuel technology change value in different price brackets.")
        
        # --- 1. Engine size vs price analysis ---
        st.subheader("1. Engine Size vs Price")
        st.write("Spearman Correlation of Price vs. Engine Capacity, grouped by Vehicle Segment.")
        
        segments = sorted(df['engine_segment'].unique())
        seg_corrs = []
        for seg in segments:
            seg_df = df[df['engine_segment'] == seg]
            if len(seg_df) > 5 and seg_df['engine_cc'].nunique() > 1 and seg_df['price'].nunique() > 1:
                corr, p = stats.spearmanr(seg_df['engine_cc'], seg_df['price'])
                if not pd.isna(corr):
                    seg_corrs.append({'Segment': seg, 'Correlation': corr, 'p-value': p})
        
        if seg_corrs:
            sc_df = pd.DataFrame(seg_corrs)
            sc_col1, sc_col2 = st.columns([1, 2])
            with sc_col1:
                st.write("**Price Sensitivity to Engine Size by Vehicle Segment**")
                for _, row in sc_df.iterrows():
                    st.write(f"- **{row['Segment']}**: {row['Correlation']:.3f} ($ρ$)")
            
            with sc_col2:
                fig_seg_corr = px.bar(sc_df, x='Segment', y='Correlation', color='Segment',
                                    title="Sensitivity to Price by Engine Size",
                                    labels={'Correlation': 'Spearman Correlation (ρ)'})
                fig_seg_corr.update_layout(title_x=0.5, title_xanchor='center')
                st.plotly_chart(fig_seg_corr, width="stretch")
                
        st.divider()

        # --- 2. Price volatility by fuel type ---
        st.subheader("2. Price Volatility by Fuel Type")
        st.write("Testing the variance (spread) of prices across all fuel types using Levene's Test.")
        
        valid_fuel_groups = {name: group['price'] for name, group in df.groupby('fuel_type') if len(group) > 5}
        
        if len(valid_fuel_groups) >= 2:
            stat_l, p_l = stats.levene(*[v.values for v in valid_fuel_groups.values()])
            
            l_col1, l_col2 = st.columns([1, 2])
            with l_col1:
                render_stat_card("Price Volatility (Levene)", stat_l, p_l)
                st.markdown('<div style="margin-top: -1rem;"></div>', unsafe_allow_html=True)
                if p_l < 0.05:
                    variances = {name: vals.var() for name, vals in valid_fuel_groups.items()}
                    most_volatile = max(variances, key=variances.get)
                    least_volatile = min(variances, key=variances.get)
                    st.error(f"**Finding:** There is a significant difference in price volatility. **{most_volatile}** vehicles have the most variable prices, while **{least_volatile}** prices are more stable.")
                else:
                    st.warning("**Finding:** No significant difference in volatility detected between fuel types.")
            
            with l_col2:
                valid_fuels = list(valid_fuel_groups.keys())
                fig_risk = px.box(df[df['fuel_type'].isin(valid_fuels)], x='fuel_type', y='price', 
                                    color='fuel_type', points="all",
                                    title="Breadth of Price Ranges")
                fig_risk.update_layout(title_x=0.5, title_xanchor='center')
                st.plotly_chart(fig_risk, width="stretch")

    # --- Features & Association ---
    with tabs[2]:
        st.header("Features & Association")
        
        # --- 1. Feature price multipliers ---
        st.subheader("1. Comfort features as 'Price Multipliers'")
        st.caption("Methodology: **Mann-Whitney U Test** (One-tailed Binary Impact Analysis)")
        features = ['air_condition', 'power_steering', 'power_mirror', 'power_window']
        
        feature_stats = []
        for feat in features:
            with_f = df[df[feat] == True]['price']
            without_f = df[df[feat] == False]['price']
            u, p = stats.mannwhitneyu(with_f, without_f, alternative='greater')
            lift = with_f.median() - without_f.median()
            feature_stats.append({'Feature': feat.replace('_', ' ').title(), 'p-value': p, 'Value Lift': lift})
        
        st.write("Are these extras worth it? We tested if their presence significantly raises median price.")
        f_cols = st.columns(4)
        for i, f_res in enumerate(feature_stats):
            with f_cols[i]:
                is_sig = f_res['p-value'] < 0.05
                st.metric(f_res['Feature'], f"+{f_res['Value Lift']:.1f}L", delta="Significant" if is_sig else "Not Sig", delta_color="normal" if is_sig else "off")
        
        st.divider()

        # --- 2. Feature bundling (Cramér's V) ---
        st.subheader("2. How Features are Bundled (Cramér's V)")
        st.write("Cramér's V determines the strength of association between binary features (0 = independent, 1 = perfectly bundled).")
        
        def calculate_cramers_v(x, y):
            confusion_matrix = pd.crosstab(x, y)
            chi2 = stats.chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            phi2 = chi2 / n
            r, k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

        v_matrix = pd.DataFrame(index=features, columns=features)
        for f1 in features:
            for f2 in features:
                v_matrix.loc[f1, f2] = calculate_cramers_v(df[f1], df[f2])
        
        v_matrix = v_matrix.astype(float)
        fig_heart = px.imshow(v_matrix, text_auto=".2f", color_continuous_scale="RdPu",
                             title="Cramér's V: Feature Association Heatmap",
                             labels=dict(x="Feature A", y="Feature B", color="Association"))
        fig_heart.update_layout(title_x=0.5, title_xanchor='center')
        st.plotly_chart(fig_heart, width="stretch")
        
        st.info("**Insight:** High association (e.g., Power Windows + Mirrors) suggests these features almost always appear together in listings.")

def render_stat_card(title, statistic, p_value):
    is_significant = p_value < 0.05
    color = "var(--secondary-background-color)"
    border = "#1f77b4" if is_significant else "#ff7f0e"
    text_color = "var(--text-color)"
    
    # SVG icons for the statistics cards
    check_svg = '<svg xmlns="http://www.w3.org/2000/svg" height="20" viewBox="0 -960 960 960" width="20" fill="#28a745" style="vertical-align: middle; margin-right: 5px;"><path d="M382-240 154-468l57-57 171 171 367-367 57 57-424 424Z"/></svg>'
    warn_svg = '<svg xmlns="http://www.w3.org/2000/svg" height="20" viewBox="0 -960 960 960" width="20" fill="#dc3545" style="vertical-align: middle; margin-right: 5px;"><path d="M480-280q17 0 28.5-11.5T520-320q0-17-11.5-28.5T480-360q-17 0-28.5 11.5T440-320q0 17 11.5 28.5T480-280Zm-40-160h80v-240h-80v240Zm40 360q-83 0-156-31.5T197-197q-54-54-85.5-127T80-480q0-83 31.5-156T197-763q54-54 127-85.5T480-880q83-0 156 31.5T763-763q54 54 85.5 127T880-480q0 83-31.5 156T763-197q-54 54-127 85.5T480-80Z"/></svg>'
    icon = check_svg if is_significant else warn_svg
    
    st.markdown(clean_html(f"""
        <div style="background-color: {color}; padding: 18px; border-radius: 12px; border-left: 5px solid {border}; color: {text_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <h5 style="margin: 0; display: flex; align-items: center;">{icon} {title}</h5>
            <div style="margin-top: 10px; font-size: 0.95rem;">
                <p style="margin: 2px 0;">Stat: <b>{statistic:.2f}</b></p>
                <p style="margin: 2px 0;">p-value: <b>{p_value:.4f}</b></p>
                <p style="margin: 8px 0 0 0; color: {'#28a745' if is_significant else '#dc3545'}; font-weight: bold; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px;">
                    {"✔ Statistically Significant" if is_significant else "✘ Not Significant"}
                </p>
            </div>
        </div>
    """), unsafe_allow_html=True)
