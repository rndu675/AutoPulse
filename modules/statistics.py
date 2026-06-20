import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px

from utils.style_loader import clean_html

def render_insight_panel(key_finding, strength, direction, metrics, bullets):
    """
    Renders a unified, styled insight card on the left panel of a statistical test section.
    """
    status = metrics.get('Status', 'Significant')
    is_sig = 'not' not in status.lower() and 'fail' not in status.lower()
    border_color = "#2ca02c" if is_sig else "#d62728"
    badge_bg = "rgba(44, 160, 44, 0.15)" if is_sig else "rgba(214, 39, 39, 0.15)"
    badge_fg = "#2ca02c" if is_sig else "#d62728"
    
    def format_bold_markdown(text):
        text = str(text)
        parts = text.split("**")
        new_parts = []
        for i, part in enumerate(parts):
            if i % 2 == 1:
                new_parts.append(f"<b>{part}</b>")
            else:
                new_parts.append(part)
        return "".join(new_parts)

    metrics_html = "".join([
        f"<div style='display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.9rem; border-bottom: 1px dashed rgba(255,255,255,0.08); padding-bottom: 4px;'>"
        f"<span style='color: #a1a1aa;'>{format_bold_markdown(k)}</span>"
        f"<span style='font-weight: bold; color: var(--text-color);'>{format_bold_markdown(v)}</span>"
        f"</div>"
        for k, v in metrics.items() if k != 'Status'
    ])
    
    bullets_html = "".join([
        f"<li style='margin-bottom: 6px; font-size: 0.9rem; color: var(--text-color);'>{format_bold_markdown(b)}</li>"
        for b in bullets
    ])
    
    strength_desc = f"<b>Strength:</b> {format_bold_markdown(strength)}"
    if direction:
        strength_desc += f" ({format_bold_markdown(direction)})"
        
    html = f"""
    <div style="background-color: var(--secondary-background-color); border-left: 5px solid {border_color}; padding: 20px; border-radius: 10px; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <h4 style="margin: 0; font-size: 1.15rem; color: var(--text-color); font-weight: 600;">Key Validation Insights</h4>
            <span style="background-color: {badge_bg}; color: {badge_fg}; padding: 3px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">
                {status}
            </span>
        </div>
        
        <div style="margin-bottom: 16px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 12px;">
            <p style="margin: 0; font-size: 0.95rem; line-height: 1.45; color: var(--text-color); font-weight: 500;">
                {format_bold_markdown(key_finding)}
            </p>
        </div>
        
        <div style="margin-bottom: 16px; background: rgba(255,255,255,0.03); padding: 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.05);">
            <div style="font-size: 0.85rem; color: #a1a1aa; margin-bottom: 4px;">Relationship Profile</div>
            <div style="font-size: 0.95rem; font-weight: bold; color: var(--text-color);">{strength_desc}</div>
        </div>
        
        <div style="margin-bottom: 16px;">
            <div style="font-size: 0.85rem; color: #a1a1aa; margin-bottom: 8px;">Statistical Metrics</div>
            {metrics_html}
        </div>
        
        <div>
            <div style="font-size: 0.85rem; color: #a1a1aa; margin-bottom: 8px;">Key Takeaways</div>
            <ul style="margin: 0; padding-left: 18px; line-height: 1.4; color: var(--text-color);">
                {bullets_html}
            </ul>
        </div>
    </div>
    """
    st.markdown(clean_html(html), unsafe_allow_html=True)

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
        
        # Calculate dynamic insights
        brand_medians = df[df['brand'].isin(top_10_brands)].groupby('brand')['price'].median().sort_values(ascending=False)
        highest_brand = brand_medians.index[0]
        highest_val = brand_medians.iloc[0]
        lowest_brand = brand_medians.index[-1]
        lowest_val = brand_medians.iloc[-1]
        median_spread = highest_val - lowest_val
        sample_size_kw = len(df[df['brand'].isin(top_10_brands)])
        
        col_kw1, col_kw2 = st.columns([2, 3])
        with col_kw1:
            finding_kw = "Vehicle asking prices in Sri Lanka vary significantly depending on the brand. Brand choice is confirmed as a primary driver of listing valuations."
            strength_kw = "Very Significant Variance" if p_val < 0.001 else "Significant Variance" if p_val < 0.05 else "No Significant Variance"
            metrics_kw = {
                'H-Statistic': f"{h_stat:.2f}",
                'p-value': f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}",
                'Brands Analyzed': f"{len(top_10_brands)}",
                'Listings Count': f"{sample_size_kw:,}",
                'Status': 'Significant' if p_val < 0.05 else 'Not Significant'
            }
            bullets_kw = [
                f"**{highest_brand}** holds the highest median price among the top brands at **{highest_val:.1f} Lakhs LKR**.",
                f"**{lowest_brand}** represents the most economical top brand with a median price of **{lowest_val:.1f} Lakhs LKR**.",
                f"The median price spread between the highest and lowest top brands is **{median_spread:.1f} Lakhs LKR**."
            ]
            render_insight_panel(finding_kw, strength_kw, None, metrics_kw, bullets_kw)
            
        with col_kw2:
            fig_kv = px.box(df[df['brand'].isin(top_10_brands)], x='brand', y='price', color='brand',
                               points="all", title="Price Spread across Top 10 Brands")
            fig_kv.update_layout(title_x=0.5, title_xanchor='center')
            st.plotly_chart(fig_kv, width="stretch")
            
        with st.expander("Statistical Details (Kruskal-Wallis H-Test)", expanded=False):
            st.markdown("""
            **Methodology Description**
            The Kruskal-Wallis H-test is a non-parametric method for testing whether multiple independent samples originate from the same distribution. It is used here because vehicle prices are highly skewed and do not follow a normal distribution.
            
            **Test Parameters:**
            - **Null Hypothesis ($H_0$):** The median price is identical across all top 10 brands.
            - **Alternative Hypothesis ($H_1$):** At least one brand has a different price distribution.
            - **Significance Threshold (α):** 0.05
            """)

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
            
            col_mw1, col_mw2 = st.columns([2, 3])
            with col_mw1:
                if p_u < 0.05:
                    finding_mw = f"There is a statistically significant price difference between {brand_a} and {brand_b}."
                    strength_mw = "Significant Difference"
                else:
                    finding_mw = f"There is no statistically significant price difference between {brand_a} and {brand_b}."
                    strength_mw = "Negligible Difference"
                    
                direction_mw = "Positive Shift" if diff > 0 else "Negative Shift" if diff < 0 else None
                metrics_mw = {
                    'U-Statistic': f"{u_stat:,.1f}",
                    'p-value': f"{p_u:.4e}" if p_u < 0.0001 else f"{p_u:.4f}",
                    'Median Difference': f"{diff:+.2f} Lakhs",
                    'Status': 'Significant' if p_u < 0.05 else 'Not Significant'
                }
                bullets_mw = [
                    f"**{brand_a}** median price: **{group_a.median():.1f} Lakhs LKR** ({len(group_a)} listings).",
                    f"**{brand_b}** median price: **{group_b.median():.1f} Lakhs LKR** ({len(group_b)} listings).",
                    f"Pricing spread: {brand_a} is median-priced **{abs(diff):.1f} Lakhs LKR** {'higher' if diff > 0 else 'lower'} than {brand_b}."
                ]
                render_insight_panel(finding_mw, strength_mw, direction_mw, metrics_mw, bullets_mw)
                
            with col_mw2:
                fig_comp = px.histogram(df[df['brand'].isin([brand_a, brand_b])], x='price', color='brand', barmode='overlay',
                                       marginal='box', title=f"{brand_a} vs {brand_b}")
                fig_comp.update_layout(title_x=0.5, title_xanchor='center')
                st.plotly_chart(fig_comp, width="stretch")
                
            with st.expander("Statistical Details (Mann-Whitney U Test)", expanded=False):
                st.markdown(f"""
                **Methodology Description**
                The Mann-Whitney U test is a non-parametric statistical test used to compare whether two independent groups differ significantly on a continuous variable (price). It is the non-parametric equivalent of the independent samples t-test.
                
                **Test Parameters:**
                - **Groups Compared:** {brand_a} vs. {brand_b}
                - **Null Hypothesis ($H_0$):** The distribution of prices is the same for both brands.
                - **Alternative Hypothesis ($H_1$):** The distributions of prices differ systematically.
                - **Significance Threshold (α):** 0.05
                """)

        st.divider()

        # --- 3. Age & performance correlation ---
        st.subheader("3. True Impact of Age & Performance")
        st.write("Spearman Correlation ($ρ$) measures how strictly two variables move together. Filter by brand to see specific depreciation trends.")
        
        selected_brands = st.multiselect("Filter Brands for Analysis", sorted(df['brand'].unique()), default=['TOYOTA', 'SUZUKI', 'NISSAN'])
        
        stat_df = df[df['brand'].isin(selected_brands)] if selected_brands else df
        
        if not stat_df.empty:
            if stat_df['age'].nunique() > 1 and stat_df['price'].nunique() > 1:
                corr_age, p_age = stats.spearmanr(stat_df['age'], stat_df['price'])
                
                col_age1, col_age2 = st.columns([2, 3])
                with col_age1:
                    abs_corr = abs(corr_age)
                    if abs_corr > 0.7:
                        strength_age = "Strong"
                    elif abs_corr > 0.4:
                        strength_age = "Moderate"
                    elif abs_corr > 0.1:
                        strength_age = "Weak"
                    else:
                        strength_age = "Negligible"
                        
                    direction_age = "Negative Correlation" if corr_age < 0 else "Positive Correlation" if corr_age > 0 else "None"
                    
                    if corr_age < 0:
                        finding_age = f"Vehicle age shows a {strength_age.lower()} negative relationship with price, indicating that older vehicles tend to have lower prices."
                    elif corr_age > 0:
                        finding_age = f"Vehicle age shows a {strength_age.lower()} positive relationship with price, indicating that older vehicles tend to have higher prices."
                    else:
                        finding_age = "There is no clear relationship between vehicle age and asking price."
                        
                    metrics_age = {
                        'Spearman ρ': f"{corr_age:.3f}",
                        'p-value': f"{p_age:.4e}" if p_age < 0.0001 else f"{p_age:.4f}",
                        'Sample Size': f"{len(stat_df):,}",
                        'Status': 'Significant' if p_age < 0.05 else 'Not Significant'
                    }
                    age_trend = "lower" if corr_age < 0 else "higher"
                    bullets_age = [
                        f"Older vehicles tend to have **{age_trend}** prices in the selected brands subset.",
                        f"The relationship is statistically classified as a **{strength_age}** correlation.",
                        f"Calculated dynamically based on a sample of **{len(stat_df):,}** listings."
                    ]
                    render_insight_panel(finding_age, strength_age, direction_age, metrics_age, bullets_age)
                    
                with col_age2:
                    fig_age_scatter = px.scatter(stat_df, x='age', y='price', color='brand',
                                               trendline="ols", title="Price vs. Age (Depreciation Curve)",
                                               labels={'age': 'Vehicle Age (Years)', 'price': 'Price (Lakhs LKR)'})
                    fig_age_scatter.update_layout(title_x=0.5, title_xanchor='center')
                    st.plotly_chart(fig_age_scatter, width="stretch")
                    
                with st.expander("Statistical Details (Spearman Rank Correlation)", expanded=False):
                    st.markdown(f"""
                    **Methodology Description**
                    Spearman's rank correlation coefficient (ρ) measures the strength and direction of the monotonic relationship between rank-ordered variables. Unlike Pearson correlation, it does not assume linearity or normal distributions.
                    
                    **Test Parameters:**
                    - **Variables:** Price vs. Age
                    - **Null Hypothesis ($H_0$):** There is no correlation between vehicle age and price.
                    - **Alternative Hypothesis ($H_1$):** There is a statistically significant correlation.
                    - **Significance Threshold (α):** 0.05
                    """)
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
            
            # Sort for ranked comparison chart
            sc_df = sc_df.sort_values(by='Correlation', ascending=False)
            
            # Calculate overall correlation
            if len(df) > 1 and df['engine_cc'].nunique() > 1 and df['price'].nunique() > 1:
                overall_cc, _ = stats.spearmanr(df['engine_cc'], df['price'])
            else:
                overall_cc = np.nan
            
            # Identify strongest and weakest segment
            sc_df['abs_corr'] = sc_df['Correlation'].abs()
            sc_df_sorted = sc_df.sort_values(by='abs_corr', ascending=False)
            strongest_seg = sc_df_sorted.iloc[0]['Segment']
            strongest_val = sc_df_sorted.iloc[0]['Correlation']
            weakest_seg = sc_df_sorted.iloc[-1]['Segment']
            weakest_val = sc_df_sorted.iloc[-1]['Correlation']
            
            col_eng1, col_eng2 = st.columns([2, 3])
            with col_eng1:
                if not pd.isna(overall_cc):
                    abs_overall = abs(overall_cc)
                    overall_strength = "strong" if abs_overall > 0.6 else "moderate" if abs_overall > 0.3 else "weak"
                    overall_dir = "positive" if overall_cc > 0 else "negative"
                    finding_eng = f"The relationship between engine capacity and price varies across segments. Overall, engine size shows a {overall_strength} {overall_dir} relationship with price across the entire market (ρ = {overall_cc:.2f})."
                else:
                    finding_eng = "The relationship between engine capacity and vehicle price varies across vehicle segments."
                    
                strength_eng = "Varying Segment Correlation"
                direction_eng = "Generally Positive" if overall_cc > 0 else "Generally Negative" if overall_cc < 0 else None
                
                metrics_eng = {
                    'Strongest Segment': f"{strongest_seg} (ρ = {strongest_val:.2f})",
                    'Weakest Segment': f"{weakest_seg} (ρ = {weakest_val:.2f})",
                    'Overall Spearman ρ': f"{overall_cc:.2f}" if not pd.isna(overall_cc) else "N/A",
                    'Status': 'Evaluated'
                }
                
                if abs(weakest_val) < 0.1:
                    weakest_desc = f"show almost no relationship between engine size and price (ρ = {weakest_val:.2f})"
                else:
                    weakest_desc = f"show the weakest relationship between engine size and price (ρ = {weakest_val:.2f})"
                    
                bullets_eng = [
                    f"**{strongest_seg}** vehicles show the strongest association between engine capacity and price (ρ = **{strongest_val:.2f}**).",
                    f"**{weakest_seg}** vehicles {weakest_desc}.",
                    f"In segments with positive correlations, larger engines are associated with higher pricing."
                ]
                render_insight_panel(finding_eng, strength_eng, direction_eng, metrics_eng, bullets_eng)
                
            with col_eng2:
                fig_seg_corr = px.bar(sc_df, x='Segment', y='Correlation', color='Segment',
                                    title="Price Sensitivity (Spearman ρ) by Engine Segment",
                                    labels={'Correlation': 'Spearman Correlation (ρ)'})
                fig_seg_corr.update_layout(title_x=0.5, title_xanchor='center')
                st.plotly_chart(fig_seg_corr, width="stretch")
                
            with st.expander("Statistical Details (Spearman Rank Correlation by Segment)", expanded=False):
                st.markdown("""
                **Methodology Description**
                Spearman's rank correlation coefficient (ρ) measures the strength and direction of association between engine size (cc) and vehicle price. Segmenting this correlation shows which types of vehicles carry a premium for larger engine capacity.
                
                **Test Parameters:**
                - **Group By:** Engine Segment
                - **Method:** Spearman Rank Correlation (ρ)
                - **Significance Threshold (α):** 0.05
                """)
                sc_df_show = sc_df[['Segment', 'Correlation', 'p-value']].copy()
                sc_df_show['Correlation'] = sc_df_show['Correlation'].map(lambda x: f"{x:.4f}")
                sc_df_show['p-value'] = sc_df_show['p-value'].map(lambda x: f"{x:.4e}" if x < 0.0001 else f"{x:.4f}")
                st.dataframe(sc_df_show)
                
        st.divider()

        # --- 2. Price volatility by fuel type ---
        st.subheader("2. Price Volatility by Fuel Type")
        st.write("Testing the variance (spread) of prices across all fuel types using Levene's Test.")
        
        valid_fuel_groups = {name: group['price'] for name, group in df.groupby('fuel_type') if len(group) > 5}
        
        if len(valid_fuel_groups) >= 2:
            stat_l, p_l = stats.levene(*[v.values for v in valid_fuel_groups.values()])
            
            col_lev1, col_lev2 = st.columns([2, 3])
            with col_lev1:
                if p_l < 0.05:
                    finding_lev = "There is a statistically significant difference in price volatility (variance) between vehicle fuel types."
                    strength_lev = "Significant Volatility Difference"
                else:
                    finding_lev = "No statistically significant difference in price volatility (variance) is detected between vehicle fuel types."
                    strength_lev = "Uniform Volatility"
                    
                metrics_lev = {
                    'Levene W-Stat': f"{stat_l:.2f}",
                    'p-value': f"{p_l:.4e}" if p_l < 0.0001 else f"{p_l:.4f}",
                    'Fuel Groups': f"{len(valid_fuel_groups)}",
                    'Status': 'Significant' if p_l < 0.05 else 'Not Significant'
                }
                
                bullets_lev = []
                if p_l < 0.05:
                    variances = {name: vals.var() for name, vals in valid_fuel_groups.items()}
                    most_volatile = max(variances, key=variances.get)
                    least_volatile = min(variances, key=variances.get)
                    bullets_lev.append(f"**{most_volatile}** vehicles exhibit the highest price volatility (variance = **{variances[most_volatile]:,.0f}**).")
                    bullets_lev.append(f"**{least_volatile}** vehicles are the most price-stable (variance = **{variances[least_volatile]:,.0f}**).")
                else:
                    bullets_lev.append("Price spread is statistically uniform across the different fuel categories.")
                bullets_lev.append(f"Analyzed {len(valid_fuel_groups)} fuel types with at least 5 listings.")
                
                render_insight_panel(finding_lev, strength_lev, None, metrics_lev, bullets_lev)
                
            with col_lev2:
                valid_fuels = list(valid_fuel_groups.keys())
                fig_risk = px.box(df[df['fuel_type'].isin(valid_fuels)], x='fuel_type', y='price', 
                                    color='fuel_type', points="all",
                                    title="Breadth of Price Ranges")
                fig_risk.update_layout(title_x=0.5, title_xanchor='center')
                st.plotly_chart(fig_risk, width="stretch")
                
            with st.expander("Statistical Details (Levene's Test for Homogeneity of Variance)", expanded=False):
                st.markdown("""
                **Methodology Description**
                Levene's test is used to assess if multiple groups have equal variances (homogeneity of variance). Equal variance across groups is an assumption of many parametric tests; here, we use it to identify price risk and market volatility across different fuel types.
                
                **Test Parameters:**
                - **Null Hypothesis ($H_0$):** Price variance is equal across all fuel types.
                - **Alternative Hypothesis ($H_1$):** Price variance differs significantly between fuel types.
                - **Significance Threshold (α):** 0.05
                """)

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
            feature_stats.append({'Feature': feat.replace('_', ' ').title(), 'p-value': p, 'Value Lift': lift, 'U-Statistic': u})
            
        feature_stats_df = pd.DataFrame(feature_stats).sort_values(by='Value Lift', ascending=False)
        sig_feats = [f['Feature'] for f in feature_stats if f['p-value'] < 0.05]
        
        col_feat1, col_feat2 = st.columns([2, 3])
        with col_feat1:
            if sig_feats:
                finding_feat = f"Features such as {', '.join(sig_feats)} show a statistically significant positive price lift on vehicle listings."
                strength_feat = "Significant Feature Premiums"
            else:
                finding_feat = "No features show a statistically significant positive price lift on vehicle listings."
                strength_feat = "No Significant Premium"
                
            metrics_feat = {f['Feature']: f"+{f['Value Lift']:.1f} L" for f in feature_stats}
            metrics_feat['Status'] = 'Significant Premiums' if sig_feats else 'Not Significant'
            
            highest_feat = feature_stats_df.iloc[0]['Feature']
            highest_lift = feature_stats_df.iloc[0]['Value Lift']
            lowest_feat = feature_stats_df.iloc[-1]['Feature']
            lowest_lift = feature_stats_df.iloc[-1]['Value Lift']
            
            bullets_feat = [
                f"**{highest_feat}** adds the highest premium with a median lift of **+{highest_lift:.1f} Lakhs LKR**.",
                f"**{lowest_feat}** adds the lowest premium with a median lift of **+{lowest_lift:.1f} Lakhs LKR**.",
                f"A total of **{len(sig_feats)}** comfort features are statistically significant value multipliers."
            ]
            render_insight_panel(finding_feat, strength_feat, "Positive Lift", metrics_feat, bullets_feat)
            
        with col_feat2:
            fig_lift = px.bar(feature_stats_df, x='Feature', y='Value Lift', color='Feature',
                              title="Median Price Lift by Comfort Feature",
                              labels={'Value Lift': 'Median Price Lift (Lakhs LKR)'})
            fig_lift.update_layout(title_x=0.5, title_xanchor='center')
            st.plotly_chart(fig_lift, width="stretch")
            
        with st.expander("Statistical Details (Mann-Whitney U One-Tailed Test)", expanded=False):
            st.markdown("""
            **Methodology Description**
            The one-tailed Mann-Whitney U test is used to check if the presence of a binary feature (e.g. Air Conditioning) is associated with systematically higher listing prices compared to listings without the feature.
            
            **Test Parameters:**
            - **Groups:** Listings 'With' vs. 'Without' feature
            - **Null Hypothesis ($H_0$):** Median price with feature is less than or equal to median price without feature.
            - **Alternative Hypothesis ($H_1$):** Median price with feature is greater.
            - **Significance Threshold (α):** 0.05
            """)
            feature_stats_df_show = feature_stats_df[['Feature', 'Value Lift', 'p-value', 'U-Statistic']].copy()
            feature_stats_df_show['Value Lift'] = feature_stats_df_show['Value Lift'].map(lambda x: f"{x:.2f}")
            feature_stats_df_show['p-value'] = feature_stats_df_show['p-value'].map(lambda x: f"{x:.4e}" if x < 0.0001 else f"{x:.4f}")
            feature_stats_df_show['U-Statistic'] = feature_stats_df_show['U-Statistic'].map(lambda x: f"{x:,.1f}" if isinstance(x, float) else f"{x:,}")
            st.dataframe(feature_stats_df_show)
            
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
        
        # Calculate highest and lowest association and classify strength
        def classify_cramers_v(val):
            if val > 0.5:
                return "strong"
            elif val >= 0.3:
                return "moderate"
            elif val >= 0.1:
                return "weak"
            else:
                return "negligible"

        associations = []
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                f1 = features[i]
                f2 = features[j]
                val = v_matrix.loc[f1, f2]
                associations.append(((f1, f2), val))
                
        strongest_pair, max_assoc = max(associations, key=lambda x: x[1])
        weakest_pair, min_assoc = min(associations, key=lambda x: x[1])
        
        f1_str = strongest_pair[0].replace('_', ' ').title()
        f2_str = strongest_pair[1].replace('_', ' ').title()
        w1_str = weakest_pair[0].replace('_', ' ').title()
        w2_str = weakest_pair[1].replace('_', ' ').title()
        
        strongest_strength = classify_cramers_v(max_assoc)
        weakest_strength = classify_cramers_v(min_assoc)
        
        col_cram1, col_cram2 = st.columns([2, 3])
        with col_cram1:
            finding_cram = "Comfort features exhibit varying degrees of association, suggesting standard bundles or packages in listing configurations."
            strength_cram = f"{strongest_strength.title()} Association"
            
            metrics_cram = {
                f"Strongest ({f1_str} & {f2_str})": f"{max_assoc:.2f}",
                f"Weakest ({w1_str} & {w2_str})": f"{min_assoc:.2f}",
                'Status': 'Evaluated'
            }
            
            bullets_cram = [
                f"The strongest bundling is seen between **{f1_str}** and **{f2_str}**, showing a **{strongest_strength}** association (Cramér's V = **{max_assoc:.2f}**).",
                f"The weakest bundling is between **{w1_str}** and **{w2_str}**, showing a **{weakest_strength}** association (Cramér's V = **{min_assoc:.2f}**).",
                f"Features with **strong** or **moderate** associations are frequently bundled together in listings."
            ]
            render_insight_panel(finding_cram, strength_cram, "Positive Association", metrics_cram, bullets_cram)
            
        with col_cram2:
            fig_heart = px.imshow(v_matrix, text_auto=".2f", color_continuous_scale="RdPu",
                                 title="Cramér's V: Feature Association Heatmap",
                                 labels=dict(x="Feature A", y="Feature B", color="Association"))
            fig_heart.update_layout(title_x=0.5, title_xanchor='center')
            st.plotly_chart(fig_heart, width="stretch")
            
        with st.expander("Statistical Details (Cramér's V Coefficient)", expanded=False):
            st.markdown("""
            **Methodology Description**
            Cramér's V is a measure of association between two nominal variables, giving a value between 0 (no association) and 1 (perfect association). It is based on Pearson's chi-squared statistic.
            
            **Thresholds of Association:**
            - **> 0.5:** Strong Association
            - **0.3 to 0.5:** Moderate Association
            - **0.1 to 0.3:** Weak Association
            - **< 0.1:** Negligible Association
            """)
