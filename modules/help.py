import streamlit as st
from utils.style_loader import load_css, clean_html

def app(df):
    """
    Renders the Help Guide & Documentation page of the AutoPulse.AI web application.
    Builds subtab guides for the Insights Dashboard, the statistical engines, the XGBoost 
    price prediction models, and the Kaggle dataset attribution.

    Parameters:
        df (pd.DataFrame): The preprocessed vehicle listings dataset.
    """
    # Custom CSS for styling the help cards, glassmorphic UI, hover effects, and typography
    st.markdown(load_css("help.css"), unsafe_allow_html=True)

    # --- Header section ---
    st.title("Help & Platform Guide")
    st.markdown("""
        Welcome to the **AutoPulse.AI Help Center**. This portal explains the core features, 
        statistical mechanics, and predictive models powering the platform. Use this guide to get the most 
        out of our market intelligence tools.
    """)
    
    st.divider()

    # --- Core modules grid ---
    st.subheader("Explore the Platform Features")

    st.markdown(clean_html(f"""
        <div class="help-grid">
            <div class="help-card">
                <div class="icon-badge blue">
                    <span class="material-symbols-rounded">home</span>
                </div>
                <h4 class="card-title">Home Page</h4>
                <p class="card-desc">
                    Get an introduction to the AutoPulse.AI platform, view raw data samples, and understand our data-driven methodology.
                </p>
            </div>
            <div class="help-card">
                <div class="icon-badge orange">
                    <span class="material-symbols-rounded">analytics</span>
                </div>
                <h4 class="card-title">Insights Dashboard</h4>
                <p class="card-desc">
                    Explore vehicle demographics, regional heatmaps, price retention, and feature premiums. Customize using global filters.
                </p>
            </div>
            <div class="help-card">
                <div class="icon-badge purple">
                    <span class="material-symbols-rounded">query_stats</span>
                </div>
                <h4 class="card-title">Statistical Tests</h4>
                <p class="card-desc">
                    Verify market assumptions using rigorous non-parametric tests, volatility checks, and feature bundling heatmaps.
                </p>
            </div>
            <div class="help-card">
                <div class="icon-badge green">
                    <span class="material-symbols-rounded">auto_awesome</span>
                </div>
                <h4 class="card-title">Price Predictor</h4>
                <p class="card-desc">
                    Get accurate vehicle valuations using our optimized XGBoost regressor, and inspect individual SHAP force plots.
                </p>
            </div>
            <div class="help-card">
                <div class="icon-badge red">
                    <span class="material-symbols-rounded">help</span>
                </div>
                <h4 class="card-title">Help Guide & FAQ</h4>
                <p class="card-desc">
                    Access detailed instructions, read statistical test documentation, explore dataset coverage, and review platform FAQs.
                </p>
            </div>
        </div>
    """), unsafe_allow_html=True)

    st.markdown("---")

    # --- Detailed documentation tabs ---
    help_tabs = st.tabs([
        ":material/analytics: Insights Dashboard Guide",
        ":material/query_stats: Statistical Tests",
        ":material/auto_awesome: Prediction & SHAP",
        ":material/info: Platform FAQ & Data"
    ])

    # --- 1. Insights Dashboard ---
    with help_tabs[0]:
        st.header("How to Use the Insights Dashboard")
        st.markdown("""
            The Insights Dashboard aggregates Sri Lankan secondary market data into **5 key analysis areas**. 
            This gives you a multi-dimensional look at how cars hold and lose value.
        """)
        
        col_d1, col_d2 = st.columns([1, 1])
        
        with col_d1:
            st.markdown("""
                ### :material/filter_alt: Interactive Global Filters
                At the top of the dashboard, you can narrow down the market analysis to match your interest:
                - **Price Range (Lakhs LKR)**: Focus on entry-level, mid-range, or luxury tiers.
                - **Year of Manufacture (YOM)**: Filter out older vehicles or look strictly at modern imports.
                - **Mileage Range (km)**: Benchmark low-mileage models against heavily driven ones.
                - **Remove Outliers Option**: Toggle this to exclude the top 5% highest-priced cars dynamically from the current dataset query.
                
                #### :material/filter_list: The Outlier Filter Mechanism
                The outlier filter uses the **95th percentile** threshold on vehicle prices. When enabled:
                - It removes the top 5% of highest-priced vehicle records (rare luxury brands, supercars, or listings with anomalous data).
                - This ensures that descriptive metrics (like median depreciation curves and average price tiers) accurately reflect the **mainstream used car market** in Sri Lanka and are not skewed by extreme values.
            """)
            
            st.markdown("""
                ### :material/explore: The 5 Key Analysis Areas
                1. **Budget & Availability**: Plots the volume of vehicles across price points (using a histogram and smoothed KDE curve) and identifies dominant brands in your selection.
                2. **Value Retention & Fuel Types**: Tracks how median price decays over a vehicle's age and compares pricing across fuel categories.
                3. **Transmission & Engine Specs**: Visualizes the price premium for Automatic vs Manual gearboxes, and shows how engine displacement (cc) correlates with value.
            """)
            
        with col_d2:
            st.markdown("""
                4. **Features & Comfort**: Analyzes the price difference (value lift) between cars with and without essential comfort options (AC, power windows, etc.). It also shows a standardization heatmap of how features became standard equipment over YOM.
                5. **Regional Price Distribution**: Renders a **Choropleth Map** of Sri Lanka. Hovering over a province reveals its median vehicle price and total listing counts.
            """)
            
            st.info("""
                **Quick Tip:** Plotly charts are fully interactive! You can zoom in, hover over data points for tooltips, and double-click the legend to isolate specific brands or segments.
            """)

    # --- 2. Statistical Tests ---
    with help_tabs[1]:
        st.header("Understanding the Statistical Engine")
        st.markdown("""
            AutoPulse.AI uses scientific statistical tests (from `scipy.stats`) to confirm whether the patterns we see 
            in the market are genuinely meaningful or just random sample noise.
        """)

        with st.expander(":material/filter_list: Outlier Mitigation & Statistical Integrity"):
            st.markdown("""
                - **What it does:** Allows removing listings in the top 5% price bracket before conducting statistical tests.
                - **Statistical Significance:** Extreme outliers have high leverage in parametric and rank-based tests by inflating variance. Removing price outliers stabilizes the variance ($σ^2$) and ensures correlation coefficients ($ρ$) and test statistics ($H, U$) accurately represent mainstream market trends.
                - **User Value:** Toggle the outlier option at the top of the statistics page to observe if a relationship holds for the general market or is dominated by luxury exceptions.
            """)

        # Expanders for individual statistical tests
        with st.expander(":material/bar_chart: Brand Variance: Kruskal-Wallis H-Test", expanded=True):
            st.markdown("""
                - **What it does:** Compares the price distributions of the top 10 brands.
                - **How to interpret:** A low p-value (< 0.05) proves that different brand names carry significantly different market values in Sri Lanka.
                - **Why not just look at averages?** Averages can be easily distorted by a few extremely expensive listings. The Kruskal-Wallis test looks at the overall distribution of ranks, providing a much more robust proof.
            """)

        with st.expander(":material/balance: Brand Benchmarking: Mann-Whitney U Test"):
            st.markdown("""
                - **What it does:** Allows you to select any two brands and checks if their price distributions are statistically different.
                - **How to interpret:** If the p-value is less than 0.05, the difference in their prices is statistically significant (meaning the market values them differently).
                - **User Value:** Useful for comparing competitor brands (e.g., Toyota vs. Suzuki) to see if one has a genuine price premium over the other.
            """)

        with st.expander(":material/trending_down: Value Decay: Spearman's Rank Correlation ($ρ$)"):
            st.markdown("""
                - **What it does:** Measures the strength and direction of the relationship between Vehicle Age and Price. You can filter by brand to observe brand-specific depreciation trends.
                - **How to interpret:** A value near `-1.0` indicates a perfect negative correlation (as the car gets older, the price strictly drops). A value near `0.0` means no relationship.
                - **User Value:** Tracks depreciation rate consistency. If you filter for a specific brand, a strong negative Spearman value shows the brand follows a highly predictable depreciation curve.
            """)

        with st.expander(":material/speed: Engine Size vs Price: Segmented Spearman Correlation"):
            st.markdown("""
                - **What it does:** Computes the Spearman Correlation between Engine Capacity (cc) and Price, grouped by vehicle engine segment (Micro, Compact, Mid-Range, Large).
                - **How to interpret:** A high positive correlation in a segment means larger engines command higher prices within that category. A low or near-zero value means engine size has less influence.
                - **User Value:** Reveals whether upgrading engine size yields a proportional price gain within a specific vehicle class, or if there is a ceiling effect.
            """)

        with st.expander(":material/warning: Volatility & Risk: Levene's Test"):
            st.markdown("""
                - **What it does:** Tests if different fuel type groups have significantly different spreads (variances) in their prices.
                - **How to interpret:** If significant (p-value < 0.05), it tells you that one fuel category has much higher price volatility than the others.
                - **User Value:** Higher volatility means buying/selling is riskier, as prices are highly spread out, while low volatility suggests stable, predictable pricing.
            """)

        with st.expander(":material/verified: Feature Price Multipliers: Mann-Whitney U (One-Tailed)"):
            st.markdown("""
                - **What it does:** Tests whether the presence of a comfort feature (AC, Power Steering, Power Mirror, Power Window) significantly raises the median vehicle price, using a one-tailed Mann-Whitney U test.
                - **How to interpret:** A significant result (p-value < 0.05) confirms the feature adds genuine value. The "Value Lift" metric shows the median price difference in Lakhs.
                - **User Value:** Quantifies exactly how much each comfort extra is worth in real market terms, helping you decide which features matter most when buying or selling.
            """)

        with st.expander(":material/link: Feature Bundling: Cramér's V"):
            st.markdown("""
                - **What it does:** Computes the strength of association between binary features (like Power Windows, Power Mirrors, Power Steering, AC).
                - **How to interpret:** `0.0` means features are independent; `1.0` means they are perfectly bundled (always appearing together).
                - **User Value:** Shows packages of features that sellers group together. Highly correlated features are usually standard on premium trims.
            """)

    # --- 3. Prediction & SHAP ---
    with help_tabs[2]:
        st.header("The AI Predictive & Explainability Models")
        st.markdown("""
            AutoPulse.AI uses an advanced machine learning pipeline to estimate vehicle values. 
            However, we believe models should be a **Glass Box**, not a Black Box. Here is how it works:
        """)
        
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.markdown("""
                ### :material/psychology: The Machine Learning Model (XGBoost)
                Our predictions are generated by an **XGBoost Regressor** model trained on historical Sri Lankan car listings.
                - **Hyperparameter Optimization**: The model was tuned using `RandomizedSearchCV` across 300 parameter combinations.
                - **Evaluation Metric**: Achieved an **R-squared of 0.89**, which means the model explains 89% of the price variance.
                - **Outlier Handling & Log Transformation**: Price outliers in the target variable were **not removed** during training because high-end luxury vehicle data is important for full market coverage. Instead, a natural log transformation was applied to the target variable (`price`) to compress the scale, handle skewness, and stabilize variance during XGBoost learning.
                - **Feature Engineering**: Numerical features like engine size are log-transformed to stabilize variance, and categorical features are one-hot encoded.
                - **Electric Vehicle Handling**: Selecting `Electric` fuel type disables `Engine Capacity (cc)` and `Transmission` inputs, while the backend overrides them to `1500cc` and `Automatic` respectively, conforming to the model's structural schema.
            """)
            
            st.markdown("""
                ### :material/compare: SHAP Force Plots (Attribution)
                After you run a prediction, we render a **SHAP Force Plot**:
                - **Base Value**: Represents the average price (in log scale) predicted across all cars in the database.
                - **Positive Impact (Red Bars, Left-to-Right)**: Features that **pushed the price higher** for this specific vehicle.
                - **Negative Impact (Blue Bars, Right-to-Left)**: Features that **pulled the price lower** for this specific vehicle.
            """)
            
        with col_p2:
            st.markdown("""
                ### :material/table_chart: Feature Impact Estimates Table
                We translate complex SHAP log-impacts into readable monetary values (Lakhs LKR) and percentage shifts:
                - **Log Impact**: The mathematical value assigned to the feature by the model.
                - **Est. Price Change (%)**: The relative multiplier effect of the feature.
                - **Est. Impact (Lakhs)**: The estimated monetary addition (e.g., `+4.50L` for Toyota brand) or subtraction (e.g., `-3.20L` for high age) from the final price.
            """)
            
            st.markdown("""
                ### :material/insights: Model Insights Panel
                Toggle the **"Show Model Insights"** checkbox at the bottom of the Predictor page to access two additional diagnostic views:
                - **Aggregated Feature Importance**: A horizontal bar chart ranking each feature group (Age, Brand, Fuel Type, etc.) by its overall contribution to the XGBoost model during training.
                - **Aggregated SHAP Analysis**: Computes mean absolute SHAP values across an optimized, downsampled background subset of 150 records to ensure quick load times and avoid memory exhaustion (OOM) on deployment, showing how much each feature group typically shifts predictions across all vehicles — not just a single estimate.
            """)
            
            st.success("""
                **Transparency standard:** Using SHAP ensures that you can see exactly why the AI appraised your car at a specific value, making the valuation fully auditable.
            """)

    # --- 4. Platform FAQ & Data ---
    with help_tabs[3]:
        st.header("Frequently Asked Questions & Data Overview")
        
        faq1, faq2 = st.columns(2)
        
        with faq1:
            with st.container(border=True):
                st.markdown("""
                    **Q: What currency is used on the platform?**
                    **A:** All prices are shown in **Lakhs (LKR)**. One Lakh equals 100,000 Sri Lankan Rupees.
                """)
            
            with st.container(border=True):
                st.markdown("""
                    **Q: How is age calculated?**
                    **A:** Age is computed dynamically using the current calendar year as the baseline (e.g., if the current year is 2026, a car manufactured in 2020 is treated as 6 years old).
                """)
                
            with st.container(border=True):
                st.markdown("""
                    **Q: How are Electric Vehicles handled in the Price Predictor?**
                    **A:** Selecting `Electric` fuel type dynamically disables the `Engine Capacity (cc)` and `Transmission` input fields. Before calling the machine learning model, the backend overrides these values to `1500cc` and `Automatic` transmission to match the structure of the training dataset, ensuring stable predictions.
                """)
                
            with st.container(border=True):
                st.markdown("""
                    **Q: What does 'RARE' or 'OTHER' brand mean in the Predictor?**
                    **A:** To ensure model stability, brands with fewer than 10 listings are grouped as `RARE` and brands with 10-100 listings are grouped as `OTHER`.
                """)
                
        with faq2:
            with st.container(border=True):
                st.markdown("""
                    **Q: Can I use this for unregistered or brand-new cars?**
                    **A:** The dataset contains records of **USED** cars only, aligning with secondary market dynamics.
                """)
                
            with st.container(border=True):
                st.markdown("""
                    **Q: How do the town-to-province mappings work?**
                    **A:** Towns are mapped to Sri Lanka's 9 provinces (Western, Central, Southern, Northern, Eastern, North Western, North Central, Uva, Sabaragamuwa) based on standard administrative boundaries.
                """)

            with st.container(border=True):
                st.write(f"**Current Dataset Size:** {len(df):,} listings")
                st.write(f"**Coverage:** {len(df['brand'].unique())} Brands, across all 9 Provinces.")

        st.divider()
        st.markdown("### :material/database: Data Source & Attribution")
        st.markdown("""
            AutoPulse.AI is powered by the public **Sri Lankan Second Vehicle/Car Price Dataset** hosted on Kaggle:
            
            - **Dataset URL**: [Kaggle - Sri Lankan Second Vehicle/Car Price Dataset](https://www.kaggle.com/datasets/prasadnirmal/srilankan-second-vehiclecar-price-dataset)
            - **Author**: Prasad Nirmal
            - **Scope**: Contains listing prices, model details, mileage, engine capacity, gear transmissions, fuel types, and locations for secondary vehicles in Sri Lanka.
            - **Integrity**: Standardized, preprocessed, and filtered to remove null variables, providing a clean foundation for analysis and machine learning predictions.
        """)

    # --- Bottom Callout ---
    st.divider()
    st.info(":material/info: Need to return to data exploration? Click **Home**, **Insights Dashboard**, **Statistical Tests**, or **Price Predictor** in the navigation bar.")
