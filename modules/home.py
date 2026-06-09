import streamlit as st
from utils.style_loader import load_css, clean_html

def app(df):
    """
    Renders the Home page of the AutoPulse.AI web application.
    Displays the hero banner, value proposition feature cards, platform statistics bar,
    project capability overview, and methodology details.

    Parameters:
        df (pd.DataFrame): The preprocessed vehicle listings dataset.
    """
    # --- Load custom styles ---
    st.markdown(load_css("home.css"), unsafe_allow_html=True)



    # --- 1. Hero Section (using Streamlit native columns) ---
    col_hero_left, col_hero_right = st.columns([1, 1], gap="large")

    with col_hero_left:
        st.markdown("""
            <h1 class="hero-heading">
                AI-Powered Vehicle<br>
                <span class="highlight">Valuation &amp; Market<br>Intelligence</span>
            </h1>
            <div class="hero-line"></div>
            <p class="hero-desc">
                AutoPulse.AI helps you make informed decisions in the Sri Lankan used
                car market through data driven insights and intelligent valuation tools.
            </p>
            <div class="hero-buttons">
                <a href="/dashboard" target="_self" class="hero-btn hero-btn-primary">
                    Explore Dashboard
                    <span class="material-symbols-rounded">analytics</span>
                </a>
                <a href="/prediction" target="_self" class="hero-btn hero-btn-secondary">
                    Try Price Predictor
                    <span class="material-symbols-rounded">auto_awesome</span>
                </a>
            </div>
        """, unsafe_allow_html=True)

    with col_hero_right:
        st.markdown('<div style="margin-top: 30px;">', unsafe_allow_html=True)
        st.image("images/image.png", width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- 2. Why Choose AutoPulse.AI Feature Cards ---
    st.markdown("""
        <div class="section-heading">Why Choose AutoPulse.AI?</div>
        <div class="section-underline"></div>

        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-icon-circle blue">
                    <span class="material-symbols-rounded">monitoring</span>
                </div>
                <div class="feature-card-title">Accurate Valuations</div>
                <p class="feature-card-desc">
                    Understand how features actually shift car value with precision.
                </p>
            </div>
            <div class="feature-card">
                <div class="feature-icon-circle violet">
                    <span class="material-symbols-rounded">show_chart</span>
                </div>
                <div class="feature-card-title">Interactive Charts</div>
                <p class="feature-card-desc">
                    Explore price movements and market trends through beautiful visualizations.
                </p>
            </div>
            <div class="feature-card">
                <div class="feature-icon-circle rose">
                    <span class="material-symbols-rounded">verified</span>
                </div>
                <div class="feature-card-title">Trusted Data</div>
                <p class="feature-card-desc">
                    Make confident decisions backed by reliable data, not guesswork.
                </p>
            </div>
            <div class="feature-card">
                <div class="feature-icon-circle amber">
                    <span class="material-symbols-rounded">bolt</span>
                </div>
                <div class="feature-card-title">Smart & Fast</div>
                <p class="feature-card-desc">
                    AI-driven insights in seconds, saving you time and effort.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- 3. Platform Statistics Bar ---
    total_records = f"{len(df):,}"
    total_brands = f"{df['brand'].nunique()}" if 'brand' in df.columns else "10+"
    total_towns = f"{df['town'].nunique()}" if 'town' in df.columns else "50+"

    st.markdown(clean_html(f"""
        <div class="stats-bar">
            <div class="stat-item">
                <div class="stat-icon">
                    <span class="material-symbols-rounded">directions_car</span>
                </div>
                <div>
                    <div class="stat-value">{total_records}+</div>
                    <div class="stat-label">Vehicles Analyzed</div>
                </div>
            </div>
            <div class="stat-item">
                <div class="stat-icon">
                    <span class="material-symbols-rounded">sell</span>
                </div>
                <div>
                    <div class="stat-value">{total_brands}+</div>
                    <div class="stat-label">Brands Represented</div>
                </div>
            </div>
            <div class="stat-item">
                <div class="stat-icon">
                    <span class="material-symbols-rounded">trending_up</span>
                </div>
                <div>
                    <div class="stat-value">89%</div>
                    <div class="stat-label">Prediction R² Score</div>
                </div>
            </div>
            <div class="stat-item">
                <div class="stat-icon">
                    <span class="material-symbols-rounded">location_on</span>
                </div>
                <div>
                    <div class="stat-value">{total_towns}+</div>
                    <div class="stat-label">Cities & Towns</div>
                </div>
            </div>
        </div>
    """), unsafe_allow_html=True)

    st.divider()

    # --- 4. Core Capabilities ---
    st.header(":material/rocket_launch: Core Capabilities")

    st.markdown("""
        <div class="cap-grid">
            <div class="cap-card">
                <div class="cap-icon-badge blue">
                    <span class="material-symbols-rounded">psychology</span>
                </div>
                <h4 class="cap-card-title">AI Price Prediction</h4>
                <p class="cap-card-sub">XGBoost Regressor Model</p>
                <ul class="cap-bullet-list">
                    <li class="cap-bullet-item">
                        <span class="material-symbols-rounded">check_circle</span>
                        Predicts prices with high accuracy (R² = 0.89)
                    </li>
                    <li class="cap-bullet-item">
                        <span class="material-symbols-rounded">check_circle</span>
                        Interpretable SHAP transparency
                    </li>
                    <li class="cap-bullet-item">
                        <span class="material-symbols-rounded">check_circle</span>
                        Handles log-transformed distributions
                    </li>
                </ul>
            </div>
            <div class="cap-card">
                <div class="cap-icon-badge orange">
                    <span class="material-symbols-rounded">analytics</span>
                </div>
                <h4 class="cap-card-title">Insights Dashboard</h4>
                <p class="cap-card-sub">Dynamic Market Analysis</p>
                <ul class="cap-bullet-list">
                    <li class="cap-bullet-item">
                        <span class="material-symbols-rounded">check_circle</span>
                        Budget & revenue visibility
                    </li>
                    <li class="cap-bullet-item">
                        <span class="material-symbols-rounded">check_circle</span>
                        Reliability & depreciation curves
                    </li>
                    <li class="cap-bullet-item">
                        <span class="material-symbols-rounded">check_circle</span>
                        Regional price choropleth heatmaps
                    </li>
                </ul>
            </div>
            <div class="cap-card">
                <div class="cap-icon-badge green">
                    <span class="material-symbols-rounded">query_stats</span>
                </div>
                <h4 class="cap-card-title">Advanced Statistics</h4>
                <p class="cap-card-sub">Statistical Tests & Validation</p>
                <ul class="cap-bullet-list">
                    <li class="cap-bullet-item">
                        <span class="material-symbols-rounded">check_circle</span>
                        Hypothesis validation (Kruskal-Wallis)
                    </li>
                    <li class="cap-bullet-item">
                        <span class="material-symbols-rounded">check_circle</span>
                        Risk assessment & volatility (Levene's)
                    </li>
                    <li class="cap-bullet-item">
                        <span class="material-symbols-rounded">check_circle</span>
                        Feature bundling (Cramér's V)
                    </li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    # --- 5. Methodology & Statistical Robustness ---
    st.header(":material/analytics: Our Methodology")
    col_m1, col_m2 = st.columns([2, 1])
    
    with col_m1:
        st.markdown("""
            ### A Data-Driven Approach
            This application goes beyond basic averages. We analyze thousands of real-world listings from the Sri Lankan 
            secondary market, focusing on the **2025 import liberalization phase**.
            
            Our platform uses **Non-Parametric Testing** (Kruskal-Wallis, Mann-Whitney U) to ensure that the "Brand 
            Premiums" and "Feature Multipliers" you see are statistically significant and not just noise.

            ### Predictive Modeling
            To deliver precise valuations, we employ an **XGBoost Regressor** trained on historical market data. 
            The model was optimized using **RandomizedSearchCV** (300 fits) to balance depth and learning rate, achieving 
            an **R-squared of 0.89**.
            
            For transparency, we use **SHAP (SHapley Additive exPlanations)**. This allows the portal to explain exactly 
            which features (e.g., brand, age, or fuel type) influenced a specific valuation, ensuring transparency 
            instead of a Black Box.
        """)
            
    with col_m2:
        st.success("""
            **Market Coverage:**
            - 10+ Major Brands
            - All 9 Provinces
            - Petrol, Diesel, Hybrid & EV
            - 1956 - 2024 Models
        """)

    # --- 6. Raw Data Preview ---
    st.divider()
    if st.checkbox("Show Raw Data Sample"):
        st.dataframe(df.head(10), width="stretch")
        st.write(f"**Total Records Analyzed:** {len(df):,}")

    st.info(":material/info: Use the navigation bar at the top to explore the Prediction Model, Market Dashboard, or Statistics module.")
