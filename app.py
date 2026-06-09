import streamlit as st
import base64
import os
from utils.data_loader import load_data
from utils.style_loader import load_css

# ==============================================================================
# 1. GLOBAL PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="AutoPulse.AI", 
    page_icon="images/logo.png", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# 2. CUSTOM CSS & BRANDING
# ==============================================================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet">
""", unsafe_allow_html=True)
st.markdown(load_css("header.css"), unsafe_allow_html=True)
# ==============================================================================
# 3. DATA LOADING & INITIALIZATION
# ==============================================================================
df = load_data()

# ==============================================================================
# 4. PAGE VIEW FUNCTIONS
# ==============================================================================
def show_home():
    """Renders the Home page view of the application."""
    import modules.home as home
    home.app(df)

def show_dashboard():
    """Renders the Market Insights Dashboard view of the application."""
    import modules.dashboard as dashboard
    dashboard.app(df)

def show_statistics():
    """Renders the Statistical Validation view of the application."""
    import modules.statistics as statistics
    statistics.app(df)

def show_prediction():
    """Renders the Price Predictor view of the application."""
    import modules.prediction as prediction
    prediction.app(df)

def show_help():
    """Renders the Help Center and Platform Guide view of the application."""
    import modules.help as help_module
    help_module.app(df)

# ==============================================================================
# 5. NATIVE MODERN NAVIGATION
# ==============================================================================
pages = [
    st.Page(show_home, title="Home", icon=":material/home:", default=True, url_path=""),
    st.Page(show_dashboard, title="Insights Dashboard", icon=":material/analytics:", url_path="dashboard"),
    st.Page(show_statistics, title="Statistical Tests", icon=":material/query_stats:", url_path="statistics"),
    st.Page(show_prediction, title="Price Predictor", icon=":material/auto_awesome:", url_path="prediction"),
    st.Page(show_help, title="Help Guide", icon=":material/help:", url_path="help")
]

pg = st.navigation(pages, position="top")
st.logo("images/logo.png", size="large")

# Make navigation logo clickable to navigate to the home page.
st.markdown("""
<script>
(function() {
    function addLogoClick() {
        const logo = document.querySelector('img[data-testid="stLogo"]');
        if (logo && !logo.dataset.clickBound) {
            logo.dataset.clickBound = 'true';
            logo.style.cursor = 'pointer';
            logo.addEventListener('click', function() {
                window.location.href = '/';
            });
        }
    }
    // Run on load and observe for re-renders
    addLogoClick();
    const observer = new MutationObserver(addLogoClick);
    observer.observe(document.body, {childList: true, subtree: true});
})();
</script>
""", unsafe_allow_html=True)

# --- Render active page ---
pg.run()

# ==============================================================================
# 6. GLOBAL FOOTER
# ==============================================================================

logo_base64 = ""
logo_path = os.path.join(os.path.dirname(__file__), "images/logo.png")
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode("utf-8")

footer_html = load_css("footer.css") + f"""
<div class="custom-footer">
    <div class="footer-layout">
        <div class="footer-col">
            <a href="/" target="_self" class="footer-title-link">
                <div class="footer-title">
                    <img class="footer-logo-img" src="data:image/png;base64,##LOGO_BASE64##" alt="AutoPulse.AI Logo">
                    AutoPulse.AI
                </div>
            </a>
            <div class="footer-badge">AI POWERED &bull; DATA DRIVEN &bull; TRUSTED</div>
        </div>
        <div class="footer-col">
            <div class="footer-col-title">Navigation</div>
            <ul class="footer-links">
                <li><a href="/" target="_self"><span class="material-symbols-rounded">home</span>Home</a></li>
                <li><a href="/dashboard" target="_self"><span class="material-symbols-rounded">analytics</span>Insights Dashboard</a></li>
                <li><a href="/statistics" target="_self"><span class="material-symbols-rounded">query_stats</span>Statistical Tests</a></li>
                <li><a href="/prediction" target="_self"><span class="material-symbols-rounded">auto_awesome</span>Price Predictor</a></li>
                <li><a href="/help" target="_self"><span class="material-symbols-rounded">help</span>Help Guide</a></li>
            </ul>
        </div>
        <div class="footer-col">
            <div class="footer-col-title" style="opacity: 0; pointer-events: none; user-select: none;">Spacer</div>
            <p class="footer-text">
                Built with <b>Streamlit</b>. 
            </p>
            <div class="footer-credits">
                © 2026 <b>AutoPulse.AI</b>. Developed by <b>Ranindu Kariyapperuma</b>.
            </div>
        </div>
    </div>
</div>
"""
st.markdown(footer_html.replace("##LOGO_BASE64##", logo_base64), unsafe_allow_html=True)
