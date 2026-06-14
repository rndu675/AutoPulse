# <img width="80" height="45" alt="image" src="https://github.com/user-attachments/assets/a22bf0bc-d29d-4f8f-993c-d51a99631204" /> AutoPulse.AI

**AutoPulse.AI** is an automotive market insights and valuation platform built with `Streamlit`. The application combines interactive analytics, statistical validation techniques and machine learning to help users explore Sri Lankan used car market trends and estimate prices using historical data.

<p align="center">
 <img width="945" height="531.5" alt="image" src="https://github.com/user-attachments/assets/a04931dc-426e-4db7-80a6-3656278279d9" />
</p>

<p align="center">
  <a href="https://autopulse-ai.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-Streamlit-blue?style=for-the-badge&logo=streamlit" alt="Live Demo">
  </a>
</p>



## <img src="https://cdn.simpleicons.org/streamlit" width="18" /> Live Application

The application is deployed on **Streamlit Community Cloud** and can be accessed here: **https://autopulse-ai.streamlit.app**



## Features

* **Interactive Homepage** -
A clean landing page introducing the platform and its capabilities.

* **Insights Dashboard** -
Visual exploration of vehicle market trends, price distributions, brand comparisons and geographical distributions with dynamic interpretations, global filters and interactive charts powered by `Plotly` and `Folium`.

* **Statistical Tests** -
Hypothesis testing, correlation analysis and regression checks using `SciPy` and `Statsmodels` to statistically validate market assumptions.

* **Price Predictor** -
Vehicle price prediction powered by an `XGBoost` regressor (R² = 0.89), with integrated `SHAP` explanations and feature importance analysis.

* **Help Guide** -
Comprehensive documentation and interactive usage guides.



## Project Structure

```text
.
├── .gitignore                       # Git ignore file (excludes venv, python cache)
├── LICENSE                          # MIT License
├── README.md                        # Project documentation
├── app.py                           # Main Streamlit application entry point
├── requirements.txt                 # Python package dependencies
│
├── css/                             # Custom styling rules
│   ├── footer.css                     # Modern footer styling
│   ├── header.css                     # Page header layout styling
│   ├── help.css                       # Help page specific styling
│   └── home.css                       # Landing page styles
│
├── data/                            # Data resources
│   ├── car_price_dataset.csv          # Vehicle market dataset
│   └── provinces.geojson              # Geographical mapping boundary data
│
├── images/                          # Visual assets & brand elements
│   ├── image.png                      # Landing page image
│   ├── logo.png                       # Brand logo
│   └── icons/                         # SVG icons for dashboard (for faster loading)
│       ├── brand.svg
│       ├── calendar.svg
│       ├── car.svg
│       ├── fuel.svg
│       ├── mileage.svg
│       └── price.svg
│
├── models/                          # Trained ML models
│   └── xgboost_model_artifacts.joblib  # Trained XGBoost model
│
├── modules/                         # Application modules
│   ├── dashboard.py                   # Visual analysis dashboard module
│   ├── help.py                        # User support and platform guide module
│   ├── home.py                        # Landing page module
│   ├── prediction.py                  # Price estimator module
│   └── statistics.py                  # Statistical testing module
│
└── utils/                           # Helper functions
    ├── data_loader.py                 # Cache supported data load and cleaning
    └── style_loader.py                # Custom CSS injection
```



## Installation

**Prerequisites**

- Python `3.9` or higher
- `pip`

**1. Clone the Repository**

```bash
git clone https://github.com/rndu675/AutoPulse.git
cd AutoPulse
```

**2. Set Up a Virtual Environment (Recommended)**

This step is optional but recommended to keep dependencies isolated:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate (Linux/macOS):
source .venv/bin/activate

# Activate (Windows):
.venv\Scripts\activate
```

**3. Install Dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the Application**

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.



## Tech Stack

| Layer | Libraries |
|---|---|
| Web Framework | [`Streamlit`](https://streamlit.io/) |
| Data Processing | [`Pandas`](https://pandas.pydata.org/), [`NumPy`](https://numpy.org/) |
| Visualizations | [`Plotly`](https://plotly.com/), [`Matplotlib`](https://matplotlib.org/), [`Folium`](https://python-visualization.github.io/folium/) |
| Machine Learning | [`Scikit-Learn`](https://scikit-learn.org/), [`XGBoost`](https://xgboost.readthedocs.io/), [`SHAP`](https://shap.readthedocs.io/), [`Joblib`](https://joblib.readthedocs.io/) |
| Statistical Analysis | [`SciPy`](https://scipy.org/), [`Statsmodels`](https://www.statsmodels.org/) |



## Dataset

The vehicle market data is sourced from Kaggle:
[Sri Lankan Second-Hand Vehicle/Car Price Dataset](https://www.kaggle.com/datasets/prasadnirmal/srilankan-second-vehiclecar-price-dataset)



## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

Copyright &copy; 2026 Ranindu
