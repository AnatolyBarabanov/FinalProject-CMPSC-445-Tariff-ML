# FinalProject-CMPSC-445-Tariff-ML
Made by AnatolyBarabanov and Matthew Danese 
Professor: Janghoon Yang

## Description of the Project
This project develops a machine learning-based Tariff Impact Dashboard that enables users to:

- forecast commodity prices,

- cluster commodities based on their price behavior,

- classify commodity risk levels.

The system uses the World Bank's "CMO-Historical-Data-Monthly.xlsx" dataset and integrates three trained models (Regression, Clustering, Classification) into interactive Streamlit web application.

Live Website: [![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-green?logo=streamlit)](https://finalproject-cmpsc-445-tariff-ml-ka3ig4hwa93cmwpuusycyv.streamlit.app)

---
## Significance of the Project

Understanding commodity market behavior is crucial for traders, and businesses.
Our dashboard provides:

- Forecasting of future prices to support better economic decisions,

- Clustering of commodities for comparative analysis,

- Risk classification to highlight commodities with high volatility risks.

The project integrates multiple ML techniques into one web platform, offering real-time insights into complex tariff-related economic trends.

---
## Project Goals

- Predict future commodity prices.
- Group commodities with similar behavior.
- Classify commodities into risk categories.
- Provide visual insights via a web dashboard.

---

## Data Collection

- **Source:** [World Bank – Commodity Markets](https://www.worldbank.org/en/research/commodity-markets)
- **File:** `CMO-Historical-Data-Monthly.xlsx`
- Monthly commodity prices across agriculture, energy, and metals sectors.

Data Characteristics:

- Covers over 80 commodities.

- Time range: 1980s–2025s.

Data points: Monthly price observations.

Metadata Fields:

- Commodity name

- Units (e.g., USD/ton, USD/barrel)

- Monthly price values
---

## Project Structure

FinalProject-CMPSC-445-Tariff-ML/

├── models/                # Trained machine learning models (.pkl)

│    └──kmeans_model.pkl

│    └── price_forecast_model_commodities.pkl

│    └── kmeans_scaler.pkl

│    └── risk_classification_commodities.pkl

├── data/                  # Dataset (World Bank CMO data)

│    └── CMO-Historical-Data-Monthly.xlsx

├── scripts/

│    └── tariff_models.py

│    └── train_regression.py

│    └── train_kmeans.py

│    └── train_classifier.py

├── requirements.txt       # Python libraries to install

└── README.md              # Project description

└── tariff_dashboard1.py

Scripts:

 ### 1. `tariff_models.py` — First to Run: Shared Model Functions

- Contains **helper functions** for training and saving machine learning models.
- Functions are imported into the training scripts (`train_classifier.py`, `train_kmean.py`, `train_regression.py`).

 ### 2. `train_regression.py` — Price Forecasting Model Trainer

- Trains a **Linear Regression model** to **predict future commodity prices**.
- Loads historical data from **`CMO-Historical-Data-Monthly.xlsx`**.
- Fits the model using previous months' prices as input features.
- Saves the trained model to:
   ```bash
   models/price_forecast_model.pkl

> **Run this file after `tariff_models.py` to create the price forecasting model.**

 ### 3. `train_kmean.py` — Clustering Model Trainer

- Trains a **KMeans clustering model** to **group commodities** based on their price trends.
- Uses **Principal Component Analysis (PCA)** for cluster visualization.
- Automatically determines the **optimal number of clusters** using the **Silhouette Score**.
- Saves:
- Clustering model:
  ```
  models/clustering_model.pkl
  ```
- Cluster visualization plot:
  ```
  commodity_clusters.png
  ```
  
> **Run this file third to create the clustering model.**

 ### 4. `train_classifier.py` — Risk Classification Model Trainer

- Trains a **Decision Tree Classifier** to **predict commodity risk levels** (`High`, `Medium`, `Low`).
- Defines risk categories based on **recent price volatility**.
- Saves the trained classifier model to:
   ```bash
   models/risk_classification_model.pkl
> **Run this file fourth to create the risk classification model.**

### 5. `tariff_dashboard1.py` — Streamlit Web Application

- Combines all three trained models into an **interactive Streamlit dashboard**.
- Features:
- **Forecast future commodity prices**
- **View commodity clusters**
- **Predict commodity risk levels**
- Loads models from the `models/` folder.
- Displays **interactive charts**, **tables**, and **real-time predictions**.

> **After training all models, run this file to launch the web app:**
> ```bash
> streamlit run tariff_dashboard1.py
> ```
---
## Instructions for Web Usage

To access and use the web app:

1. Open the website:
- [![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-green?logo=streamlit)](https://finalproject-cmpsc-445-tariff-ml-ka3ig4hwa93cmwpuusycyv.streamlit.app)

2. Choose a feature from the sidebar:

- Price Forecasting: Select a commodity to predict future prices.

- Commodity Clustering: View clustering results with visualization.

- Risk Classification: Select a commodity and get risk predictions (High / Medium / Low).

---

## Functionalities and Test Results / Machine Learning Models

| Task                   | Model                | Description                                   |
|------------------------|----------------------|-----------------------------------------------|
| Price Forecasting      | Linear Regression, ARIMA | Predicts future commodity prices     |
| Industry Clustering    | KMeans + PCA         | Clusters commodities by trends                |
| Risk Classification    | Decision Tree        | Categorizes commodities into risk classes     |

Test Results:
- Models load correctly from /models/.

- Forecasted prices align with known trends.

- Clusters are distinct and logically grouped.

- Risk classifier produces consistent predictions across different commodities.
---
## Data Processing

Preprocessing Steps:

- Dropped commodities with excessive missing values.

- Selected the most recent 60 months (5 years) for consistency.

- Scaled features (StandardScaler) for KMeans clustering.

Feature Engineering:

- Created lag-based features for time series forecasting.

- Computed price volatility to categorize risk levels.

- PCA (Principal Component Analysis) applied for visual clustering plots.
---
## Model Development 
| Model                   | Inputs                | Output                                   |
|------------------------|----------------------|-----------------------------------------------|
| Linear Regression	      | Last 5 months of prices | Next month price     |
| KMeans Clustering   |Scaled monthly price trends	| Cluster assignment              |
| Decision Tree Classifier	    | Price volatility	      | Risk level (High/Medium/Low)     |

Algorithms Justification:
- Linear Regression: Interpretable and effective for short-term forecasting.

- KMeans: Best suited for unsupervised commodity behavior grouping.

- Decision Tree: Easy to interpret and efficient for risk prediction.

---
## How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/AnatolyBarabanov/FinalProject-CMPSC-445-Tariff-ML.git
   cd FinalProject-CMPSC-445-Tariff-ML
2. **Install dependencies**
   ```bash
    pip install -r requirements.txt
3. **Run model scripts (optional) If you want to retrain models**
   ```bash
    python train_forecasting.py
    python train_clustering_enhanced.py
    python train_risk_classification.py
4. **Launch the dashboard**
   ```bash
    streamlit run tariff_dashboard1.py
---
## Discussion and conclusions

This was a very interesting project. Especially it was very fun as a group. We were able to work together to make this project exactly the way we wanted it. We quickly determined what we would do and found a data source. Working together allowed to add more to the project.

Conclusions:

- Price forecasting works well for most stable goods.

- Clustering identifies logical groups such as precious metals, agriculture and energy.

- Risk classification effectively identifies goods with high volatility.

Project problems:

- Initially there were problems with regression due to improper cleaning, but after better cleaning, everything went well.

- Creating the site was also difficult, because there is very little experience working with python sites. For a long time, Price Forecasting simply did not load on the site, we were able to solve the problem by adding some edits to the tariff_dashboard1.py code.

- And unstable commodities occasionally caused forecasting errors.
