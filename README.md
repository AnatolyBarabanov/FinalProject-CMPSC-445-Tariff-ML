# FinalProject-CMPSC-445-Tariff-ML

This project analyzes the impact of tariffs on global commodity markets using machine learning techniques. 

It includes consumer price forecasting, industry clustering, and risk classification — all integrated into an interactive Streamlit dashboard.

🌐 Live Website: [![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-green?logo=streamlit)](https://finalproject-cmpsc-445-tariff-ml-ka3ig4hwa93cmwpuusycyv.streamlit.app)

---

## Project Goals

- Predict future commodity prices.
- Group commodities with similar behavior.
- Classify commodities into risk categories.
- Provide visual insights via a web dashboard.

---

## Dataset

- **Source:** [World Bank – Commodity Markets](https://www.worldbank.org/en/research/commodity-markets)
- **File:** `CMO-Historical-Data-Monthly.xlsx`
- Monthly commodity prices across agriculture, energy, and metals sectors.

---

## Project Structure

FinalProject-CMPSC-445-Tariff-ML/

├── models/                # Trained machine learning models (.pkl)

│    └── price_forecast_model.pkl

│    └── clustering_model.pkl

│    └── risk_classification_model.pkl

├── data/                  # Dataset (World Bank CMO data)

│    └── CMO-Historical-Data-Monthly.xlsx

├── requirements.txt       # Python libraries to install

└── README.md              # Project description

and scripts:

   1. tariff_models.py — First to run, Shared Model Functions

Contains helper functions for training and saving machine learning models.

Functions are imported into the training scripts (train_classifier.py, train_kmean.py, train_regression.py) to avoid code duplication.

   2. train_regression.py — Price Forecasting Model Trainer

Trains a Linear Regression model to predict future commodity prices.

Loads historical commodity price data from CMO-Historical-Data-Monthly.xlsx.

Fits the model using previous months' prices as input features.

Saves the trained model to:
➔ models/price_forecast_model.pkl

Run this file after tariff_models.py to create the forecasting model.

   3. train_kmean.py — Clustering Model Trainer
   
Trains a KMeans clustering model to group commodities based on their price trends.

Uses Principal Component Analysis (PCA) for cluster visualization.

Automatically determines the optimal number of clusters using Silhouette Score.

Saves:

Clustering model ➔ models/clustering_model.pkl

Cluster visualization ➔ commodity_clusters.png

Run this file third to create the clustering model.

   4. train_classifier.py — Risk Classification Model Trainer
   
Trains a Decision Tree Classifier to predict commodity risk levels (High / Medium / Low).

Defines risk categories based on recent price volatility.

Saves the trained classifier model to:
➔ models/risk_classification_model.pkl

Run this file fourth to create the risk prediction model.

   5. tariff_dashboard.py — Streamlit Web Application
Combines all three models into an interactive Streamlit dashboard.

Allows users to:

Forecast future commodity prices.

View cluster analysis of commodities.

Predict commodity risk levels.

Loads models from the models/ folder.

Displays interactive charts, tables, and predictions.

## Machine Learning Models

| Task                   | Model                | Description                                   |
|------------------------|----------------------|-----------------------------------------------|
| Price Forecasting      | Linear Regression, ARIMA | Predicts future commodity prices     |
| Industry Clustering    | KMeans + PCA         | Clusters commodities by trends                |
| Risk Classification    | Decision Tree        | Categorizes commodities into risk classes     |

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
    streamlit run tariff_dashboard.py
