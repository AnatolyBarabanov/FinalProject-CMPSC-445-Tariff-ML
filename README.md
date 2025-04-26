# FinalProject-CMPSC-445-Tariff-ML

This project analyzes the impact of tariffs on global commodity markets using machine learning techniques. 

It includes consumer price forecasting, industry clustering, and risk classification — all integrated into an interactive Streamlit dashboard.

---

## 📌 Project Goals

- Predict future commodity prices.
- Group commodities with similar behavior.
- Classify commodities into risk categories.
- Provide visual insights via a web dashboard.

---

## 📊 Dataset

- **Source:** [World Bank – Commodity Markets](https://www.worldbank.org/en/research/commodity-markets)
- **File:** `CMO-Historical-Data-Monthly.xlsx`
- Monthly commodity prices across agriculture, energy, and metals sectors.

---

## 🔧 Project Structure





---

## 🧠 Machine Learning Models

| Task                   | Model                | Description                                   |
|------------------------|----------------------|-----------------------------------------------|
| Price Forecasting      | Linear Regression, ARIMA | Predicts future commodity prices     |
| Industry Clustering    | KMeans + PCA         | Clusters commodities by trends                |
| Risk Classification    | Decision Tree        | Categorizes commodities into risk classes     |

---

## 🚀 How to Run

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
