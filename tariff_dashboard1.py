import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from io import BytesIO

st.set_page_config(page_title="Tariff Impact Dashboard", layout="wide")

# Directories
DATA_DIR = "data"
MODELS_DIR = "models"

# Utility to sanitize commodity names for filenames
def clean_commodity_name(name: str) -> str:
    return name.replace(' ', '_').replace(',', '').replace('**', '').lower()

# Caching loaders
@st.cache_data
def load_summary():
    return pd.read_csv(os.path.join(DATA_DIR, "risk_classification_summary.csv"))

@st.cache_data
def load_clustered_data():
    return pd.read_excel(os.path.join(DATA_DIR, "clustered_commodities.xlsx"), index_col=0)

@st.cache_data
def load_forecast_model(cleaned_name):
    path = os.path.join(MODELS_DIR, f"price_forecast_model_{cleaned_name}.pkl")
    return joblib.load(path)

@st.cache_data
def load_risk_model(cleaned_name):
    path = os.path.join(MODELS_DIR, f"risk_classifier_{cleaned_name}.pkl")
    return joblib.load(path)

@st.cache_data
def load_kmeans_plot():
    img_path = os.path.join(DATA_DIR, "commodity_clusters.png")
    return open(img_path, "rb").read()

# Load data
clustered_df = load_clustered_data()

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "Price Forecasting",
    "Risk Classification",
    "Industry Clustering"
])

# Main layout
st.title("Tariff Impact Dashboard")

if section == "Price Forecasting":
    st.header("Consumer Price Forecasting")

    # List all forecast models
    forecast_files = [
        f for f in os.listdir(MODELS_DIR)
        if f.startswith("price_forecast_model_") and f.endswith(".pkl")
    ]
    cleaned_names = [f[len("price_forecast_model_"):-len(".pkl")] for f in forecast_files]
    display_names = [name.replace("_", " ").title() for name in cleaned_names]

    # User selection
    idx = st.selectbox("Select a commodity to forecast", range(len(display_names)),
                       format_func=lambda i: display_names[i])
    cleaned = cleaned_names[idx]
    pretty = display_names[idx]

    try:
        # Load model
        model = load_forecast_model(cleaned)
        
        # Get price columns (exclude Cluster)
        price_columns = [col for col in clustered_df.columns if col != "Cluster"]
        
        # Custom date parser function
        def parse_date(col_str):
            # First try common date formats without time
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y-%m', '%b-%Y', '%B-%Y']:
                try:
                    return pd.to_datetime(col_str, format=fmt)
                except ValueError:
                    continue
            # If all else fails, try coercing
            return pd.to_datetime(col_str, errors='coerce')
        
        # Parse all column names as dates
        date_series = pd.Series(price_columns).apply(parse_date)
        
        # Find the most recent valid date
        valid_dates = date_series[date_series.notna()]
        if valid_dates.empty:
            st.error("No valid dates found in the dataset columns")
            st.stop()
            
        last_date_idx = valid_dates.argmax()
        last_date = valid_dates.iloc[last_date_idx]
        last_col = price_columns[last_date_idx]
        
        # Improved commodity name matching
        def find_commodity(target, available):
            target_clean = target.lower().replace(',', '').replace(' ', '').strip()
            for commodity in available:
                commodity_clean = commodity.lower().replace(',', '').replace(' ', '').strip()
                if commodity_clean == target_clean:
                    return commodity
            return None

        matched_name = find_commodity(pretty, clustered_df.index)
        if not matched_name:
            st.error(f"""
            Commodity '{pretty}' not matched in dataset. 
            Model name: {pretty}
            Closest matches: {clustered_df.index[clustered_df.index.str.contains(pretty.split()[0], case=False)].tolist()}
            Full list: {clustered_df.index.tolist()}
            """)
            st.stop()

        # Get the last value
        try:
            last_value = clustered_df.loc[matched_name, last_col]
            if pd.isna(last_value):
                st.error(f"No price data available for {matched_name} in {last_col}")
                st.stop()
        except KeyError:
            st.error(f"Data access error for {pretty} (matched as {matched_name})")
            st.stop()

        # Make prediction
        forecast = model.predict(np.array([[last_value]]))[0]
        
        # Display results
        next_month = last_date + pd.DateOffset(months=1)
        st.metric(
            label=f"Forecasted Price for {next_month.strftime('%B %Y')}",
            value=f"${forecast:.2f}",
            delta=f"from ${last_value:.2f} in {last_date.strftime('%B %Y')}"
        )

    except FileNotFoundError:
        st.warning(f"No forecast model found for {pretty}")
    except Exception as e:
        st.error(f"Failed to forecast for {pretty}: {str(e)}")

elif section == "Risk Classification":
    st.header("Risk Classification")

    summary = load_summary()
    st.dataframe(summary, use_container_width=True)

    commodities = clustered_df.index.tolist()
    selected = st.selectbox("Select a commodity to classify risk", commodities)
    cleaned = clean_commodity_name(selected)

    try:
        model = load_risk_model(cleaned)
        # use all historical columns except 'Cluster'
        features = clustered_df.columns.drop("Cluster")
        latest_values = clustered_df.loc[selected, features].values.reshape(1, -1)
        risk = model.predict(latest_values)[0]
        st.metric(label="Predicted Risk Category", value=risk)
    except FileNotFoundError:
        st.warning(f"No risk classifier found for {selected}")

    # Download summary CSV
    csv = summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Risk Summary CSV",
        data=csv,
        file_name="risk_classification_summary.csv",
        mime="text/csv"
    )

elif section == "Industry Clustering":
    st.header("Industry Clustering")

    st.subheader("Cluster Assignments")
    st.dataframe(clustered_df[["Cluster"]], use_container_width=True)

    st.subheader("Cluster Visualization")
    img_bytes = load_kmeans_plot()
    st.image(img_bytes, caption="Commodity Clusters", use_container_width=True)

    # Download clustered commodities
    buf = BytesIO()
    clustered_df.to_excel(buf, index=True)
    st.download_button(
        "Download Clustered Commodities",
        data=buf.getvalue(),
        file_name="clustered_commodities.xlsx"
    )