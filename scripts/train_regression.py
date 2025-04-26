# train_regression_all_commodities.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

# 1. Load cleaned data
print("Loading cleaned data...")
data_path = os.path.join("data", "CMO-Historical-Data-Monthly-CLEANED.xlsx")
df = pd.read_excel(data_path, sheet_name="Cleaned Data", index_col="Date")

# 2. List of all commodities to process
commodities = [
    "Crude oil, average", "Crude oil, Brent", "Crude oil, Dubai", 
    "Crude oil, WTI", "Coal, Australian", "Coal, South African", 
    "Natural gas, US", "Natural gas, Europe", "Liquefied natural gas, Japan", 
    "Natural gas index", "Cocoa", "Coffee, Arabica", "Coffee, Robusta", 
    "Tea, avg 3 auctions", "Tea, Colombo", "Tea, Kolkata", "Tea, Mombasa", 
    "Coconut oil", "Groundnuts", "Fish meal", "Groundnut oil", 
    "Palm oil", "Soybeans", "Soybean oil", "Soybean meal", "Barley", 
    "Maize", "Sorghum", "Rice, Thai 5%", "Rice, Thai 25%", "Rice, Thai A.1", 
    "Wheat, US SRW", "Wheat, US HRW", "Banana, US", "Orange", "Beef", 
    "Chicken", "Lamb", "Shrimps, Mexican", "Sugar, EU", "Sugar, US", 
    "Sugar, world", "Tobacco, US import u.v.", "Logs, Cameroon", 
    "Logs, Malaysian", "Sawnwood, Cameroon", "Sawnwood, Malaysian", 
    "Plywood", "Cotton, A Index", "Rubber, RSS3", "Phosphate rock", 
    "DAP", "TSP", "Urea", "Potassium chloride", "Aluminum", 
    "Iron ore, cfr spot", "Copper", "Lead", "Tin", "Nickel", "Zinc", 
    "Gold", "Platinum", "Silver"
]

# 3. Create models directory if it doesn't exist
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# 4. Process each commodity
for commodity in commodities:
    print(f"\n{'='*50}")
    print(f"Processing commodity: {commodity}")
    print(f"{'='*50}")
    
    try:
        # Prepare data for this commodity
        commodity_data = df[[commodity]].dropna()
        commodity_data['previous_month'] = commodity_data[commodity].shift(1)
        commodity_data = commodity_data.dropna()

        X = commodity_data[['previous_month']]
        y = commodity_data[commodity]

        # Split data chronologically
        test_size = 0.2
        split_idx = int(len(commodity_data) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"\nData split:")
        print(f"- Training set: {len(X_train)} samples ({X_train.index.min()} to {X_train.index.max()})")
        print(f"- Test set: {len(X_test)} samples ({X_test.index.min()} to {X_test.index.max()})")

        # Train model
        print("\nTraining model...")
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluate model
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        print("\nModel Performance:")
        print(f"- Train MAE: {mean_absolute_error(y_train, train_pred):.2f}")
        print(f"- Test MAE: {mean_absolute_error(y_test, test_pred):.2f}")
        print(f"- Train RMSE: {mean_squared_error(y_train, train_pred)**0.5:.2f}")
        print(f"- Test RMSE: {mean_squared_error(y_test, test_pred)**0.5:.2f}")
        print(f"- R-squared: {model.score(X_test, y_test):.2f}")

        # Save model
        model_filename = f"price_forecast_model_{commodity.replace(' ', '_').replace(',', '').replace('**', '').lower()}.pkl"
        model_path = os.path.join(models_dir, model_filename)
        joblib.dump(model, model_path)
        print(f"\nModel saved to: {model_path}")

        # Example prediction
        if len(X) > 0:
            last_price = X.iloc[-1, 0]
            next_month_pred = model.predict([[last_price]])[0]
            print(f"\nExample prediction for next month:")
            print(f"- Last known price ({X.index[-1].strftime('%m/%d/%Y')}): {last_price:.2f}")
            print(f"- Predicted next month price: {next_month_pred:.2f}")
    
    except Exception as e:
        print(f"\nError processing {commodity}: {str(e)}")
        continue

print("\nCompleted processing all commodities!")