import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os
import joblib

# Config
n_lags = 12  # use past 12 months
data_path = os.path.join("data", "CMO-Historical-Data-Monthly-CLEANED.xlsx")
df = pd.read_excel(data_path, sheet_name="Cleaned Data", index_col="Date")
commodities = df.columns.tolist()
os.makedirs("models", exist_ok=True)

summary = []

for commodity in commodities:
    try:
        print(f"\n{'='*50}\nProcessing: {commodity}\n{'='*50}")
        series = df[commodity].dropna()

        # Create lag features
        data = pd.DataFrame({f"lag_{i+1}": series.shift(i+1) for i in range(n_lags)})
        data["target"] = series
        data = data.dropna()

        # Define X and y
        X = data[[f"lag_{i+1}" for i in range(n_lags)]]
        thresholds = [data["target"].mean() - data["target"].std(),
                      data["target"].mean() + data["target"].std()]
        y = pd.cut(data["target"], bins=[-np.inf, *thresholds, np.inf], labels=["Low", "Medium", "High"])

        # Train/test split
        split_idx = int(len(data) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        overfit_flag = (train_acc - test_acc) > 0.1

        report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)

        summary.append({
            "Commodity": commodity,
            "Train Accuracy": round(train_acc, 3),
            "Test Accuracy": round(test_acc, 3),
            "Overfitting?": "Yes" if overfit_flag else "No",
            "Precision": round(report["weighted avg"]["precision"], 3),
            "Recall": round(report["weighted avg"]["recall"], 3),
            "F1-Score": round(report["weighted avg"]["f1-score"], 3)
        })

        # Save model
        model_name = f"risk_classifier_{commodity.replace(' ', '_').replace(',', '').replace('**', '').lower()}.pkl"
        joblib.dump(model, os.path.join("models", model_name))

    except Exception as e:
        print(f"Error processing {commodity}: {e}")

# Export results
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join("data", "risk_classification_summary.csv"))

print("\Summary saved to: risk_classification_summary.csv")
