import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime

# Load dataset
data = pd.read_csv("gold-prices.csv")

# Detect date and price columns automatically
date_col = None
for col in data.columns:
    if 'date' in col.lower():
        date_col = col
        break

if date_col is None:
    raise ValueError("No date column found in the dataset!")

data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
data = data.dropna(subset=[date_col])

price_col = None
for col in data.columns:
    if 'price' in col.lower() or 'gold' in col.lower():
        price_col = col
        break

if price_col is None:
    raise ValueError("No price column found in the dataset!")

# Extract date features
data['Year'] = data[date_col].dt.year
data['Month'] = data[date_col].dt.month
data['Day'] = data[date_col].dt.day

X = data[['Year', 'Month', 'Day']]
y = data[price_col]

# Split for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Save model + accuracy
pickle.dump(model, open('gold_price_predictor.pkl', 'wb'))
with open('model_metrics.txt', 'w') as f:
    f.write(f"R2 Score: {r2:.4f}\nRMSE: {rmse:.4f}")

print("✅ Model trained successfully!")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print("Metrics saved to model_metrics.txt")
