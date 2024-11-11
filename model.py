import pickle
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Example: load your dataset
data = pd.read_csv('C:/Users/abhis/OneDrive/Desktop/python/ML ca2/gold-prices.csv')

# Example: preprocess your data, assuming the data has columns like 'Date' and 'Price'
# Convert Date to datetime format and extract year, month, and day as features
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Define features and target
X = data[['Year', 'Month', 'Day']]
y = data['Price']

# Train a model
model = RandomForestRegressor()
model.fit(X, y)

# Save the model
with open('gold_price_predictor.pkl', 'wb') as file:
    pickle.dump(model, file)
