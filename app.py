from flask import Flask, render_template, request
import datetime as dt
import pickle
import numpy as np
import os

app = Flask(__name__)

# Path to the model file
model_path = 'gold_price_predictor.pkl'

# Check if the model file exists before loading it
if os.path.exists(model_path):
    model = pickle.load(open(model_path, 'rb'))
else:
    raise FileNotFoundError(f"The model file '{model_path}' was not found. Please make sure the model is saved correctly.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    date_str = request.form['date']
    try:
        # Convert the input date to a datetime object
        future_date = dt.datetime.strptime(date_str, '%Y-%m-%d').date()
        today = dt.date.today()

        # Check if the date is in the future
        if future_date <= today:
            error = "Please select a future date."
            return render_template('index.html', error=error)

        # Call the predict_price function
        prediction = predict_price(future_date)

        return render_template('index.html', prediction=prediction, date=future_date)

    except ValueError:
        error = "Invalid date format. Please enter a valid date."
        return render_template('index.html', error=error)

def predict_price(future_date):
    """
    Function to predict the gold price for a given future date.
    The model takes the year, month, and day as features to make a prediction.
    """
    # Extract features from the future_date (year, month, day)
    year = future_date.year
    month = future_date.month
    day = future_date.day

    # Example: Create a feature vector for prediction (this should match the model's training data structure)
    features = np.array([[year, month, day]])

    # Get the prediction from the model
    prediction = model.predict(features)

    # Return the predicted price (round it to 2 decimal places)
    return round(prediction[0], 2)

if __name__ == '__main__':
    app.run(debug=True)
