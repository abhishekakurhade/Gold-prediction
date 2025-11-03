from flask import Flask, render_template, request
import datetime as dt
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model
model_path = 'gold_price_predictor.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError("Model not found! Run model_train.py first.")

model = pickle.load(open(model_path, 'rb'))

# Load model accuracy (if available)
accuracy_info = "N/A"
if os.path.exists('model_metrics.txt'):
    with open('model_metrics.txt', 'r') as f:
        accuracy_info = f.read()

@app.route('/')
def index():
    return render_template('index.html', accuracy=accuracy_info)

@app.route('/predict', methods=['POST'])
def predict():
    date_str = request.form['date']
    try:
        future_date = dt.datetime.strptime(date_str, '%Y-%m-%d').date()
        today = dt.date.today()

        if future_date <= today:
            return render_template('index.html', error="Please select a future date.", accuracy=accuracy_info)

        features = np.array([[future_date.year, future_date.month, future_date.day]])
        prediction = model.predict(features)[0]
        prediction = round(float(prediction), 2)

        return render_template('index.html', prediction=prediction, date=future_date, accuracy=accuracy_info)

    except ValueError:
        return render_template('index.html', error="Invalid date format. Please enter a valid date.", accuracy=accuracy_info)

if __name__ == '__main__':
    app.run(debug=True)
