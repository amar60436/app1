#!/usr/bin/env python
# coding: utf-8

# In[3]:


from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('linear_regression_model.pkl')

# Define a route to render the HTML form
@app.route('/')
def index():
    return render_template('index.html', prediction=None)

# Define a route to handle form submissions and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input from the HTML form
        hours_studied = float(request.form['hours'])
        attendance = float(request.form['attendance'])
        
        # Make a prediction using the model
        predicted_score = model.predict(np.array([[hours_studied, attendance]]))[0]
        
        # Render the HTML template with the prediction
        return render_template('index.html', prediction=predicted_score)
    
    except Exception as e:
        return render_template('index.html', prediction=None, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)

