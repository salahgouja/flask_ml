from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the pre-trained Random Forest model
model = joblib.load('random_forest.pkl')  # Path to the model

# Get the expected feature names from the model
expected_features = model.best_estimator_.feature_names_in_

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Handle input data
    if request.is_json:
        data = request.get_json()  # JSON payload
    else:
        data = request.form  # Form data

    # Prepare input data
    processed_data = {
        "State": data['State'],  # Pass raw State value; we'll one-hot encode it
        "Account length": int(data['Account_length']),
        "International plan": data['International_plan'],
        "Voice mail plan": data['Voice_mail_plan'],
        "Number vmail messages": int(data['Number_vmail_messages']),
        "Total day minutes": float(data['Total_day_minutes']),
        "Total day calls": int(data['Total_day_calls']),
        "Total eve minutes": float(data['Total_eve_minutes']),
        "Total eve calls": int(data['Total_eve_calls']),
        "Total night minutes": float(data['Total_night_minutes']),
        "Total night calls": int(data['Total_night_calls']),
        "Total intl minutes": float(data['Total_intl_minutes']),
        "Total intl calls": int(data['Total_intl_calls']),
        "Customer service calls": int(data['Customer_service_calls'])
    }

    # Convert to DataFrame
    input_data = pd.DataFrame([processed_data])

    # One-hot encode the State column
    input_data = pd.get_dummies(input_data, columns=['State'])

    # Handle binary features for consistency
    input_data['International plan_Yes'] = input_data['International plan'].apply(lambda x: 1 if x == "Yes" else 0)
    input_data['Voice mail plan_Yes'] = input_data['Voice mail plan'].apply(lambda x: 1 if x == "Yes" else 0)

    # Drop the original binary columns
    input_data.drop(['International plan', 'Voice mail plan'], axis=1, inplace=True)

    # Align input data with the expected feature names
    input_data = input_data.reindex(columns=expected_features, fill_value=0)

    # Predict churn using the model
    prediction = model.predict(input_data)

    # Prepare the result
    result = "Churn" if prediction[0] else "No Churn"

    # Return the prediction as JSON
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True , host="0.0.0.0", port="5000")
