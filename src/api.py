from flask import Flask, request, jsonify
import joblib
import numpy as np
from pathlib import Path
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
MODEL_DIR = Path("models/trained")
model = joblib.load(MODEL_DIR / "gradient_boosting_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")

# Load model metadata for feature information
metadata = pd.read_json(MODEL_DIR / "model_metadata.json", typ='series')

# Define severity mapping for human-readable output
SEVERITY_MAPPING = {
    0: 'Minor',
    1: 'Moderate',
    2: 'Major',
    3: 'Severe'
}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model": "gradient_boosting"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for injury risk prediction"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Required features for prediction
        required_features = [
            'Age', 'FIFA rating',
            'Form_Before_Injury', 'Total_Injuries',
            'Avg_Injury_Duration', 'Overall_Risk_Score',
            'Pre_Injury_GD_Trend', 'Recent_Opposition_Strength',
            'Age_Risk_Score', 'Position_Risk_Score', 'History_Risk_Score',
            'Form_After_Return', 'Performance_Impact',
            'Pre_Injury_GD_Volatility', 'Team_Performance_During_Absence'
        ]
        
        # Validate input features
        missing_features = [f for f in required_features if f not in data]
        if missing_features:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing_features
            }), 400
        
        # Prepare input data
        input_data = np.array([data[f] for f in required_features]).reshape(1, -1)
        
        # Scale input features
        scaled_input = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_input)[0]
        probabilities = model.predict_proba(scaled_input)[0]
        
        # Prepare response
        response = {
            "prediction": {
                "severity": SEVERITY_MAPPING[prediction],
                "severity_code": int(prediction)
            },
            "probabilities": {
                SEVERITY_MAPPING[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            },
            "risk_factors": {
                "age_risk": data['Age_Risk_Score'],
                "history_risk": data['History_Risk_Score'],
                "overall_risk": data['Overall_Risk_Score']
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Endpoint for model information"""
    return jsonify({
        "model_type": "gradient_boosting",
        "best_score": float(metadata['best_score']),
        "features": required_features,
        "severity_levels": SEVERITY_MAPPING
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)