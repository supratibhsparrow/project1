"""
Brain Tumor Detection Flask API
================================
This Flask application provides a REST API for brain tumor detection.
It loads a pre-trained Keras model and processes uploaded MRI images.

Endpoints:
----------
GET  /health - Health check endpoint
POST /predict - Upload MRI image and get tumor prediction

Usage:
------
python app.py

The server will start on http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import os
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes (frontend integration)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_PATH = 'model.h5'
IMG_SIZE = 150
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary Tumor']

# Load model at startup
model = None

def load_model():
    """
    Load the pre-trained Keras model.
    Returns None if model file doesn't exist.
    """
    global model
    try:
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading model from {MODEL_PATH}...")
            model = keras.models.load_model(MODEL_PATH)
            logger.info("✓ Model loaded successfully!")
            return True
        else:
            logger.warning(f"Model file '{MODEL_PATH}' not found!")
            logger.warning("Please train the model first using train_model.py")
            return False
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def preprocess_image(image_bytes):
    """
    Preprocess uploaded image for model prediction.
    
    Args:
        image_bytes: Raw image bytes from uploaded file
        
    Returns:
        Preprocessed numpy array ready for model input
    """
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB (in case image is grayscale or has alpha channel)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to numpy array and normalize
        img_array = np.array(image)
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    Returns server status and model availability.
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'message': 'Brain Tumor Detection API is running'
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint.
    Accepts an image file and returns tumor classification.
    
    Request:
        - Method: POST
        - Content-Type: multipart/form-data
        - Body: file (image file)
        
    Response:
        {
            "success": true,
            "prediction": "Glioma",
            "confidence": 0.95,
            "probabilities": {
                "Glioma": 0.95,
                "Meningioma": 0.03,
                "No Tumor": 0.01,
                "Pituitary Tumor": 0.01
            }
        }
    """
    # Check if model is loaded
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please ensure model.h5 exists.'
        }), 500
    
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file provided. Please upload an image.'
        }), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected.'
        }), 400
    
    # Validate file type
    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
    file_ext = file.filename.rsplit('.', 1)[-1].lower()
    
    if file_ext not in allowed_extensions:
        return jsonify({
            'success': False,
            'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}'
        }), 400
    
    try:
        # Read and preprocess image
        logger.info(f"Processing image: {file.filename}")
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        logger.info("Running model prediction...")
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        # Create probabilities dictionary
        probabilities = {
            CLASS_NAMES[i]: float(predictions[0][i])
            for i in range(len(CLASS_NAMES))
        }
        
        logger.info(f"Prediction: {predicted_class} (Confidence: {confidence*100:.2f}%)")
        
        # Return result
        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities
        }), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Brain Tumor Detection API Server")
    print("=" * 60)
    
    # Load model
    model_loaded = load_model()
    
    if not model_loaded:
        print("\n⚠ WARNING: Model not loaded!")
        print("The API will start but predictions will fail.")
        print("Please train the model using train_model.py first.\n")
    
    print("\nStarting Flask server...")
    print("API Endpoints:")
    print("  - GET  http://localhost:5000/health")
    print("  - POST http://localhost:5000/predict")
    print("\nPress CTRL+C to stop the server")
    print("=" * 60 + "\n")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
