# Brain Tumor Detection Backend

This directory contains the backend components for the Brain Tumor Detection application, including model training and Flask API.

## 📁 Contents

- `train_model.py` - CNN model training script
- `app.py` - Flask REST API server
- `requirements.txt` - Python dependencies
- `model.h5` - Trained model file (generated after training)

## 🧠 About the Model

The brain tumor detection model is a Convolutional Neural Network (CNN) that classifies MRI brain scans into four categories:

1. **Glioma** - Most common malignant brain tumor
2. **Meningioma** - Tumor that forms on membranes covering the brain
3. **Pituitary Tumor** - Tumor in the pituitary gland
4. **No Tumor** - Healthy brain scan

### Model Architecture

- 4 Convolutional blocks with Batch Normalization and MaxPooling
- Dropout layers for regularization (prevent overfitting)
- Dense layers for classification
- Softmax activation for multi-class output
- Input: 150x150x3 RGB images
- Output: 4-class probability distribution

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster training

### Installation

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Training the Model

### Option 1: Local Training

1. **Download the dataset:**
   - Dataset: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
   - Extract to a folder named `dataset` in the backend directory

2. **Expected folder structure:**
   ```
   backend/
   ├── dataset/
   │   ├── Training/
   │   │   ├── glioma/
   │   │   ├── meningioma/
   │   │   ├── notumor/
   │   │   └── pituitary/
   │   └── Testing/
   │       ├── glioma/
   │       ├── meningioma/
   │       ├── notumor/
   │       └── pituitary/
   ├── train_model.py
   └── app.py
   ```

3. **Run training:**
   ```bash
   python train_model.py
   ```

4. **Training output:**
   - `model.h5` - Final trained model
   - `best_model.h5` - Best model during training
   - `training_history.png` - Accuracy/loss visualization

### Option 2: Google Colab Training (Recommended)

Training on Google Colab provides free GPU access for faster training:

1. **Upload `train_model.py` to Google Colab**
2. **Upload dataset to Google Drive or download directly in Colab:**
   ```python
   !pip install kaggle
   !kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
   !unzip brain-tumor-mri-dataset.zip -d dataset
   ```

3. **Install dependencies in Colab:**
   ```python
   !pip install tensorflow pillow numpy matplotlib
   ```

4. **Update `DATA_DIR` path in the script to match your dataset location**

5. **Run all cells to train**

6. **Download `model.h5` file** to your local `backend/` directory

## 🌐 Running the API Server

1. **Ensure `model.h5` exists in the backend directory**

2. **Start the Flask server:**
   ```bash
   python app.py
   ```

3. **Server will start on:** `http://localhost:5000`

### API Endpoints

#### Health Check
```bash
GET http://localhost:5000/health

Response:
{
  "status": "healthy",
  "model_loaded": true,
  "message": "Brain Tumor Detection API is running"
}
```

#### Predict Tumor
```bash
POST http://localhost:5000/predict
Content-Type: multipart/form-data
Body: file=<image_file>

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
```

### Testing with cURL

```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@/path/to/mri_image.jpg"
```

## 🔧 Configuration

### Model Parameters (train_model.py)
- `IMG_SIZE` - Image dimensions (default: 150)
- `BATCH_SIZE` - Training batch size (default: 32)
- `EPOCHS` - Training epochs (default: 25)
- `DATA_DIR` - Dataset directory path

### API Configuration (app.py)
- `MODEL_PATH` - Path to trained model file
- `HOST` - Server host (default: 0.0.0.0)
- `PORT` - Server port (default: 5000)

## 📝 Model Performance

Expected performance metrics:
- Training Accuracy: ~95-98%
- Validation Accuracy: ~92-95%
- Test Accuracy: ~90-93%

Note: Performance depends on dataset quality and training duration.

## 🐛 Troubleshooting

### "Model not loaded" error
- Ensure `model.h5` exists in the backend directory
- Train the model using `train_model.py` first

### Memory errors during training
- Reduce `BATCH_SIZE` in `train_model.py`
- Use Google Colab with GPU for better performance

### CORS errors from frontend
- CORS is enabled by default in `app.py`
- Ensure Flask server is running on port 5000

### Import errors
- Activate virtual environment
- Reinstall dependencies: `pip install -r requirements.txt`

## 📚 Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Keras CNN Guide](https://keras.io/guides/sequential_model/)

## 📄 License

This project is for educational purposes only. The dataset is subject to its original license terms.
