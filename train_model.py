"""
Brain Tumor Detection Model Training Script
============================================
This script trains a Convolutional Neural Network (CNN) to classify brain MRI images
into four categories: Glioma, Meningioma, Pituitary Tumor, and No Tumor.

Dataset Structure Expected:
----------------------------
Training/
    glioma/
        image1.jpg
        image2.jpg
        ...
    meningioma/
        image1.jpg
        ...
    pituitary/
        image1.jpg
        ...
    notumor/
        image1.jpg
        ...

Testing/
    (same structure as Training/)

Dataset Source: Brain Tumor MRI Dataset (Kaggle)
Link: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Google Colab Setup:
-------------------
1. Upload this script to Colab
2. Mount Google Drive or upload dataset
3. Install dependencies: !pip install tensorflow pillow numpy matplotlib
4. Update DATA_DIR path to your dataset location
5. Run all cells
6. Download the generated model.h5 file
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

# Configuration
IMG_SIZE = 150  # Image dimensions (150x150 pixels)
BATCH_SIZE = 32
EPOCHS = 25
DATA_DIR = 'dataset'  # Update this path to your dataset location
MODEL_PATH = 'model.h5'

# Class labels
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def create_data_generators():
    """
    Create data generators with augmentation for training and validation.
    Data augmentation helps prevent overfitting by creating variations of training images.
    """
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,           # Normalize pixel values to [0,1]
        rotation_range=20,        # Randomly rotate images up to 20 degrees
        width_shift_range=0.2,    # Randomly shift images horizontally
        height_shift_range=0.2,   # Randomly shift images vertically
        shear_range=0.2,          # Shear transformation
        zoom_range=0.2,           # Random zoom
        horizontal_flip=True,     # Randomly flip images horizontally
        fill_mode='nearest',      # Fill strategy for new pixels
        validation_split=0.2      # Use 20% of training data for validation
    )
    
    # Validation data (no augmentation, only rescaling)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'Training'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    # Load validation data
    validation_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'Training'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    # Load test data
    test_generator = test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'Testing'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator

def build_cnn_model():
    """
    Build a Convolutional Neural Network for brain tumor classification.
    
    Architecture:
    - 3 Convolutional blocks with MaxPooling
    - Batch Normalization for training stability
    - Dropout layers to prevent overfitting
    - Dense layers for classification
    - Softmax activation for multi-class output
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output Layer (4 classes: glioma, meningioma, notumor, pituitary)
        layers.Dense(4, activation='softmax')
    ])
    
    return model

def plot_training_history(history):
    """
    Visualize training and validation accuracy/loss over epochs.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'")
    plt.show()

def train_model():
    """
    Main training function.
    """
    print("=" * 60)
    print("Brain Tumor Detection Model Training")
    print("=" * 60)
    
    # Create data generators
    print("\n[1/5] Loading and preparing data...")
    train_gen, val_gen, test_gen = create_data_generators()
    
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    print(f"Classes: {list(train_gen.class_indices.keys())}")
    
    # Build model
    print("\n[2/5] Building CNN model...")
    model = build_cnn_model()
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    # Define callbacks
    callbacks = [
        # Stop training if validation loss doesn't improve for 5 epochs
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate if validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        # Save best model during training
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\n[3/5] Training model...")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n[4/5] Evaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(test_gen)
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save final model
    print(f"\n[5/5] Saving model to '{MODEL_PATH}'...")
    model.save(MODEL_PATH)
    print(f"✓ Model saved successfully!")
    
    # Plot training history
    print("\nGenerating training visualizations...")
    plot_training_history(history)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nFiles generated:")
    print(f"  - {MODEL_PATH} (final model)")
    print(f"  - best_model.h5 (best model during training)")
    print(f"  - training_history.png (training visualization)")
    print(f"\nModel is ready for deployment! 🎉")
    
    return model, history

if __name__ == "__main__":
    # Check if dataset exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory '{DATA_DIR}' not found!")
        print("\nPlease download the Brain Tumor MRI Dataset:")
        print("https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
        print(f"\nExtract it and update DATA_DIR in this script.")
    else:
        # Start training
        model, history = train_model()
