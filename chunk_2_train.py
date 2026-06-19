"""
CHUNK 2: Train CNN Model - NO AUGMENTATION
Save this as: chunk_2_train.py
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from app import preprocess_image, enhance_image

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10

# Path to your dataset
DATASET_PATH = "dataset/chest_xray/"

def load_data(data_path):
    """
    Load images - NO AUGMENTATION
    """
    images = []
    labels = []
    
    # Load NORMAL (label 0)
    normal_path = os.path.join(data_path, 'NORMAL')
    if os.path.exists(normal_path):
        for img_file in os.listdir(normal_path):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(normal_path, img_file)
                
                # Apply DIP preprocessing
                img = preprocess_image(img_path, IMG_SIZE)
                if img is not None:
                    # Apply enhancement
                    img = enhance_image(img)
                    
                    # Normalize
                    img = img / 255.0
                    img = img.reshape(224, 224, 1)
                    
                    images.append(img)
                    labels.append(0)  # Normal
    
    # Load PNEUMONIA (label 1)
    pneumonia_path = os.path.join(data_path, 'PNEUMONIA')
    if os.path.exists(pneumonia_path):
        for img_file in os.listdir(pneumonia_path):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(pneumonia_path, img_file)
                
                # Apply DIP preprocessing
                img = preprocess_image(img_path, IMG_SIZE)
                if img is not None:
                    # Apply enhancement
                    img = enhance_image(img)
                    
                    # Normalize
                    img = img / 255.0
                    img = img.reshape(224, 224, 1)
                    
                    images.append(img)
                    labels.append(1)  # Pneumonia
    
    return np.array(images), np.array(labels)

def create_cnn_model():
    """
    Create CNN model
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def plot_training_history(history):
    """
    Plot training history
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].set_title('Model Accuracy')
    axes[0].grid(True)
    
    axes[1].plot(history.history['loss'], label='Train Loss', marker='o')
    axes[1].plot(history.history['val_loss'], label='Validation Loss', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].set_title('Model Loss')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("="*50)
    print("PNEUMONIA DETECTION - TRAINING (NO AUGMENTATION)")
    print("="*50)
    
    # Check dataset
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at {DATASET_PATH}")
        print("Please update DATASET_PATH variable")
        exit()
    
    # Load training data
    print("\n📂 Loading training data...")
    train_path = os.path.join(DATASET_PATH, 'train')
    X_train, y_train = load_data(train_path)
    print(f"✅ Training images: {len(X_train)}")
    print(f"   Normal (0): {sum(y_train==0)}")
    print(f"   Pneumonia (1): {sum(y_train==1)}")
    
    # Load validation data
    print("\n📂 Loading validation data...")
    val_path = os.path.join(DATASET_PATH, 'val')
    X_val, y_val = load_data(val_path)
    print(f"✅ Validation images: {len(X_val)}")
    
    print(f"\n📊 Data shapes:")
    print(f"   X_train: {X_train.shape}")
    print(f"   y_train: {y_train.shape}")
    print(f"   X_val: {X_val.shape}")
    
    # Create model
    print("\n🤖 Creating CNN model...")
    model = create_cnn_model()
    model.summary()
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    print("\n🏋️ Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/pneumonia_model.h5')
    print("\n✅ Model saved to 'models/pneumonia_model.h5'")
    
    # Plot results
    plot_training_history(history)
    
    # Final accuracy
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    print(f"\n📈 Final Results:")
    print(f"   Training Accuracy: {train_acc:.4f}")
    print(f"   Validation Accuracy: {val_acc:.4f}")