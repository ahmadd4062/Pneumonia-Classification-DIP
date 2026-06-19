# 🫁 PneumoScan - AI-Powered Pneumonia Detection

> An end-to-end deep learning system for pneumonia detection from chest X-ray images, featuring a 9-step Digital Image Processing (DIP) pipeline with full visualization.


## 🎯 Overview

PneumoScan is an advanced pneumonia classification system that combines:

- **9-Step Digital Image Processing (DIP) Pipeline** - Enhances chest X-ray images through multiple stages including gamma correction, histogram equalization, Otsu thresholding, and bit-plane slicing
- **Convolutional Neural Network (CNN)** - Deep learning model trained on thousands of chest X-ray images
- **Interactive Web Interface** - Built with Streamlit for real-time diagnosis with full pipeline visualization

The system processes chest X-ray images through all DIP stages, then feeds the enhanced image into a CNN, achieving **94%+ accuracy** on the test set.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔬 **9-Step DIP Pipeline** | Each processing stage visualized individually for complete transparency |
| 🤖 **CNN Classification** | Deep learning model with 94%+ accuracy on test data |
| ⚡ **Real-Time Results** | Instant predictions with confidence scores and probability bars |
| 📊 **Dual Edge Detection** | Side-by-side comparison of Sobel vs Laplacian edge detection |
| 🎨 **Interactive UI** | Modern dark-themed interface built with Streamlit |
| 📈 **Visual Analytics** | Probability bars, raw model outputs, and processing history |
| 🏥 **Clinical Disclaimer** | Clear medical disclaimer for responsible use |

## 🔬 DIP Pipeline

The system implements a comprehensive 9-step image processing pipeline:

| Step | Name | Description |
|------|------|-------------|
| 01 | **Source Image** | Raw chest X-ray loaded as grayscale |
| 02 | **Resize 224×224** | Standardized dimensions for CNN input |
| 03 | **Gamma Correction** | γ = 1.5 · Enhances dark lung tissue details |
| 04 | **Histogram Equalization** | Global contrast enhancement · **Used as CNN input** |
| 05 | **Laplacian Edges** | Second-order derivative for fine structure detection |
| 06 | **Binary Threshold** | Fixed threshold T=127 · Baseline segmentation |
| 07 | **Otsu Threshold** | Automatic threshold · Optimal class separation |
| 08 | **Bit-Plane Slicing** | MSB plane 7 isolation · Dominant intensity structure |
| 09 | **Histogram Stretching** | 2-98 percentile stretch · Maximizes dynamic range |


### Pipeline Visualization
Each step is displayed in a 3-column grid, with the histogram-equalized image (Step 4) highlighted as the CNN input.

## 📊 Dataset

The project uses the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle.

### Dataset Statistics

| Split | Normal | Pneumonia | Total |
|-------|--------|-----------|-------|
| **Training** | 1,341 | 3,875 | 5,216 |
| **Validation** | 8 | 8 | 16 |
| **Test** | 234 | 390 | 624 |
| **Total** | 1,583 | 4,273 | 5,856 |

### Dataset Structure
```
dataset/chest_xray/
├── train/
│   ├── NORMAL/      (1,341 images)
│   └── PNEUMONIA/   (3,875 images)
├── val/
│   ├── NORMAL/      (8 images)
│   └── PNEUMONIA/   (8 images)
└── test/
    ├── NORMAL/      (234 images)
    └── PNEUMONIA/   (390 images)
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning)

### Step-by-Step Installation

#### 1. Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running the Web Application
```bash
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`

### How to Use the App
1. **Upload** a chest X-ray image (JPG, JPEG, or PNG)
2. **View** the 9-step DIP pipeline processing
3. **See** the diagnosis result with confidence score
4. **Compare** Sobel vs Laplacian edge detection
5. **Review** probability bars and raw model output

## 🧠 Model Architecture

The CNN architecture used for classification:

```
Layer (type)                 Output Shape              Param #
=================================================================
Conv2D (32 filters)          (None, 222, 222, 32)      320
MaxPooling2D                 (None, 111, 111, 32)      0
Conv2D (64 filters)          (None, 109, 109, 64)      18,496
MaxPooling2D                 (None, 54, 54, 64)        0
Conv2D (128 filters)         (None, 52, 52, 128)       73,856
MaxPooling2D                 (None, 26, 26, 128)       0
Conv2D (128 filters)         (None, 24, 24, 128)       147,584
MaxPooling2D                 (None, 12, 12, 128)       0
Flatten                      (None, 18432)             0
Dropout (0.5)                (None, 18432)             0
Dense (512 units)            (None, 512)               9,437,696
Dense (1 unit)               (None, 1)                 513
=================================================================
Total params: 9,678,465
Trainable params: 9,678,465
Non-trainable params: 0
```

### Training Configuration
- **Input Size**: 224×224 (grayscale)
- **Batch Size**: 16
- **Epochs**: 10 (can be increased for better accuracy)
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy


## 📈 Results

### Model Performance
- **Training Accuracy**: 98.7%
- **Validation Accuracy**: 93.7%
- **Test Accuracy**: 94.0%


### Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 98.0% |
| **Precision** | 94.0% |
| **Recall** | 95.9% |
| **F1-Score** | 94.9% |
| **AUC-ROC** | 94.2% |

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.8+ | Core programming language |
| **TensorFlow** | 2.0+ | Deep learning framework |
| **Keras** | 2.0+ | Neural network API |
| **OpenCV** | 4.5+ | Digital image processing |
| **Streamlit** | 1.0+ | Web application framework |
| **NumPy** | 1.19+ | Numerical computations |
| **Matplotlib** | 3.3+ | Data visualization |
| **Scikit-learn** | 0.24+ | Model evaluation metrics |
| **Pillow** | 8.0+ | Image handling |

## 📁 Project Structure

```
Pneumonia-classification-dip/
├── app.py                    # Main Streamlit application
├── chunk_2_train.py          # Model training script
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .gitignore               # Git ignore file
├── models/
│   └── pneumonia_model.h5   # Trained model (generated)
└── dataset/
    └── chest_xray/          # Dataset folder
        ├── train/
        ├── val/
        └── test/
```

## ⚠️ Disclaimer

> **Always consult a qualified healthcare professional** for accurate diagnosis and treatment decisions. The developers assume no responsibility for any misuse or misinterpretation of results.

## 🙏 Acknowledgments

- [Kaggle](https://www.kaggle.com/) for providing the chest X-ray dataset
- [TensorFlow](https://www.tensorflow.org/) for the deep learning framework
- [Streamlit](https://streamlit.io/) for the web application framework

