# Brain Tumor Classification Project

## Overview
This project implements a Convolutional Neural Network (CNN) for classifying brain tumor types from medical imaging using TensorFlow and Keras. The application provides a Streamlit-based web interface for easy image upload and classification.

## Features
- Classifies brain tumor images into 4 categories:
  - Glioma Tumor
  - Meningioma Tumor
  - No Tumor
  - Pituitary Tumor
- Uses a deep learning CNN model
- Interactive web interface with image upload
- Provides classification prediction and confidence score

## Prerequisites
- Python 3.8+
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pillow (PIL)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Dataset
The model is trained on a brain tumor classification dataset with four classes:
- Glioma Tumor
- Meningioma Tumor
- No Tumor
- Pituitary Tumor

Recommended dataset structure:
```
brain_tumor_dataset/
│
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── no_tumor/
    └── pituitary/
```

## Model Architecture
The CNN model consists of:
- 3 Convolutional layers with ReLU activation
- Max Pooling layers for feature reduction
- Dropout layer to prevent overfitting
- Dense layers for classification
- Softmax activation for multi-class output

## Training
- Input image size: 128x128 pixels
- Color mode: Grayscale
- Batch size: 32
- Epochs: 20
- Optimizer: Adam
- Loss function: Categorical Cross-Entropy

## Running the Application

1. Train the model:
```bash
python train_model.py
```

2. Launch the Streamlit app:
```bash
streamlit run app.py
```

## Usage
1. Upload a brain tumor X-ray image
2. The app will display:
   - Predicted tumor type
   - Confidence score

## Limitations
- Accuracy depends on training dataset quality
- Works best with clear, preprocessed medical images
- Requires professional medical interpretation

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
[Specify your license here, e.g., MIT License]

## Acknowledgments
- Dataset source
- Any other credits or references

## Contact
[Your contact information]
```

## dataset File
https://www.kaggle.com/datasets/arifmia/brain-tumor-dataset



tensorflow==2.15.0
keras==2.15.0
streamlit==1.29.0
numpy==1.23.5
Pillow==10.1.0
