# Multiclass Chest X-Ray Image Classification

A deep learning project for multiclass classification of chest X-ray images using convolutional neural networks (CNNs). This project aims to automatically classify chest X-ray scans into different disease categories to assist in medical image analysis and diagnosis.

---

## 🚀 Features

- Multiclass chest X-ray image classification
- Deep learning-based CNN model
- Image preprocessing and augmentation
- Model training and evaluation
- Performance visualization with accuracy/loss graphs
- Easy-to-use project structure

---

## 📂 Project Structure

```bash
multiclass_chest_x_ray_image_classification/
│── dataset/                 # Chest X-ray dataset
│── models/                  # Saved trained models
│── src/                     # Source code
│   ├── train.py             # Training script
│── app.py                   # Streamlit script
│── requirements.txt         # Python dependencies
│── README.md                # Project documentation
```

---



## 🧠 Technologies Used

Python
TensorFlow
NumPy
Matplotlib
Scikit-learn

---

## 📊 Dataset

The project uses chest X-ray image datasets containing multiple disease classes such as:
Normal
Viral Pneumonia
COVID-19

Ensure the dataset is organized into separate folders for each class before training.

---

## ⚙️ Installation

Clone the repository:
git clone https://github.com/harishgundapuUD/multiclass_chest_x_ray_image_classification.git
cd multiclass_chest_x_ray_image_classification

Install dependencies:
pip install -r requirements.txt

---

## ▶️ Training the Model

Run the training script:
python src/train.py

---

## 🔍 Making Predictions

Use the streamlit script:
python -m streamlit run .\app.py

---

## 📈 Model Evaluation

The project evaluates model performance using:
Accuracy
Precision
Recall
F1-score
Confusion Matrix

---

## 🩺 Applications

Automated medical image analysis
Early disease detection
AI-assisted radiology systems
Healthcare research

---

## 🔗 Repository

https://github.com/harishgundapuUD/multiclass_chest_x_ray_image_classification
