# Breast Cancer Detection using SVM

## Project Overview

This project predicts whether a breast tumor is **Benign** or **Malignant** using **Machine Learning (Support Vector Machine - SVM)** and deploys the model using **Streamlit**.

The model is trained on the **Breast Cancer Wisconsin Dataset** available in Scikit-Learn and provides real-time tumor classification through an interactive dashboard.

---

## Features

- Data Visualization Dashboard
- Feature Scaling using StandardScaler
- Support Vector Machine (SVM) Classification
- Model Evaluation
- Confusion Matrix
- Probability Prediction
- Interactive Streamlit Web App
- Real-Time Tumor Classification

---

## Dataset

Dataset Used:

**Breast Cancer Wisconsin Dataset**

Source:

Scikit-Learn Built-in Dataset

Target Classes:

- 0 → Malignant (Cancerous)
- 1 → Benign (Non-Cancerous)

Dataset Information:

- Samples: 569
- Features: 30

Example Features:

- Mean Radius
- Mean Texture
- Mean Perimeter
- Mean Area
- Mean Smoothness

---

## Project Workflow

### 1. Data Collection

Load dataset using:

```python
load_breast_cancer()
```

### 2. Data Preprocessing

Performed:

- Feature Scaling using StandardScaler
- Train-Test Split

### 3. Exploratory Data Analysis

Visualizations included:

- Tumor Distribution Pie Chart
- Correlation Heatmap
- Radius vs Texture Scatter Plot
- Prediction Probability Graph

### 4. Model Training

Algorithm Used:

```python
SVC(kernel='rbf')
```

Train-Test Split:

```text
80% Train
20% Test
```

### 5. Model Evaluation

Metrics Used:

- Accuracy
- Confusion Matrix
- Classification Report

Typical Model Performance:

| Metric | Value |
|--------|--------|
| Accuracy | 97–99% |
| Precision | ~0.98 |
| Recall | ~0.97 |

---

## Streamlit Application

The Streamlit app allows users to:

- Enter tumor measurements
- Predict tumor type instantly
- View classification probability
- Explore dataset visualizations

Example:

```text
Radius = 14
Texture = 20
Perimeter = 90
Area = 600
Smoothness = 0.10
```

Prediction:

```text
Benign Tumor
```

---

## Project Structure

```text
BreastCancerPrediction/
│
├── train_model.py
├── app.py
├── model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

---

## Installation

Clone repository:

```bash
git clone <repository-link>
```

Move into folder:

```bash
cd BreastCancerPrediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Train Model

Run:

```bash
python train_model.py
```

Generated files:

```text
model.pkl
scaler.pkl
```

---

## Run Streamlit App

Start application:

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

---

## Machine Learning Concepts Used

- Classification
- Support Vector Machine (SVM)
- Feature Scaling
- Model Evaluation
- Data Visualization
- Deployment

---

## Future Improvements

Possible enhancements:

- Hyperparameter Tuning
- Feature Selection
- Explainable AI (SHAP/LIME)
- Deep Learning Integration
- Cloud Deployment

---

## Screenshots

Add screenshots of:

- Streamlit Dashboard
- Prediction Output
- Dataset Visualizations

---

## Author

Koushik Patel

---

## License

This project is for educational and learning purposes.
