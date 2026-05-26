import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load dataset
data = load_breast_cancer()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df['target'] = data.target

# Basic info
print(df.head())
print(df.shape)
print(df.info())

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(
    df.corr(),
    cmap='coolwarm'
)
plt.show()

# Features and target
X = df.drop(
    'target',
    axis=1
)

y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)

# SVM
model = SVC(
    kernel='rbf',
    probability=True
)

model.fit(
    X_train,
    y_train
)

# Prediction
y_pred = model.predict(
    X_test
)

# Evaluation
acc = accuracy_score(
    y_test,
    y_pred
)

print(
    "Accuracy:",
    acc
)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print(
    classification_report(
        y_test,
        y_pred
    )
)

# Save
joblib.dump(
    model,
    "model.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("Model saved")