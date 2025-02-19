import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import numpy as np

# Load cleaned dataset
df = pd.read_csv("cleaned_large_noisy_text_data.csv")

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Cleaned_Text"].astype(str))
y = df["Label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probability scores for ROC curve

# Streamlit Dashboard
st.title("Text Classification Dashboard")

# Model Evaluation Metrics
st.subheader("Model Performance")
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
st.subheader("Confusion Matrix")
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Clean", "Noisy"], yticklabels=["Clean", "Noisy"], ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
st.pyplot(fig)

# Anomaly Detection Distribution
st.subheader("Anomaly Detection Distribution")
anomaly_scores = np.abs(y_test - y_pred)
fig, ax = plt.subplots()
sns.histplot(anomaly_scores, bins=3, kde=True, color='red', ax=ax)
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
st.pyplot(fig)

# ROC Curve Visualization
st.subheader("ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
st.pyplot(fig)

# Pie Chart Visualization of Class Distribution
st.subheader("Class Distribution")
class_counts = df["Label"].value_counts()
fig, ax = plt.subplots()
ax.pie(class_counts, labels=["Clean", "Noisy"], autopct='%1.1f%%', colors=['lightblue', 'salmon'], startangle=140)
plt.title("Class Distribution of Labels")
st.pyplot(fig)
