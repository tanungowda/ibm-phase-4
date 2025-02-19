import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Clean", "Noisy"], yticklabels=["Clean", "Noisy"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Anomaly Detection Distribution
anomaly_scores = np.abs(y_test - y_pred)  # Difference between actual and predicted labels
plt.figure(figsize=(8, 5))
sns.histplot(anomaly_scores, bins=3, kde=True, color='red')
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.title("Anomaly Detection Distribution")
plt.show()

# ROC Curve Visualization
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()

# Pie Chart Visualization of Class Distribution
class_counts = df["Label"].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(class_counts, labels=["Clean", "Noisy"], autopct='%1.1f%%', colors=['lightblue', 'salmon'], startangle=140)
plt.title("Class Distribution of Labels")
plt.show()
