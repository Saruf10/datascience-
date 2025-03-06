import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Step 1: Load dataset
file_path = "creditcard.csv"  # Replace with actual file path
df = pd.read_csv(file_path)

# Step 2: Data Exploration
print("Dataset Overview:")
print(df.head())
print("\nClass Distribution:")
print(df['Class'].value_counts())  # Check fraud vs genuine transactions

# Step 3: Data Preprocessing
# Standardizing 'Amount' column (highly skewed)
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Dropping 'Time' column (optional)
df.drop(['Time'], axis=1, inplace=True)

# Step 4: Handle Imbalanced Data using SMOTE
X = df.drop('Class', axis=1)  # Features
y = df['Class']  # Target variable

# Splitting data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to oversample the minority class (fraud cases)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nClass Distribution After SMOTE Oversampling:")
print(pd.Series(y_train_smote).value_counts())  # Check new class distribution

# Step 5: Train Model (Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_smote, y_train_smote)

# Step 6: Model Evaluation
y_pred = rf_model.predict(X_test)

# Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Genuine", "Fraud"], yticklabels=["Genuine", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
