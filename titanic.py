import pandas as pd  
import numpy as np  
import seaborn as sns  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score, classification_report  

# Load dataset  
df = pd.read_csv("titanic.csv")  # Make sure titanic.csv is in the same folder

# Display first few rows  
print(df.head())

# Check for missing values  
print(df.isnull().sum())

# Fill missing values  
df["Age"].fillna(df["Age"].median(), inplace=True)  
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)  

# Drop unnecessary columns  
df.drop(["Cabin", "Name", "Ticket"], axis=1, inplace=True)  

# Encode categorical variables  
label_encoder = LabelEncoder()  
df["Sex"] = label_encoder.fit_transform(df["Sex"])  
df["Embarked"] = label_encoder.fit_transform(df["Embarked"])  

# Define features and target  
X = df.drop("Survived", axis=1)  
y = df["Survived"]  

# Split dataset into training and testing sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Train the model  
model = LogisticRegression()  
model.fit(X_train, y_train)  

# Predict & Evaluate  
y_pred = model.predict(X_test)  
accuracy = accuracy_score(y_test, y_pred)  

# Print results  
print("Model Accuracy:", accuracy)  
print(classification_report(y_test, y_pred))  
