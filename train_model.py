# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import joblib

# Load your dataset
df = pd.read_csv("Loan Prediction system.csv")

# Preprocess
df = df.dropna()
df.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)
df.replace(to_replace='3+', value=4, inplace=True)
df.replace({
    "Married": {"No": 0, "Yes": 1},
    "Gender": {"Male": 1, "Female": 0},
    "Self_Employed": {"No": 0, "Yes": 1},
    "Education": {"Graduate": 1, "Not Graduate": 0},
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2}
}, inplace=True)

X = df.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
Y = df['Loan_Status']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)

# Train the model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Save the model
joblib.dump(classifier, 'loan_model.pkl')
print("Model saved as loan_model.pkl")
