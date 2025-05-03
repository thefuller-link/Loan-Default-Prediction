# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 19:01:25 2025

@author: lfull
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



df = pd.read_csv("Loan_Default.csv")
df = df.drop('LoanID', axis=1)

y = df["Default"]

binary_cols = ['HasMortgage', 'HasDependents', 'HasCoSigner']
df[binary_cols] = df[binary_cols].map(lambda val: 1 if val == 'Yes' else 0)
  

X_num = ["Age", "Income", "LoanAmount", "CreditScore",
           "MonthsEmployed", "NumCreditLines",
         "InterestRate", "LoanTerm", "DTIRatio"] #Needs to be scaled


X_cat = ["Education", "EmploymentType", "MaritalStatus", "LoanPurpose"] #One-hot label encoder?

# Numerical pipeline: impute + scale
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

# Categorical pipeline: impute + one-hot encode
categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(drop='first', sparse_output=False))])

# ColumnTransformer to update features
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, X_num),
    ('cat', categorical_transformer, X_cat),
    ('bin', 'passthrough', binary_cols),])
# Full pipeline
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced',max_iter=1000))])

X = df.drop("Default", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf.fit(X_train, y_train)

print("Model Accuracy:", clf.score(X_test, y_test))


