# -*- coding: utf-8 -*-
"""
Created on Thu May  1 17:42:56 2025

@author: lfull
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA

df = pd.read_csv("Loan_default.csv")
print(df.columns)
df = df.drop("LoanID", axis=1)
X = df.drop("Default", axis=1)
y = df["Default"]
y = y.astype(int)


# binary categorical features (Yes/No to 1/0)
binary_features = ['HasMortgage', 'HasDependents', 'HasCoSigner']

# numerical features
numerical_features = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio','ITLRatio']

# categorical features for OHE
categorical_features = [
    'Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose']

# map Yes/No to 1/0
yes_no_map = FunctionTransformer(lambda df: df.replace({'Yes': 1, 'No': 0}))

# pipelines for numerical and categorical data
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

binary_pipeline = Pipeline(steps=[
    ('yesno_mapper', yes_no_map)])

categorical_pipeline = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

# using ColumnTransformer to update df features
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_features),
    ('bin', binary_pipeline, binary_features),
    ('cat', categorical_pipeline, categorical_features)])

# Full pipeline example (w/o model yet)
run_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=0.95))])

X_processed = run_pipe.fit_transform(X)


from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# train and test split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y)

y_train = y_train.astype(int).to_numpy()
y_test = y_test.astype(int).to_numpy()

classes = np.unique(y_train)
class_weights_array = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, class_weights_array))

# Build the model
LD_model = Sequential()
LD_model.add(Input(shape=(X_train.shape[1],)))
LD_model.add(Dense(128, activation='relu'))
LD_model.add(Dropout(0.3))
LD_model.add(Dense(64, activation='relu'))
LD_model.add(Dropout(0.2))
LD_model.add(Dense(32, activation='sigmoid'))
LD_model.add(Dropout(0.1))
LD_model.add(Dense(1, activation='sigmoid'))
LD_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy',
                 metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

LD_model.fit(
    X_train, y_train,
    validation_split=0.25,
    epochs=40,
    batch_size=64,
    class_weight=class_weights,
    callbacks=[early_stop])

# model evaluation
loss, accuracy = LD_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay, roc_auc_score

y_pred = (LD_model.predict(X_test) > 0.5).astype("int32")

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred))

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# getting predicted probabilities for class 1
y_probs = LD_model.predict(X_test).flatten()

y_pred = (LD_model.predict(X_test) > 0.5).astype("int32")

# computing and displaying confusion matrix
cm = confusion_matrix(y_test, y_pred)


disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Default", "Default"])
disp.plot(cmap="Blues", values_format='d')
plt.title("Confusion Matrix - Loan Default Prediction")
plt.grid(False)
plt.show()


from sklearn.ensemble import RandomForestClassifier


# now lets train Random Forest while using the same split
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# randomforest predictions
y_rf_pred = rf_model.predict(X_test)

from sklearn.metrics import accuracy_score

# print accuracy
rf_accuracy = accuracy_score(y_test, y_rf_pred)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# classification report
print("Random Forest Classification Report:")
print(classification_report(y_test, y_rf_pred))

# confusion matrix
cm_rf = confusion_matrix(y_test, y_rf_pred)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=["No Default", "Default"])
disp_rf.plot(cmap="Greens", values_format='d')
plt.title("Confusion Matrix - Random Forest")
plt.grid(False)
plt.show()

# predictions for rocauc
y_nn_probs = LD_model.predict(X_test).flatten()
fpr_nn, tpr_nn, _ = roc_curve(y_test, y_nn_probs)
auc_nn = roc_auc_score(y_test, y_nn_probs)


y_rf_probs = rf_model.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_rf_probs)
auc_rf = roc_auc_score(y_test, y_rf_probs)

# overlayed curve plot
plt.figure(figsize=(10, 6))
plt.plot(fpr_nn, tpr_nn, label=f"Neural Network (AUC = {auc_nn:.4f})", linewidth=2)
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest  (AUC = {auc_rf:.4f})", linewidth=2, linestyle='--')
plt.plot([0, 1], [0, 1], linestyle=':', color='gray', label="Random Guess")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve Comparison: Neural Network vs Random Forest")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Neural Network ROC AUC Score:  {auc_nn:.4f}")
print(f"Random Forest ROC AUC Score:   {auc_rf:.4f}")
