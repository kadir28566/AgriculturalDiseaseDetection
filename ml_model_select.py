# -*- coding: utf-8 -*-
"""ML model select.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14KLTIl0M85avDvwsqmvEv10jNhghzPl6
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

X = pd.read_csv("/content/drive/MyDrive/X_features.txt", sep=",", header=None)
y = pd.read_csv("/content/drive/MyDrive/y_labels.txt",  header=None)

print(X.shape)
print(y.shape)

#RandomForest

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)
y_train = y_train.values.reshape(-1)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Sonuçları değerlendirme
print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.svm import SVC

#SVM

svm_model = SVC(kernel='rbf', C=1)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

# Sonuçları değerlendirme
print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

#XGBoost

le = LabelEncoder()

y_encoded = le.fit_transform(y)

print(le.classes_)

print(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Doğruluk Skoru: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

from sklearn.neighbors import KNeighborsClassifier

#KNN

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)

print("Doğruluk Skoru:", accuracy_score(y_test, y_pred_knn))
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_knn))

from sklearn.model_selection import GridSearchCV

# K değerleri için aralık belirle
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}

# GridSearchCV ile KNN'yi optimize et
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# En iyi parametrelerle tahmin yap
best_knn = grid_search.best_estimator_
y_pred_best_knn = best_knn.predict(X_test)

# Sonuçları değerlendirme
print("En İyi KNN Modeli:", best_knn)
print("Doğruluk Skoru:", accuracy_score(y_test, y_pred_best_knn))
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_best_knn))

