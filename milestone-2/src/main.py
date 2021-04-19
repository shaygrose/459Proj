import numpy as np
import pandas as pd
import pickle
import sys
import os
import csv
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import xgboost as xgb
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
import datetime as dt
from os import path
from sklearn.metrics import f1_score, classification_report, plot_confusion_matrix, multilabel_confusion_matrix, confusion_matrix


data = pd.read_csv("data/cases_train_processed.csv")

# we will only be using the country for classifying
data.drop(columns=["latitude", "longitude", "province"], inplace=True)

# need to convert categorical data to numerical....
# categorical columns are : sex, country, outcome
# data["body_style"] = data["body_style"].astype('category')
# data["body_style_cat"] = data["body_style"].cat.codes

data['date_confirmation'] = pd.to_datetime(
    data['date_confirmation'], format='%Y-%m-%d')

data['date_confirmation'] = data['date_confirmation'].map(
    dt.datetime.toordinal)

data['sex'] = data['sex'].astype('category')
data['country'] = data['country'].astype('category')

cat_columns = data.select_dtypes(['category']).columns
data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)


X = data[["age", "sex", "country", "date_confirmation",
          "confirmed", "deaths", "recovered", "active", "incidence_rate", "fatality_ratio"]]

y = data.outcome
# 0:deceased, 1:hospitalized, 2:nonhospitalized, 3:recovered
y = y.astype('category')
y = y.cat.codes

print(X.nunique)

# 80% train, 20% test
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=69)

''' RANDOM FOREST '''
if not path.exists('models/rf_classifier.pkl'):
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=10)
    rf_model.fit(X_train, y_train)
    pickle.dump(rf_model, open('models/rf_classifier.pkl', 'wb'))


''' ADA BOOST '''
# ada = AdaBoostClassifier(n_estimators=100, random_state=69)
# ada.fit(X_train, y_train)
# ada_score = ada.score(X_valid, y_valid)

# print("AdaBoost score on validation: ", ada_score)

''' XG BOOST '''
if not path.exists('models/xgb_classifier.pkl'):
    data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
    xg_reg = xgb.XGBRegressor(objective='multi:softmax', colsample_bytree=0.3, learning_rate=0.1,
                              max_depth=10, alpha=10, n_estimators=20, num_class=4, verbosity=0)
    xg_reg.fit(X_train, y_train)
    # write model to pkl file
    pickle.dump(xg_reg, open('models/xgb_classifier.pkl', 'wb'))


''' KNN '''
# normalized = ((data - data.mean())/data.std())
if not path.exists('models/knn_classifier.pkl'):
    # weights can be distance or uniform
    n_neighbors = 11
    knn = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    knn.fit(X_train, y_train)
    pickle.dump(knn, open('models/knn_classifier.pkl', 'wb'))


''' EVALUATION '''


''' Accuracy Score '''

# RF
rf_model = pickle.load(open('models/rf_classifier.pkl', 'rb'))
rf_score = rf_model.score(X_valid, y_valid)
print("Random Forest accuracy score on validation: ", rf_score)
rf_train_score = rf_model.score(X_train, y_train)
print("Random Forest accuracy score on train: ", rf_train_score)

# XG
xg_reg = pickle.load(open('models/xgb_classifier.pkl', 'rb'))
xg_score = xg_reg.score(X_valid, y_valid)
print("XGBoost accuracy score on validation: ", xg_score)
xg_train_score = xg_reg.score(X_train, y_train)
print("XGBoost accuracy score on train: ", xg_train_score)

# KNN
knn = pickle.load(open('models/knn_classifier.pkl', 'rb'))
knn_score = knn.score(X_valid, y_valid)
print("KNN accuracy score on validation: ", knn_score)
knn_train_score = knn.score(X_train, y_train)
print("KNN accuracy score on train: ", knn_train_score)


''' F1 Score '''
rf_train_pred = rf_model.predict(X_train)
knn_train_pred = knn.predict(X_train)
xg_train_pred = xg_reg.predict(X_train)
rf_valid_pred = rf_model.predict(X_valid)
knn_valid_pred = knn.predict(X_valid)
xg_valid_pred = xg_reg.predict(X_valid)


target_names = ['0', '1', '2', '3']
print('--------------------RF Metrics----------------------------------')
print('- Train -')
print(classification_report(y_train, rf_train_pred,
                            target_names=target_names, digits=6))
print('- Validation -')
print(classification_report(y_valid, rf_valid_pred,
                            target_names=target_names, digits=6))

print('-------------------KNN Metrics----------------------------------')
print('- Train -')
print(classification_report(y_train, knn_train_pred,
                            target_names=target_names, digits=6))
print('- Validation -')
print(classification_report(y_valid, knn_valid_pred,
                            target_names=target_names, digits=6))

print('-------------------XGBoost Metrics------------------------------')
print('- Train -')
print(classification_report(y_train, xg_train_pred,
                            target_names=target_names, digits=6))
print('- Validation -')
print(classification_report(y_valid, xg_valid_pred,
                            target_names=target_names, digits=6))


# print('--------------------------------------------------------------')
# print(multilabel_confusion_matrix(y_valid, rf_valid_pred, labels=target_names))
# print(multilabel_confusion_matrix(y_valid, knn_valid_pred, labels=target_names))
# print(multilabel_confusion_matrix(y_valid, xg_valid_pred, labels=target_names))

plot_confusion_matrix(knn, X_valid, y_valid,
                      display_labels=target_names,
                      cmap=plt.cm.Blues)
plt.title('KNN Validation Confusion Matrix')
plt.savefig("plots/knn_valid_confusion.png")

plot_confusion_matrix(knn, X_train, y_train,
                      display_labels=target_names,
                      cmap=plt.cm.Blues)
plt.title('KNN Train Confusion Matrix')
plt.savefig("plots/knn_train_confusion.png")

plot_confusion_matrix(rf_model, X_valid, y_valid,
                      display_labels=target_names,
                      cmap=plt.cm.Blues)
plt.title('RF Validation Confusion Matrix')
plt.savefig("plots/rf_valid_confusion.png")

plot_confusion_matrix(rf_model, X_train, y_train,
                      display_labels=target_names,
                      cmap=plt.cm.Blues)
plt.title('RF Train Confusion Matrix')
plt.savefig("plots/rf_train_confusion.png")

print("XGBoost Train Confusion Matrix")
print(confusion_matrix(y_train, xg_train_pred))
print("XGBoost Validation Confusion Matrix")
print(confusion_matrix(y_valid, xg_valid_pred))

# plot_confusion_matrix(xg_reg, X_valid, y_valid,
#                       display_labels=target_names,
#                       cmap=plt.cm.Blues)
# plt.title('XGBoost Confusion Matrix')
# plt.savefig("plots/xgb_confusion.png")
