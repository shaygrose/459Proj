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
from sklearn.model_selection import train_test_split, GridSearchCV
import datetime as dt
from os import path
from sklearn.metrics import f1_score, recall_score, classification_report, plot_confusion_matrix, multilabel_confusion_matrix, confusion_matrix, make_scorer


data = pd.read_csv("data/cases_train_processed.csv")

# we will only be using the country for classifying
data.drop(columns=["latitude", "longitude", "province"], inplace=True)

# need to convert categorical data to numerical....
# categorical columns are : sex, country, outcome

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

# 80% train, 20% test
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=69)

# ''' RANDOM FOREST '''
# if not path.exists('models/rf_classifier.pkl'):
#     rf_model = RandomForestClassifier(
#         n_estimators=100, max_depth=10, min_samples_leaf=10)
#     rf_model.fit(X_train, y_train)
#     pickle.dump(rf_model, open('models/rf_classifier.pkl', 'wb'))

# ''' XG BOOST '''
# if not path.exists('models/xgb_classifier.pkl'):
#     data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
#     xg_reg = xgb.XGBRegressor(objective='multi:softmax', colsample_bytree=0.3, learning_rate=0.1,
#                               max_depth=10, alpha=10, n_estimators=20, num_class=4, verbosity=0)
#     xg_reg.fit(X_train, y_train)
#     # write model to pkl file
#     pickle.dump(xg_reg, open('models/xgb_classifier.pkl', 'wb'))


# ''' KNN '''
# # normalized = ((data - data.mean())/data.std())
# if not path.exists('models/knn_classifier.pkl'):
#     # weights can be distance or uniform
#     n_neighbors = 11
#     knn = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
#     knn.fit(X_train, y_train)
#     pickle.dump(knn, open('models/knn_classifier.pkl', 'wb'))


# ''' EVALUATION '''


# ''' Accuracy Score '''

# # RF
# rf_model = pickle.load(open('models/rf_classifier.pkl', 'rb'))
# rf_score = rf_model.score(X_valid, y_valid)
# print("Random Forest accuracy score on validation: ", rf_score)
# rf_train_score = rf_model.score(X_train, y_train)
# print("Random Forest accuracy score on train: ", rf_train_score)

# # XG
# xg_reg = pickle.load(open('models/xgb_classifier.pkl', 'rb'))
# xg_score = xg_reg.score(X_valid, y_valid)
# print("XGBoost accuracy score on validation: ", xg_score)
# xg_train_score = xg_reg.score(X_train, y_train)
# print("XGBoost accuracy score on train: ", xg_train_score)

# # KNN
# knn = pickle.load(open('models/knn_classifier.pkl', 'rb'))
# knn_score = knn.score(X_valid, y_valid)
# print("KNN accuracy score on validation: ", knn_score)
# knn_train_score = knn.score(X_train, y_train)
# print("KNN accuracy score on train: ", knn_train_score)


# ''' F1 Score '''
# rf_train_pred = rf_model.predict(X_train)
# knn_train_pred = knn.predict(X_train)
# xg_train_pred = xg_reg.predict(X_train)
# rf_valid_pred = rf_model.predict(X_valid)
# knn_valid_pred = knn.predict(X_valid)
# xg_valid_pred = xg_reg.predict(X_valid)


# target_names = ['0', '1', '2', '3']
# print('--------------------RF Metrics----------------------------------')
# print('- Train -')
# print(classification_report(y_train, rf_train_pred,
#                             target_names=target_names, digits=6))
# print('- Validation -')
# print(classification_report(y_valid, rf_valid_pred,
#                             target_names=target_names, digits=6))

# print('-------------------KNN Metrics----------------------------------')
# print('- Train -')
# print(classification_report(y_train, knn_train_pred,
#                             target_names=target_names, digits=6))
# print('- Validation -')
# print(classification_report(y_valid, knn_valid_pred,
#                             target_names=target_names, digits=6))

# print('-------------------XGBoost Metrics------------------------------')
# print('- Train -')
# print(classification_report(y_train, xg_train_pred,
#                             target_names=target_names, digits=6))
# print('- Validation -')
# print(classification_report(y_valid, xg_valid_pred,
#                             target_names=target_names, digits=6))


# plot_confusion_matrix(knn, X_valid, y_valid,
#                       display_labels=target_names,
#                       cmap=plt.cm.Blues)
# plt.title('KNN Validation Confusion Matrix')
# plt.savefig("plots/knn_valid_confusion.png")

# plot_confusion_matrix(knn, X_train, y_train,
#                       display_labels=target_names,
#                       cmap=plt.cm.Blues)
# plt.title('KNN Train Confusion Matrix')
# plt.savefig("plots/knn_train_confusion.png")

# plot_confusion_matrix(rf_model, X_valid, y_valid,
#                       display_labels=target_names,
#                       cmap=plt.cm.Blues)
# plt.title('RF Validation Confusion Matrix')
# plt.savefig("plots/rf_valid_confusion.png")

# plot_confusion_matrix(rf_model, X_train, y_train,
#                       display_labels=target_names,
#                       cmap=plt.cm.Blues)
# plt.title('RF Train Confusion Matrix')
# plt.savefig("plots/rf_train_confusion.png")

# print("XGBoost Train Confusion Matrix")
# print(confusion_matrix(y_train, xg_train_pred))
# print("XGBoost Validation Confusion Matrix")
# print(confusion_matrix(y_valid, xg_valid_pred))


# def check_if_file_valid(filename):
#     assert filename.endswith('predictions.txt'), 'Incorrect filename'
#     f = open(filename).read()
#     l = f.split('\n')
#     assert len(l) == 46500, 'Incorrect number of items'
#     assert (len(set(l)) == 4), 'Wrong class labels'
#     return 'Thepredictionsfile is valid'


# check_if_file_valid('predictions.txt')


''' TUNING HYPERPARAMS '''

dead_recalls = []
dead_f1s = []
overall_recalls = []
overall_f1s = []


def dead_recall_score(model, X, y):
    y_pred = model.predict(X)
    dead_recall = recall_score(y, y_pred, average=None, zero_division=0)[0]
    dead_f1 = f1_score(y, y_pred, average=None, zero_division=0)[0]
    dead_recalls.append(dead_recall)
    dead_f1s.append(dead_f1)

    overall_recalls.append(recall_score(
        y, y_pred, average='weighted', zero_division=0))
    overall_f1s.append(recall_score(
        y, y_pred, average='weighted', zero_division=0))

    return dead_recall

# model training
# create the parameter grid based on the results of random search


# RF
n_estimators = [5, 10, 20]
max_depth = [5, 10, 15]
min_samples_leaf = [5, 10, 15]

param_grid = [
    {'n_estimators': n_estimators, 'max_depth': max_depth,
        'min_samples_leaf': min_samples_leaf},
]

# base model
rf_model = RandomForestClassifier()

# instantiate the grid search model
# Exhaustive search over specified parameter values for an estimator
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    # not sure what scoring should be set to ??? need to research more
    scoring=dead_recall_score,
    verbose=1,
)

# fit the grid search to the data
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
best_rf.fit(X_train, X_train)
print(dead_f1s)
print(dead_recalls)
print(overall_f1s)
print(overall_recalls)
with open('results/rf_CVresults', 'w') as textfile:
    print(grid_search.best_params_, file=textfile)
pickle.dump(best_rf, open('models/best_rf_classifier.pkl', 'wb'))
