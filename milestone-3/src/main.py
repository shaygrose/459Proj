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




''' TRAINING DATA '''

data = pd.read_csv("data/cases_train_processed.csv")

# we will only be using the country for classifying
data.drop(columns=["latitude", "longitude", "province"], inplace=True)

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

''' TEST DATA '''
test_data = pd.read_csv("data/cases_test_processed.csv")

# we will only be using the country for classifying
test_data.drop(columns=["latitude", "longitude", "province"], inplace=True)

test_data['date_confirmation'] = pd.to_datetime(
    test_data['date_confirmation'], format='%Y-%m-%d')

test_data['date_confirmation'] = test_data['date_confirmation'].map(
    dt.datetime.toordinal)

test_data['sex'] = test_data['sex'].astype('category')
test_data['country'] = test_data['country'].astype('category')

cat_columns = test_data.select_dtypes(['category']).columns
test_data[cat_columns] = test_data[cat_columns].apply(lambda x: x.cat.codes)


test = test_data[["age", "sex", "country", "date_confirmation",
          "confirmed", "deaths", "recovered", "active", "incidence_rate", "fatality_ratio"]]



''' TUNING HYPERPARAMS '''

def dead_recall_score(model, X, y):
    y_pred = model.predict(X)

    return recall_score(y, y_pred, average=None, zero_division=0)[0]


def dead_f1_score(model, X, y):
    y_pred = model.predict(X)

    return f1_score(y, y_pred, average=None, zero_division=0)[0]


def total_recall_score(model, X, y):
    y_pred = model.predict(X)

    return recall_score(y, y_pred, average='weighted', zero_division=0)


def total_f1_score(model, X, y):
    y_pred = model.predict(X)

    return f1_score(y, y_pred, average='weighted', zero_division=0)

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

scoring = {"deceased recall": dead_recall_score,
           "deceased f1": dead_f1_score, "total recall": total_recall_score, "total f1": total_f1_score}
# instantiate the grid search model
# Exhaustive search over specified parameter values for an estimator
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
grid_search_rf = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    refit="deceased recall",
    scoring=scoring,
    verbose=1,
)

# fit the grid search to the data
grid_search_rf.fit(X_train, y_train)
res = pd.DataFrame(grid_search_rf.cv_results_)[['mean_test_deceased recall', 'mean_test_deceased f1', 'mean_test_total recall', 'mean_test_total f1',
                                                'param_min_samples_leaf',
                                                'param_n_estimators', 'param_max_depth']]

print(res)
res.to_csv("results/rf_hyperparams.csv", index=False)
best_rf = grid_search_rf.best_estimator_
best_rf.fit(X_train, y_train)
print(grid_search_rf.best_params_)
target_names = ['0', '1', '2', '3']
best_rf_pred = best_rf.predict(X_valid)
with open('results/rf_classification_report.txt', 'w') as textfile:
    print(classification_report(y_valid, best_rf_pred,
                            target_names=target_names, digits=6), file=textfile)
# pickle.dump(best_rf, open('models/best_rf_classifier.pkl', 'wb'))

param_grid_rf = [
    {'n_estimators': n_estimators, 'max_depth': max_depth,
        'min_samples_leaf': min_samples_leaf}]


# KNN
k_neighbours = [5, 10, 15]

param_grid = [
    {'n_neighbors': k_neighbours, 'weights': ['distance', 'uniform']}
]

# base model
knn_model = neighbors.KNeighborsClassifier()

scoring = {"deceased recall": dead_recall_score,
           "deceased f1": dead_f1_score, "total recall": total_recall_score, "total f1": total_f1_score}

grid_search_knn = GridSearchCV(
    estimator=knn_model,
    param_grid=param_grid,
    refit="deceased recall",
    scoring=scoring,
    verbose=1,
)


grid_search_knn.fit(X_train, y_train)
res = pd.DataFrame(grid_search_knn.cv_results_)[['mean_test_deceased recall', 'mean_test_deceased f1', 'mean_test_total recall', 'mean_test_total f1',
                                                 'param_n_neighbors', 'param_weights']]

print(res)
res.to_csv("results/knn_hyperparams.csv", index=False)
best_knn = grid_search_knn.best_estimator_
best_knn.fit(X_train, y_train)
print(grid_search_knn.best_params_)
best_knn_pred = best_knn.predict(X_valid)
with open('results/knn_classification_report.txt', 'w') as textfile:
    print(classification_report(y_valid, best_knn_pred,
                            target_names=target_names, digits=6), file=textfile)
# pickle.dump(best_knn, open('models/best_knn_classifier.pkl', 'wb'))


# XGBoost
data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
xg_model = xgb.XGBRegressor(objective='multi:softmax', colsample_bytree=0.3,
                            alpha=10, num_class=4, verbosity=0)

n_estimators = [20, 30, 50]
max_depth = [10, 12, 15]
learning_rate = [0.3, 0.5, 0.7]

param_grid_xgb = [
    {'n_estimators': n_estimators, 'max_depth': max_depth,
        'learning_rate': learning_rate},
]

grid_search_xgb = GridSearchCV(
    estimator=xg_model,
    param_grid=param_grid_xgb,
    refit="deceased recall",
    scoring=scoring,
    verbose=1,
)

# fit the grid search to the data
grid_search_xgb.fit(X_train, y_train)
xgb_res = pd.DataFrame(grid_search_xgb.cv_results_)[['mean_test_deceased recall', 'mean_test_deceased f1', 'mean_test_total recall', 'mean_test_total f1',
                                                     'param_learning_rate',
                                                     'param_n_estimators', 'param_max_depth']]

print(xgb_res)
xgb_res.to_csv("results/xgb_hyperparams.csv", index=False)
best_xgb = grid_search_xgb.best_estimator_
best_xgb.fit(X_train, y_train)
print(grid_search_xgb.best_params_)

best_xgb_pred = best_xgb.predict(X_valid)
with open('results/xgb_classification_report.txt', 'w') as textfile:
    print(classification_report(y_valid, best_xgb_pred,
                            target_names=target_names, digits=6), file=textfile)
# pickle.dump(best_xgb, open('models/best_rxgb_classifier.pkl', 'wb'))


''' PREDICTING ON TEST '''

# xgb_test_pred = best_xgb.predict(test)
rf_test_pred = best_rf.predict(test)


with open('results/pre-predictions.txt', 'w') as textfile:
    textfile.write("\n".join(map(str, rf_test_pred)))


pred = pd.read_csv('results/pre-predictions.txt', header=None, delimiter = "\n")


pred[pred == 0] = "deceased"
pred[pred == 1] = "hospitalized"
pred[pred == 2] = "nonhospitalized"
pred[pred == 3] = "recovered"

print(pred[0].value_counts())


pred.to_csv("results/pre-predictions.txt", header=None, index=None)

with open('results/pre-predictions.txt', 'r') as f:
    data = f.read()
    with open("results/predictions.txt", 'w') as w:
        w.write(data[:-1])


def check_if_file_valid(filename):
    assert filename.endswith('predictions.txt'), 'Incorrect filename'
    f = open(filename).read()
    l = f.split('\n')
    assert len(l) == 46500, 'Incorrect number of items'
    assert (len(set(l)) == 4), 'Wrong class labels'
    return 'Thepredictionsfile is valid'

check_if_file_valid('results/predictions.txt')