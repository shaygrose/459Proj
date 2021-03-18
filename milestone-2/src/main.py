import numpy as np
import pandas as pd
import pickle
import sys
import os
import csv
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
import datetime as dt
import os.path
from os import path


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

# print(X)
y = data.outcome
# 0:deceased, 1:hospitalized, 2:nonhospitalized, 3:recovered
y = y.astype('category')
y = y.cat.codes
# print(y)

# 80% train, 20% test
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=69)

''' RANDOM FOREST '''
if not path.exists('models/rf_classifier.pkl'):
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=10)
    rf_model.fit(X_train, y_train)
    pickle.dump(rf_model, open('models/rf_classifier.pkl', 'wb'))

rf_model = pickle.load(open('models/rf_classifier.pkl', 'rb'))
rf_score = rf_model.score(X_valid, y_valid)
print("Random Forest score on validation: ", rf_score)


''' ADA BOOST '''
# ada = AdaBoostClassifier(n_estimators=100, random_state=69)
# ada.fit(X_train, y_train)
# ada_score = ada.score(X_valid, y_valid)

# print("AdaBoost score on validation: ", ada_score)

''' XG BOOST '''
if not path.exists('models/xgb_classifier.pkl'):
    data_dmatrix = xgb.DMatrix(data=X, label=y)
    xg_reg = xgb.XGBRegressor(objective='multi:softmax', colsample_bytree=0.3, learning_rate=0.1,
                              max_depth=10, alpha=10, n_estimators=20, num_class=4, verbosity=0)
    xg_reg.fit(X_train, y_train)
    # write model to pkl file
    pickle.dump(xg_reg, open('models/xgb_classifier.pkl', 'wb'))

xg_reg = pickle.load(open('models/xgb_classifier.pkl', 'rb'))
xg_score = xg_reg.score(X_valid, y_valid)
print("XGBoost score on validation: ", xg_score)

''' KNN '''
# normalized = ((data - data.mean())/data.std())
if not path.exists('models/knn_classifier.pkl'):
    # weights can be distance or uniform
    n_neighbors = 11
    knn = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    knn.fit(X_train, y_train)
    pickle.dump(knn, open('models/knn_classifier.pkl', 'wb'))

knn = pickle.load(open('models/knn_classifier.pkl', 'rb'))
knn_score = knn.score(X_valid, y_valid)
print("KNN score on validation: ", knn_score)
