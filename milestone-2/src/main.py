import numpy as np
import pandas as pd
import pickle
import sys
import os
import csv
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
import datetime as dt


# how to open a pickle file in python
#   tfidf = pickle.load(open('svc_tfidf.pickle', 'rb'))


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
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_leaf=10)
rf_model.fit(X_train, y_train)
# rf_score = rf_model.score(X_valid, y_valid)

# print(rf_score)

''' ADA BOOST '''
ada = AdaBoostClassifier(n_estimators=100, random_state=0)
ada.fit(X_train, y_train)
# print(ada.score(X_valid, y_valid))

''' KNN '''

# normalized = ((data - data.mean())/data.std())

# weights can be distance or uniform
n_neighbors = 10
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
clf.fit(X_train, y_train)
print(clf.score(X_valid, y_valid))

# save a model as a pickle file
pickle.dump(rf_model, open('models/rf.pickle', 'wb'))
pickle.dump(ada, open('models/ada.pickle', 'wb'))
pickle.dump(clf, open('models/knn.pickle', 'wb'))
