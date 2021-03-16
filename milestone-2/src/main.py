import numpy as np 
import pandas as pd 
import pickle
import sys
import os
import csv
from sklearn.ensemble import AdaBoostClassifier
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics


# how to open a pickle file in python 
# with open('svc_tfidf.pickle', 'rb') as data:
#         tfidf = pickle.load(data)

# how to save a model as a pickle file
#  with open('nmf_model.pickle', 'wb') as output:
#         pickle.dump(models[best_params], output)


data = pd.read_csv("../data/cases_train_processed.csv")

print(len(data))
print(data.shape[0])




#  age,sex,province,country,latitude,longitude,date_confirmation,confirmed,deaths,recovered,active,incidence_rate,fatality_ratio
X = data[["age","province", "latitude","longitude","date_confirmation","confirmed","deaths", "recovered", "active", "incidence_rate", "fatality_ratio"]]

y = data.outcome

# ada = AdaBoostClassifier(n_estimators=100, random_state=0)
# ada.fit(X, y)
# print(ada.score(X, y))