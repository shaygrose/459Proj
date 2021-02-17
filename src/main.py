import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot as plt
import re
from datetime import datetime
from helper import *
from scipy import stats

categorical_columns = ['sex', 'province', 'country']

''' READING IN DATA '''
test = pd.read_csv("../data/cases_test.csv")
train = pd.read_csv("../data/cases_train.csv")
locations = pd.read_csv("../data/location.csv")

train.drop(columns=['source'], inplace=True)
test.drop(columns=['source'], inplace=True)

locations.drop(columns=['Last_Update'], inplace=True)
locations.columns = ['province', 'country', 'latitude', 'longitude',
                     'confirmed', 'deaths', 'recovered', 'active', 'combined_key',
                     'incidence_rate', 'fatality_ratio']


train['country'] = train.apply(lambda row: replace_taiwan(row), axis=1)
# print(train[train['province'] == "Taiwan"])

# train.reset_index(drop=True, inplace=True)

''' TASK 1.2 '''  # cases_train.csv & cases_test.csv
# Perform data cleaning steps, mainly on the age column. Reduce different formats (ex. 20-29, 25-, 13 months), to a standard format (ex. 25)
# For all attributes with missing values, discuss why and how (if applicable) you impute missing values. Apply your imputation strategy to your datasets.


train['sex'].fillna(value="Not specified", inplace=True)
test['sex'].fillna(value="Not specified", inplace=True)


train['age'] = train['age'].apply(
    lambda x: convert_age_range(x) if isinstance(x, str) else x)

test['age'] = test['age'].apply(
    lambda x: convert_age_range(x) if isinstance(x, str) else x)

train['date_confirmation'] = train['date_confirmation'].apply(
    lambda x: convert_date_range(x) if isinstance(x, str) else x)

test['date_confirmation'] = test['date_confirmation'].apply(
    lambda x: convert_date_range(x) if isinstance(x, str) else x)

train['date_confirmation'] = pd.to_datetime(
    train['date_confirmation'], format='%d.%m.%Y')

test['date_confirmation'] = pd.to_datetime(
    test['date_confirmation'], format='%d.%m.%Y')

# # if missing at least 4 of age, gender, province, country

# train.dropna(axis=1, how='all', subset=['age', 'gender', 'province', 'country'], inplace = True)

# train['province'] = train.apply(fill_province, data=train, axis=1)


# train.head(40).to_csv("../data/yolo.csv", index=False)


''' TASK 1.3 '''  # cases_train.csv
# Which attributes have outliers? How do you deal with them? Apply your strategy of dealing with outliers to your datasets

# print(train['age'].max())
# print(train['age'].min())

# test = []
# train.date_confirmation.apply(lambda x : test.append(x) if x.day > 31 else x)
# print(len(test))

# print(train[train['date_confirmation'].year < 2019 ])


''' TASK 1.4 '''  # locations.csv
# In the location dataset, transform the information for cases from the US from the country level, used in the location dataset, to the state level, used in the cases dataset. Explain your method of transformation, and why you use a particular type of aggregation on any column.

us = locations[locations['country'] == "US"]

locations = locations[locations['country'] != "US"]

aggregation_functions = {'latitude': 'mean', 'longitude': 'mean', 'confirmed': 'mean', 'deaths': 'mean',
                         'recovered': 'mean', 'active': 'mean', 'combined_key': 'first', 'incidence_rate': 'mean', 'fatality_ratio': 'mean'}
grouped = us.groupby(['province', 'country']).aggregate(
    aggregation_functions).reset_index()

grouped['combined_key'] = grouped['province'] + ", " + grouped['country']

locations = locations.append(grouped)


''' TASK 1.5 '''  # cases_train.csv, cases_test.csv & locations.csv
# The two datasets can be joined using some shared features. You can use either 'province, country' or 'latitude, longitude'. Present your strategy for joining the datasets and motivate your design decisions. Apply your join strategy to create a dataset of cases with additional features inherited from their locations.

train['combined_key'] = train.apply(combine_keys, axis=1)

merged = pd.merge(train, locations, on="combined_key", how="left")

merged.drop(columns=['province_y', 'country_y',
                     'latitude_y', 'longitude_y'], inplace=True)

merged.rename(columns={'province_x': 'province', 'country_x': 'country',
                       'longitude_x': 'longitude', 'latitude_x': 'latitude'}, inplace=True)

merged.to_csv('../data/merged.csv', index=False)


# def detect_outlier(row):
# z = np.abs(stats.zscore(merged[merged['country'] == row['country']]['fatality_ratio']))
# if z > 3:
#     print("outlier!")
#     print(row)
countries = merged.country.unique()
fatality_zscores = {}
for country in countries:
    temp = merged[merged['country'] == country]
    zscore = pd.DataFrame(
        np.abs(stats.zscore(temp['fatality_ratio']))).set_index(temp.index)
    fatality_zscores[country] = zscore.copy(deep=True)

print(fatality_zscores['Australia'])


# merged.apply(detect_outlier, axis=1)
