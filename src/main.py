import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot as plt
import re
from datetime import datetime
from helper-n import replace_taiwan, convert_age_range, convert_date_range, trim_strings, fill_province-province

categorical_columns = ['sex', 'province', 'country']

''' READING IN DATA '''
test = pd.read_csv("../data/cases_test.csv")
train = pd.read_csv("../data/cases_train.csv")
locations = pd.read_csv("../data/location.csv")

train.drop(columns=['source'], inplace=True)
test.drop(columns=['source'], inplace=True)

locations.drop(columns=['Last_Update'], inplace = True)
locations.columns = ['province', 'country', 'latitude', 'longitude', 
    'confirmed', 'deaths', 'recovered', 'active', 'combined_key', 
    'incidence_rate', 'fatality_ratio']

# train.reset_index(drop=True, inplace=True)

''' TASK 1.2 '''  # cases_train.csv & cases_test.csv
# Perform data cleaning steps, mainly on the age column. Reduce different formats (ex. 20-29, 25-, 13 months), to a standard format (ex. 25)
# For all attributes with missing values, discuss why and how (if applicable) you impute missing values. Apply your imputation strategy to your datasets.



train['sex'].fillna(value="Not specified", inplace=True)

train['age'] = train['age'].apply(
    lambda x: convert_age_range(x) if isinstance(x, str) else x)

train['date_confirmation'] = train['date_confirmation'].apply(
    lambda x: convert_date_range(x) if isinstance(x, str) else x)

train['date_confirmation'] = pd.to_datetime(
    train['date_confirmation'], format='%d.%m.%Y')


# # if missing at least 4 of age, gender, province, country

# train.dropna(axis=1, how='all', subset=['age', 'gender', 'province', 'country'], inplace = True)

# print(train[pd.isna(train['province']) & pd.isna(train['country'])])




# train['province'] = train.apply(lambda x: fill_province(x), axis=1)

provinces = []
for i, row in train.iterrows():
    val = fill_province(row)
    provinces.append(val)
    # train.iloc[i]['province'] = fill_province(row)

train['province'] = provinces

# print(train.head(35))

train.to_csv("../data/yolo.csv", index=False)
# print(train[['province','country']][142])

# merged = pd.merge(train, locations, on="country", how="outer")

# print(train.head(20))

# merged.head(20).to_csv('../data/merged.csv', index=False)

''' TASK 1.3 '''  # cases_train.csv
# Which attributes have outliers? How do you deal with them? Apply your strategy of dealing with outliers to your datasets

''' TASK 1.4 '''  # locations.csv
# In the location dataset, transform the information for cases from the US from the country level, used in the location dataset, to the state level, used in the cases dataset. Explain your method of transformation, and why you use a particular type of aggregation on any column.

''' TASK 1.5 '''  # cases_train.csv, cases_test.csv & locations.csv
# The two datasets can be joined using some shared features. You can use either 'province, country' or 'latitude, longitude'. Present your strategy for joining the datasets and motivate your design decisions. Apply your join strategy to create a dataset of cases with additional features inherited from their locations.
