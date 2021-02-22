import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot as plt
import re
from datetime import datetime
from helper import *
from scipy import stats
import seaborn as sns
import reverse_geocoder as rg

pd.set_option('mode.chained_assignment', None)

''' READING IN DATA '''
test = pd.read_csv("../data/cases_test.csv")
train = pd.read_csv("../data/cases_train.csv")
locations = pd.read_csv("../data/location.csv")

# drop source and additional information column since it isn't useful
train.drop(columns=['source', 'additional_information'], inplace=True)
test.drop(columns=['source', 'additional_information'], inplace=True)

# drop last update column since it is the same for every entry
locations.drop(columns=['Last_Update'], inplace=True)
# rename location columns so they match case column names
locations.columns = ['province', 'country', 'latitude', 'longitude',
                     'confirmed', 'deaths', 'recovered', 'active', 'combined_key',
                     'incidence_rate', 'fatality_ratio']


''' TASK 1.2 '''  # cases_train.csv & cases_test.csv

print("Doing data cleaning and imputing...")

train['country'] = train.apply(lambda row: replace_country_names(row), axis=1)
test['country'] = test.apply(lambda row: replace_country_names(row), axis=1)

train['sex'].fillna(value="Not specified", inplace=True)
test['sex'].fillna(value="Not specified", inplace=True)

# convert age ranges to single number
train['age'] = train['age'].apply(
    lambda x: convert_age_range(x) if isinstance(x, str) else x)
test['age'] = test['age'].apply(
    lambda x: convert_age_range(x) if isinstance(x, str) else x)

age_means = round(train.groupby(['outcome'])['age'].mean())
train['age'] = train.apply(impute_age, age_means=age_means, axis=1)

test_mean = round(test['age'].mean())
test['age'] = test.apply(impute_age_test, mean=test_mean, axis=1)

# convert date ranges to single date
train['date_confirmation'] = train['date_confirmation'].apply(
    lambda x: convert_date_range(x) if isinstance(x, str) else x)
test['date_confirmation'] = test['date_confirmation'].apply(
    lambda x: convert_date_range(x) if isinstance(x, str) else x)

# convert dates to proper datetime object for easier access of components (month, day etc.)
train['date_confirmation'] = pd.to_datetime(
    train['date_confirmation'], format='%d.%m.%Y')
test['date_confirmation'] = pd.to_datetime(
    test['date_confirmation'], format='%d.%m.%Y')

# drop rows that are missing all data that can identify the location, since they are useless
train.dropna(how='all', subset=[
             'province', 'country', 'longitude', 'latitude'], inplace=True)
test.dropna(how='all', subset=['province', 'country',
                               'longitude', 'latitude'], inplace=True)


''' TASK 1.3 '''  # cases_train.csv

print("Doing outlier detection...")

# Check if all coordinates are in their respective country
train['reverse_country_iso'] = train.apply(get_country_iso, axis=1)
train['reverse_country'] = train.apply(get_country, axis=1)
train['is_match'] = (train['country'] == train['reverse_country']) | (
    train['province'] == train['reverse_country'])

# for all rows that the coordinates and country don't match, replace the coordinates with the ones from locations.csv

train[train['is_match'] == False].to_csv("../data/yolo.csv")
# train[train['is_match'] == False]['latitude'] = locations[locations['country'] == train['country']]['latitude']
# train[train['is_match'] == False]['longitude'] = locations[locations['country'] == train['country']]['longitude']

lat = train[train['is_match'] == False].apply(replace_latitude, locations = locations, axis=1)
lon = train[train['is_match'] == False].apply(replace_longitude, locations = locations, axis=1)

# print(train[train['is_match'] == False])

train[train['is_match'] == False]['latitude'] = lat
train[train['is_match'] == False]['longitude'] = lon

train['reverse_country_iso'] = train.apply(get_country_iso, axis=1)
train['reverse_country'] = train.apply(get_country, axis=1)
train['is_match'] = (train['country'] == train['reverse_country']) | (
    train['province'] == train['reverse_country'])

train[train['is_match'] == False].to_csv("../data/yolo2.csv")


# check for impossible values in columns
outliers = []
check = [('age', 0, 120), ('latitude', -90, 90), ('longitude', -180, 180)]
for attr, lower, upper in check:
    if check_valid(train, attr, lower, upper) != 0:
        # print("Outlier detected")
        outliers.append(attr)

assert len(outliers) == 0, "An outlier was detected, perform further investigation!"

train.apply(check_valid_date, axis=1)


''' TASK 1.4 '''  # locations.csv

print("Converting US data to state level...")


# filter out records with US as the country
us = locations[locations['country'] == "US"]

locations = locations[locations['country'] != "US"]

# aggregate column according to different functions
aggregation_functions = {'latitude': 'mean', 'longitude': 'mean', 'confirmed': 'sum', 'deaths': 'sum',
                         'recovered': 'sum', 'active': 'sum', 'combined_key': 'first', 'incidence_rate': 'mean', 'fatality_ratio': 'mean'}
grouped = us.groupby(['province', 'country']).aggregate(
    aggregation_functions).reset_index()

# make sure combined keys are proper format
grouped['combined_key'] = grouped['province'] + ", " + grouped['country']

# recalculate fatality ratio after aggregation of other columns
grouped['fatality_ratio'] = (grouped['deaths'] / grouped['confirmed']) * 100

locations = locations.append(grouped)

# round long numbers
locations['fatality_ratio'] = round(locations['fatality_ratio'], 2)
locations['incidence_rate'] = round(locations['incidence_rate'], 2)

locations.to_csv("../results/location_transformed.csv", index=False)


''' TASK 1.5 '''  # cases_train.csv, cases_test.csv & locations.csv

print("Combining case data and location data...")

# create combined key column
train['combined_key'] = train.apply(combine_keys, axis=1)

# merge train with locations on combined key (left join)
merged_train = pd.merge(train, locations, on="combined_key", how="left")

# rename columns after join
merged_train.rename(columns={'province_x': 'province', 'country_x': 'country',
                             'longitude_x': 'longitude', 'latitude_x': 'latitude'}, inplace=True)

# drop duplicate columns
merged_train.drop(columns=['province_y', 'country_y',
                           'latitude_y', 'longitude_y'], inplace=True)


# repeat on test
test['combined_key'] = test.apply(combine_keys, axis=1)

merged_test = pd.merge(test, locations, on="combined_key", how="left")

merged_test.drop(columns=['province_y', 'country_y',
                          'latitude_y', 'longitude_y'], inplace=True)

merged_test.rename(columns={'province_x': 'province', 'country_x': 'country',
                            'longitude_x': 'longitude', 'latitude_x': 'latitude'}, inplace=True)


cols = ['confirmed', 'deaths', 'active', 'recovered', 'fatality_ratio', 'incidence_rate']

# impute values for countries which didnt have a match in the merge
for i in cols:

    train_means = round(merged_train.groupby(['country'])[i].mean())
    merged_train[i] = merged_train.apply(
    impute_columns_from_location, mean=train_means, attr=i, axis=1)

    test_means = round(merged_test.groupby(['country'])[i].mean())
    merged_test[i] = merged_test.apply(
    impute_columns_from_location, mean=test_means, attr=i, axis=1)


merged_test.to_csv("../results/cases_test_processed.csv", index=False)
merged_train.to_csv("../results/cases_train_processed.csv", index=False)


