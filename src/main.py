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

# drop source column since it isn't useful
train.drop(columns=['source', 'additional_information'], inplace=True)
test.drop(columns=['source', 'additional_information'], inplace=True)

# drop last update column since it is the same for every entry
locations.drop(columns=['Last_Update'], inplace=True)
# rename location columns so they match case column names
locations.columns = ['province', 'country', 'latitude', 'longitude',
                     'confirmed', 'deaths', 'recovered', 'active', 'combined_key',
                     'incidence_rate', 'fatality_ratio']




''' TASK 1.2 '''  # cases_train.csv & cases_test.csv

train['country'] = train.apply(lambda row: replace_taiwan(row), axis=1)
test['country'] = test.apply(lambda row: replace_taiwan(row), axis=1)

train['sex'].fillna(value="Not specified", inplace=True)
test['sex'].fillna(value="Not specified", inplace=True)

# convert age ranges to single number
train['age'] = train['age'].apply(
    lambda x: convert_age_range(x) if isinstance(x, str) else x)
test['age'] = test['age'].apply(
    lambda x: convert_age_range(x) if isinstance(x, str) else x)

age_means = round(train.groupby(['outcome'])['age'].mean())
train['age'] = train.apply(impute_age, age_means=age_means, axis=1)

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
train.dropna(how='all', subset=['province', 'country', 'longitude', 'latitude'], inplace=True)


''' TASK 1.3 '''  # cases_train.csv

# testy = rg.search((train.latitude[0], train.longitude[0]), mode=1)
# print(testy[0]['cc'])

# check for impossible values in columns
outliers = []
check = [('age', 0, 120), ('latitude', -90, 90), ('longitude', -180, 180)]
for attr, lower, upper in check:
    if check_valid(train, attr, lower, upper) !=0:
        # print("Outlier detected")
        outliers.append(attr)

assert len(outliers) == 0, "An outlier was detected, perform further investigation!"

train.apply(check_valid_date, axis=1)

# plot = sns.boxplot(x=merged_train['age'])
# plot = plot.get_figure()
# plot.savefig("../plots/test.png")

# looking at distribution of ages
plt.figure(num=1)
x = train.groupby(['age']).count().reset_index()
y = np.arange(train['age'].nunique())
plt.bar(y, x['sex'])
plt.title("Number of records with different ages")
plt.xlabel("Age")
plt.ylabel("Number of records")
plt.savefig("../plots/age_distribution.png")



''' TASK 1.4 '''  # locations.csv

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


# locations.to_csv("../data/yolo.csv", index=False)

''' TASK 1.5 '''  # cases_train.csv, cases_test.csv & locations.csv

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




empty = merged_train[pd.isna(merged_train['fatality_ratio'])]

confirmed_means = round(merged_train.groupby(['country'])['confirmed'].mean())
empty['confirmed'] = empty.apply(impute_confirmed, confirmed_means=confirmed_means, axis=1)


deaths_means = round(merged_train.groupby(['country'])['deaths'].mean())
empty['deaths'] = empty.apply(impute_deaths, deaths_means=deaths_means, axis=1)

active_means = round(merged_train.groupby(['country'])['active'].mean())
empty['active'] = empty.apply(impute_active, active_means=active_means, axis=1)

recovered_means = round(merged_train.groupby(['country'])['recovered'].mean())
empty['active'] = empty.apply(impute_recovered, recovered_means=recovered_means, axis=1)

fatality_means = round(merged_train.groupby(['country'])['fatality_ratio'].mean())
empty['fatality_ratio'] = empty.apply(impute_fatality, fatality_means=fatality_means, axis=1)

incidence_means = round(merged_train.groupby(['country'])['incidence_rate'].mean())
empty['incidence_rate'] = empty.apply(impute_incidence, incidence_means=incidence_means, axis=1)



