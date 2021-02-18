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

''' READING IN DATA '''
test = pd.read_csv("../data/cases_test.csv")
train = pd.read_csv("../data/cases_train.csv")
locations = pd.read_csv("../data/location.csv")

# drop source column since it isn't useful
train.drop(columns=['source'], inplace=True)
test.drop(columns=['source'], inplace=True)

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




  


''' TASK 1.3 '''  # cases_train.csv

# check for impossible values in columns
outliers = []
check = [('age', 0, 120), ('latitude', -90, 90), ('longitude', -180, 180), ('fatality_ratio', 0, 100)]
for attr, lower, upper in check:
    if check_valid(merged_train, attr, lower, upper) !=0:
        # print("Outlier detected")
        outliers.append(attr)

assert len(outliers) == 0, "An outlier was detected, perform further investigation!"


merged_train.apply(check_valid_date, axis=1)

# plot = sns.boxplot(x=merged_train['age'])
# plot = plot.get_figure()
# plot.savefig("../plots/test.png")

# looking at distribution of ages
plt.figure(num=1)
x = merged_train.groupby(['age']).count().reset_index()
y = np.arange(merged_train['age'].nunique())
plt.bar(y, x['sex'])
plt.title("Number of records with different ages")
plt.xlabel("Age")
plt.ylabel("Number of records")
plt.savefig("../plots/age_distribution.png")

# print(merged_train['fatality_ratio'].max())

plt.figure(num=2,figsize=(12,6))
out = pd.cut(merged_train['fatality_ratio'], bins=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20], include_lowest=True)
ax = out.value_counts(sort=False).plot.bar(rot=0, color="b")
# ax.set_xticklabels([c[1:-1].replace(","," to") for c in out.cat.categories])
plt.title("Distribution of fatality ratios")
plt.xlabel("Fatality ratio")
plt.ylabel("Number of records")
plt.savefig("../plots/fatality_ratio_distribution.png")

# merged.apply(detect_outlier, axis=1)
