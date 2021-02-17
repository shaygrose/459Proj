import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot as plt
import re
from datetime import datetime

categorical_columns = ['sex', 'province', 'country']

''' READING IN DATA '''
test = pd.read_csv("../data/cases_test.csv")
train = pd.read_csv("../data/cases_train.csv")
locations = pd.read_csv("../data/location.csv")

''' TASK 1.2 '''  # cases_train.csv & cases_test.csv
# Perform data cleaning steps, mainly on the age column. Reduce different formats (ex. 20-29, 25-, 13 months), to a standard format (ex. 25)
# For all attributes with missing values, discuss why and how (if applicable) you impute missing values. Apply your imputation strategy to your datasets.


def trim_strings(x): return x.strip() if isinstance(x, str) else x


def convert_age_range(ages):
    if "-" in ages:
        # if ages.find("-") != -1:
        beg, end = ages.split("-")
        if beg == '':
            return int(end)
        elif end == '':
            return int(beg)
        else:
            avg = int(end) + int(beg) // 2
            return int(avg)
    if "+" in ages:
        beg, end = ages.split("+")
        if beg == '':
            return int(end)
        elif end == '':
            return int(beg)
    if "." in ages:
        beg, end = ages.split(".")
        return int(beg)
    if match := re.search('([0-9]+) \w', ages, re.IGNORECASE):
        # print(ages)
        return int(round(int(match.group(1)) / 12, 0))
    else:
        return int(ages)


def convert_date_range(dates):
    if "-" in dates:
        beg, _ = dates.split("-")
        return beg.strip()
    else:
        return dates

# def fill_gender_data(train):
#     # Sex Attribute
#     attr = categorical_columns[0]

#     # Get the mode for each of the class labels
#     nonhospitalized_mode = train[(train['outcome'] == 'nonhospitalized') & pd.notna(train[attr])][attr].mode()[
#         0]

#     hospitalized_mode = train[(train['outcome'] == 'hospitalized') & pd.notna(train[attr])][attr].mode()[
#         0]

#     recovered_mode = train[(train['outcome'] ==
#                             'recovered') & pd.notna(train[attr])][attr].mode()[0]
#     deceased_mode = train[(train['outcome'] ==
#                            'deceased') & pd.notna(train[attr])][attr].mode()[0]

#     train.loc[((pd.isna(train[attr])) & (train['outcome'] == 'nonhospitalized')),
#               attr] = nonhospitalized_mode
#     train.loc[((pd.isna(train[attr])) & (train['outcome'] == 'hospitalized')),
#               attr] = hospitalized_mode
#     train.loc[((pd.isna(train[attr])) & (train['outcome'] == 'recovered')),
#               attr] = recovered_mode
#     train.loc[((pd.isna(train[attr])) & (train['outcome'] == 'deceased')),
#               attr] = deceased_mode
    # print(train['sex'].head(50))


train['sex'].fillna(value="Not specified", inplace=True)

train['age'] = train['age'].apply(
    lambda x: convert_age_range(x) if isinstance(x, str) else x)

train['date_confirmation'] = train['date_confirmation'].apply(
    lambda x: convert_date_range(x) if isinstance(x, str) else x)

# train['date_confirmation'] = datetime.strptime(
#     train['date_confirmation'], '%d/%m/%Y')

# date = train['date_confirmation'][0]
# print(date)
# date_time_obj = datetime.strptime(date, '%d.%m.%Y')
# print('Date:', date_time_obj.date())


train['date_confirmation'] = pd.to_datetime(
    train['date_confirmation'], format='%d.%m.%Y')

print(train['date_confirmation'].head(20))
print(train['date_confirmation'][0].month)
# print(train['age'][240093])
# print(train['age'][194672])


''' TASK 1.3 '''  # cases_train.csv
# Which attributes have outliers? How do you deal with them? Apply your strategy of dealing with outliers to your datasets

''' TASK 1.4 '''  # locations.csv
# In the location dataset, transform the information for cases from the US from the country level, used in the location dataset, to the state level, used in the cases dataset. Explain your method of transformation, and why you use a particular type of aggregation on any column.

''' TASK 1.5 '''  # cases_train.csv, cases_test.csv & locations.csv
# The two datasets can be joined using some shared features. You can use either 'province, country' or 'latitude, longitude'. Present your strategy for joining the datasets and motivate your design decisions. Apply your join strategy to create a dataset of cases with additional features inherited from their locations.
