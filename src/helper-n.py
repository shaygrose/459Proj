import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot as plt
import re
from datetime import datetime


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

def replace_taiwan(row):
    if row['province'] == "Taiwan":
        return "China"
    else:
        return row['country']

train['country'] = train.apply(lambda row: replace_taiwan(row), axis=1)
print(train[train['province'] == "Taiwan"])

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

# for places with no province but have a country
def fill_province(row):
    if pd.isna(row['province']):
        mode = train[train['country'] == row['country']]['province'].mode()
        if len(mode) != 0:
            row['province'] = mode[0]
            return mode[0]
    else:
        return row['province']