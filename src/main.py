import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot as plt
import re


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
        print(ages)
        return int(match.group(1)) // 12
    else:
        return int(ages)


train['age'] = train['age'].apply(
    lambda x: convert_age_range(x) if isinstance(x, str) else "?")


# train['age'] = train['age'].apply(lambda x : int(x) if isinstance(x, float) else x)

print(train['age'].head(50))
print(train['age'][240093])


''' TASK 1.3 '''  # cases_train.csv
# Which attributes have outliers? How do you deal with them? Apply your strategy of dealing with outliers to your datasets

''' TASK 1.4 '''  # locations.csv
# In the location dataset, transform the information for cases from the US from the country level, used in the location dataset, to the state level, used in the cases dataset. Explain your method of transformation, and why you use a particular type of aggregation on any column.

''' TASK 1.5 '''  # cases_train.csv, cases_test.csv & locations.csv
# The two datasets can be joined using some shared features. You can use either 'province, country' or 'latitude, longitude'. Present your strategy for joining the datasets and motivate your design decisions. Apply your join strategy to create a dataset of cases with additional features inherited from their locations.
