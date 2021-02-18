import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot as plt
import re
from datetime import datetime


def replace_taiwan(row):
    if row['province'] == "Taiwan":
        return "Taiwan*"
    else:
        return row['country']


def trim_strings(x): return x.strip() if isinstance(x, str) else x


def convert_age_range(ages):
    if "-" in ages:
        # if ages.find("-") != -1:
        beg, end = ages.split("-")
        if beg == '':
            return int(end)
        elif end == '':
            return int(beg)
        elif isinstance(end, str):
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
def fill_province(row, data):
    if pd.isna(row['province']):
        mode = data[data['country'] == row['country']]['province'].mode()
        if len(mode) != 0:
            row['province'] = mode[0]
            return mode[0]
    else:
        return row['province']

# for combining provinces and country into a single column
def combine_keys(row):
    if (pd.notna(row['province'])):
        return row['province'] + ", " + row['country']
    else:
        return row['country']
    

# check if any attributes have impossible values
def check_valid(data, attribute, lower, upper):
    bad_values = 0
    bad_values = bad_values + len(data[data[attribute] < lower])
    bad_values = bad_values + len(data[data[attribute] > upper])
    return bad_values


def check_valid_date(row):
    if row['date_confirmation'].month > 12 or row['date_confirmation'].month < 1:
        print("Bad month detected")
    if row['date_confirmation'].day > 31 or row['date_confirmation'].month < 1:
        print("Bad day detected")
    if row['date_confirmation'].year > 2021 or row['date_confirmation'].year < 2019:
        print("Bad year detected")
