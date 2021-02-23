import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot as plt
import re
from datetime import datetime
import reverse_geocoder as rg


iso_to_country = pd.read_csv("../data/iso_to_country.csv", index_col='Code')
iso_to_country_dict = iso_to_country.to_dict('index')

# Changing the Country names to match the ones in cases data (e.g. Russian Federation to Russia)
iso_to_country_dict['RU'] = {'Name': 'Russia'}
iso_to_country_dict['CZ'] = {'Name': 'Czechia'}
iso_to_country_dict['US'] = {'Name': 'US'}
iso_to_country_dict['PS'] = {'Name': 'Israel'}
iso_to_country_dict['SZ'] = {'Name': 'Eswatini'}
iso_to_country_dict['AX'] = {'Name': 'Finland'}
iso_to_country_dict['KR'] = {'Name': 'Korea, South'}
iso_to_country_dict['IR'] = {'Name': 'Iran'}
iso_to_country_dict['BO'] = {'Name': 'Bolivia'}
iso_to_country_dict['VN'] = {'Name': 'Vietnam'}
iso_to_country_dict['VE'] = {'Name': 'Venezuela'}

iso_to_country_dict['TW'] = {'Name': 'Taiwan*'}
iso_to_country_dict['CD'] = {'Name': 'Democratic Republic of the Congo'}
iso_to_country_dict['MD'] = {'Name': 'Moldova'}
iso_to_country_dict['CV'] = {'Name': 'Cabo Verde'}
iso_to_country_dict['TZ'] = {'Name': 'Tanzania'}
iso_to_country_dict['RE'] = {'Name': 'Reunion'}
iso_to_country_dict['MK'] = {'Name': 'North Macedonia'}
iso_to_country_dict['XK'] = {'Name': 'Kosova'}
iso_to_country_dict['NA'] = {'Name': 'Namibia'}
iso_to_country_dict['MO'] = {'Name': 'Macau'}
iso_to_country_dict['CI'] = {'Name': 'Cote d\'Ivoire'}


def replace_country_names(row):
    if row['province'] == "Taiwan":
        return "Taiwan*"
    if row['country'] == "United States":
        return "US"
    if row['country'] == "Czech Republic":
        return "Czechia"
    if row['country'] == "South Korea":
        return 'Korea, South'
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


def impute_age(row, age_means):
    if pd.isna(row.age):
        return int(age_means[row.outcome])
    else:
        return row.age


def impute_age_test(row, mean):
    if pd.isna(row.age):
        return mean
    else:
        return row.age


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


def impute_columns_from_location(row, attr, mean):
    if pd.isna(row[attr]):
        return mean[row['country']]
    else:
        return row[attr]


def get_country_iso(row):
    return rg.search((row.latitude, row.longitude), mode=1)[0]['cc']


def get_country(row):
    return iso_to_country_dict[row.reverse_country_iso]['Name']
