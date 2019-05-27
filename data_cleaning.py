# libraries
import pandas as pd
from pyspark.sql.types import *
import re
import numpy as np


# defined functions
def IntegerSafe(value):  # In case there are non-integer type to be converted.
    try:
        return int(value)
    except ValueError:
        return None


def FloatSafe(value):  # In case there are non-integer type to be converted.
    try:
        return float(value)
    except ValueError:
        return None


def regex_filter(val):
    """
    used within the data clean function
    """
    if val:
        mo = re.match(r'[0-9]{2}/[0-9]{2}/[0-9]{4}', str(val))
        if mo:
            return True
        else:
            return False
    else:
        return False


def data_clean(df, name_dictionary, violation_code, features):
    # subsetting and rename
    df_subset = df[features]
    df_subset = df_subset.rename(name_dictionary, axis='columns')

    # plate type
    plate_type_info = pd.DataFrame({'plate_type_class': ['PAS', 'COM', 'OMT', 'OMS']})
    df_subset = pd.merge(df_subset, plate_type_info, how='left', left_on='plate_type', right_on="plate_type_class")
    df_subset["plate_type_class"] = df_subset["plate_type_class"].fillna("OTHERS")

    # vehicle body type
    veh_btype_c = ['Convertible', 'Sedan', 'Sedan', 'Sedan', 'Motorcycle',
                   'Emergency', 'Emergency', 'Emergency', 'Bus', 'Taxi',
                   'Limousine', 'Trailer', 'Trailer', 'Trailer', 'Trailer',
                   'Trailer', 'Truck', 'Truck', 'Truck', 'Truck', 'Truck',
                   'Truck', 'Truck', 'Truck', 'Pick-up', 'Pick-up', 'Suburban']
    veh_btype = ['CONV', 'SEDN', '4DSD', '2DSD', 'MCY', 'FIRE', 'AMBU',
                 'HRSE', 'BUS', 'TAXI', 'LIM', 'POLE', 'H/TR', 'SEMI',
                 'TRLR', 'LTRL', 'LSTV', 'VAN', 'TOW', 'TANK', 'STAK',
                 'FLAT', 'DUMP', 'DELV', 'PICK', 'P-U', 'SUBN']

    veh_btype_info = pd.DataFrame({'veh_body_type': veh_btype,
                                   'veh_body_type_class': veh_btype_c})

    df_subset = pd.merge(df_subset, veh_btype_info,
                         how="left", on="veh_body_type")

    df_subset["veh_body_type_class"] = df_subset["veh_body_type_class"].fillna("Others")

    # vehicle make
    # we create the dataframe that will hold the groups we want
    veh_make = df_subset["veh_make"].value_counts()[:30].reset_index()[["index"]]
    veh_make = veh_make.rename(columns={"index": "veh_make_class"})

    df_subset = pd.merge(df_subset, veh_make, how="left", left_on="veh_make", right_on="veh_make_class")
    df_subset["veh_make_class"] = df_subset["veh_make_class"].fillna("OTHERS")

    # vehicle color
    veh_color_c = ['White', 'White', 'White', 'Gray', 'Gray', 'Gray',
                   'Gray', 'Black', 'Black', 'Black', 'Black',
                   'Red', 'Red', 'Brown', 'Brown', 'Silver',
                   'Blue', 'Green', 'Green', 'Yellow', 'Yellow',
                   'Gold', 'Gold', 'Orange', 'Orange']
    veh_color = ['WH', 'WHITE', 'WHT', 'GY', 'GREY', 'GRAY', 'GRY',
                 'BK', 'BLACK', 'BL', 'BLK', 'RD', 'RED', 'BROWN',
                 'BR', 'SILVE', 'BLUE', 'GR', 'GREEN', 'YW', 'YELLO',
                 'GOLD', 'GL', 'ORANG', 'OR']

    veh_color_info = pd.DataFrame({'veh_color': veh_color, 'veh_color_group': veh_color_c})
    df_subset = pd.merge(df_subset, veh_color_info, how="left", on="veh_color")
    df_subset["veh_color_group"] = df_subset["veh_color_group"].fillna("Others")

    # issue date, new columns: issue_date, issue_year, issue_month, issue_quarter, issue_weekday

    df_subset = df_subset[df_subset['issue_date'].apply(regex_filter)]

    date = pd.to_datetime(df_subset.loc[:, 'issue_date'], format="%m/%d/%Y")

    year = date.dt.year.astype(int)
    month = date.dt.month
    quarter = date.dt.quarter
    dow = date.dt.weekday
    weekday = ((dow <= 5) & (dow >= 1)).astype(int)
    df_subset['issue_date'] = date
    df_subset['issue_month'] = month.astype(int)
    df_subset['issue_year'] = year.astype(int)
    df_subset['issue_quarter'] = quarter.astype(int)
    df_subset['issue_weekday'] = weekday.astype(int)
    df_subset = df_subset.reset_index(drop=True)

    # violation time
    df_subset['violation_time'] = df_subset['violation_time'].map(lambda x: np.nan \
        if (re.match(r'\d{4}(A|P)', str(x).upper()) is None) else str(x))

    hour = df_subset['violation_time'].map(lambda x: x if pd.isnull(x) else int(str(x)[:2]))
    minute = df_subset['violation_time'].map(lambda x: x if pd.isnull(x) else int(str(x)[2:4]))
    section = df_subset['violation_time'].map(lambda x: x if pd.isnull(x) else str(x)[4:])
    hour_24 = []
    for h, s in zip(hour, section):
        if (pd.isnull(s) or h > 12):
            h_new = np.nan
        else:
            if s == "A":
                if h == 12:
                    h_new = 0
                else:
                    h_new = h
            if s == "P":
                if h == 12:
                    h_new = 12
                else:
                    h_new = h + 12
        hour_24.append(h_new)

    df_subset['violation_time'] = pd.Series(hour_24)

    # vehicle year
    df_subset['veh_year'] = df_subset['veh_year']\
        .map(lambda x: np.nan if (re.match(r'\d{4}', str(x)) is None) else int(str(x)[:4]))

    # violation county
    vio_county = {'K': 'Kings', 'Q': 'Queens',
                  'NY': 'Manhattan', 'BX': 'Bronx', 'R': 'Richmond'}
    df_subset["violation_county"].replace(vio_county, inplace=True)

    # registration state
    top_state = ['NY', 'NJ', 'PA', 'CT', 'FL', 'CT']
    df_subset.loc[(~df_subset['registration_state'].isin(top_state)),
                  'registration_state'] = 'OTHERS'

    # final features
    df_subset_final = df_subset_final[['plate_type', 'violation_precinct',
                                       'veh_color_group',
                                       'issue_month', 'issue_year',
                                       'issue_quarter', 'issue_weekday',
                                       'registration_state',
                                       'violation_county',
                                       'violation_infront_oppos',
                                       'violation_time',
                                       'veh_year', 'plate_type_class',
                                       'veh_body_type_class', 'veh_make_class',
                                       'violation_code']]

    return df_subset_final


df_2018_cleaned = data_clean(parking_2018_df,
                             name_dic, viol_code_park, feature_list)
df_2017_cleaned = data_clean(parking_2017_df,
                             name_dic, viol_code_park, feature_list)

frames = [df_2018_cleaned, df_2017_cleaned]
df_2017_2018 = pd.concat(frames) # data cleaning

sc.stop()
