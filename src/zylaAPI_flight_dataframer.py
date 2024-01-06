import os
import pandas as pd
import numpy as np
import json
import math
from utils import get_day_of_week, get_date_label, num_flight_within
from datetime import datetime


# Read flight dataset in .json format and load it on a dataframe
file_name = 'historical_flight_data.json'
file_path = '/mnt/c/Developer/University/SML/sml-project-2023-manfredi-meneghin/datasets/zylaAPI_flights/'
complete_name = file_path + file_name
df = pd.read_json(complete_name)

# Drop all the flights having more than one flight_number (used by different companies to sell same tickets)
df.drop_duplicates(subset = ['depScheduledTime', 'depApIataCode', 'arrApIataCode'], inplace= True)
df.reset_index(inplace = True)
df.drop(columns={'index'}, inplace = True)

# Create new columns for future df's data
new_column_names  = ['date','time', 'month', 'trip_time', 'day_of_week']
new_column_values = []


# Create data for new columns
for row in range(df.shape[0]):
    # Get year, month, day and time from the selected row
    dep_ts       = df.at[row, 'depScheduledTime']
    dep_datetime = datetime.strptime(dep_ts, "%Y-%m-%dT%H:%M:%S.%f")
    arr_ts = df.at[row, 'arrScheduledTime']
    arr_datetime = datetime.strptime(arr_ts, "%Y-%m-%dT%H:%M:%S.%f")

    delta_time = arr_datetime - dep_datetime
    trip_time  = math.ceil(delta_time.total_seconds()/60)

    dep_yyyy = dep_datetime.year
    dep_mm   = dep_datetime.month
    dep_dd   = dep_datetime.day
    dep_hh   = dep_datetime.hour

    # Get additional information from the departure datetime
    dep_date_label  = get_date_label( dep_yyyy, dep_mm, dep_dd, 'hyphen')
    day_of_the_week = get_day_of_week(dep_yyyy, dep_mm, dep_dd)
    # Save now: date_label, hour, month, trip_time, day_of_the_week

    new_column_values.append([dep_date_label, dep_hh, dep_mm, trip_time, day_of_the_week])


# Add the column "flight_within_60min" and calculate these values for each flight
flight_within, column_name = num_flight_within(60, df)
df[column_name] = flight_within
df[new_column_names] = new_column_values

# Drop useless columns
df.drop(columns={'depScheduledTime', 'arrScheduledTime'}, inplace = True)

print(df.head(5).T)

# Save the new dataframe in a new file (.csv)
ts_path = "/mnt/c/Developer/University/SML/sml-project-2023-manfredi-meneghin/datasets/zylaAPI_flights/"
ts_name = 'historical_flight_data_processed.csv'
ts_complete_path = os.path.join(ts_path, ts_name)

with open(ts_complete_path, "wb") as df_out:
    df.to_csv(df_out, index= False)
df_out.close()

