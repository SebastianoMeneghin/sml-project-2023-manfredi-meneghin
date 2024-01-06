import os
import pandas as pd
import numpy as np
import json
import math
from utils import get_day_of_week, get_date_label
from datetime import datetime



# Read flight dataset in .json format and load it on a dataframe
file_name = 'historical_flight_data.json'
file_path = '/mnt/c/Developer/University/SML/sml-project-2023-manfredi-meneghin/datasets/zylaAPI_flights/'
complete_name = file_path + file_name
df = pd.read_json(complete_name)

# Drop all the flights having more than one flight_number (used by different companies to sell same tickets)
df.drop_duplicates(subset = ['depScheduledTime', 'depApIataCode', 'arrApIataCode'], inplace= True)
df.reset_index(inplace = True)


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

    dep_date_label = get_date_label(dep_yyyy, dep_mm, dep_dd, 'hyphen')

    # Save now: date_label, hour, month, trip_time,

# Save now:  day_of_the_week, number_of_flight_1h

    
    




