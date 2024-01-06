import json
import os
import pandas as pd
import math
import numpy as np
from utils import get_wind_dir_label

######################################################################################################
# Here a smhiAPI historical mesan file is read and its values are selected, changed, dropped or
# transformed, according to various needs (uniforming them with daily forecasting data, removing
# noise, simplifying the data visualisation). Then the processed data are saved in a new file.
######################################################################################################

file_name = 'checkpoint_00.csv'
file_path = '/mnt/c/Developer/University/SML/sml-project-2023-manfredi-meneghin/datasets/smhi_historical_data/'
complete_name = file_path + file_name

df = pd.read_csv(complete_name)


# Make humidity a categorical value
for row in range(df.shape[0]):
    humidity = df.at[row,'humidity']

    if humidity < 0.375:
        df.at[row,'humidity'] = 1

    elif humidity < 0.5:
        df.at[row,'humidity'] = 2

    elif humidity < 0.625:
        df.at[row,'humidity'] = 3

    elif humidity < 0.75:
        df.at[row,'humidity'] = 4

    elif humidity < 0.875:
        df.at[row,'humidity'] = 5

    elif humidity:
        df.at[row,'humidity'] = 6


# Make cloud cover categorical values (octas)
cloud_labels = ['total_cloud', 'medium_cloud', 'high_cloud', 'low_cloud']
for row in range(df.shape[0]):
    for label in cloud_labels:
        cloud_level = df.at[row, label]

        if cloud_level < 1/16:
            df.at[row, label] = 0

        elif cloud_level < 3/16:
            df.at[row, label] = 1

        elif cloud_level < 5/16:
            df.at[row, label] = 2

        elif cloud_level < 7/16:
            df.at[row, label] = 3

        elif cloud_level < 9/16:
            df.at[row, label] = 4

        elif cloud_level < 11/16:
            df.at[row, label] = 5

        elif cloud_level < 13/16:
            df.at[row, label] = 6

        elif cloud_level < 15/16:
            df.at[row, label] = 7

        else:
            df.at[row, label] = 8


# Bring visibility to km instead of m
for row in range(df.shape[0]):
    df.at[row, 'visibility'] = df.at[row, 'visibility'] / 1000


# Bring pressure to hPa instead of Pa
for row in range(df.shape[0]):
    df.at[row, 'pressure'] = df.at[row, 'pressure'] / 100

# Make pressure a categorical variable (binning)
min_pressure = 970
max_pressure = 1060
num_interval = 8
division_gap = (max_pressure - min_pressure)/num_interval

for row in range(df.shape[0]):
    pressure = df.at[row,'pressure']

    for multiplier in range(1, num_interval):
        if pressure < min_pressure + multiplier*division_gap:
            df.at[row,'pressure'] = multiplier
            break;

# Bring temperature to Celsius instead of Kelvin
for row in range(df.shape[0]):
    df.at[row, 'temperature'] = df.at[row, 'temperature'] - 273.15


# Get windspeed and wind direction from u and v components, then rename columns
for row in range(df.shape[0]):
    u_wind = df.at[row, 'u_wind']
    v_wind = df.at[row, 'v_wind']

    wind_speed = math.sqrt(pow(u_wind, 2) + pow(v_wind, 2))
    wind_dir   = np.arctan2(v_wind, u_wind) * (180 / np.pi)

    wind_dir_label = get_wind_dir_label(wind_dir)

    df.at[row, 'u_wind'] = wind_speed
    df.at[row, 'v_wind'] = wind_dir_label

df.rename(columns={'u_wind': 'wind_speed', 'v_wind': 'wind_dir'}, inplace= True)
df.drop(columns={'prep_1h', 'snow_1h', 'gradient_snow', 'type_prep'}, inplace=True)


# Save clean dataframe in a new file (.csv)
checkpoint_path = "/mnt/c/Developer/University/SML/sml-project-2023-manfredi-meneghin/datasets/smhi_historical_data/"
checkpoint_name = 'historical_data_from_checkpoint.csv'
checkpoint_complete_path = os.path.join(checkpoint_path, checkpoint_name)

with open(checkpoint_complete_path, "wb") as df_out:
    df.to_csv(df_out, index= False)
df_out.close()

