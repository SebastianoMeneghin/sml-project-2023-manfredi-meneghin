import requests
import pandas as pd
import numpy as np
import json
import os
from utils import get_date_label, get_padded_hour, one_hour_forward, one_hour_backward, get_wind_dir_label, one_day_forward
from datetime import datetime



def get_current_date_time_and_dst():
    '''
    Return today's year, month and day numbers
    '''
    # Get today's date through TimeAPI
    time_url = "https://worldtimeapi.org/api/timezone/Europe/Stockholm"
    time_response     = requests.get(time_url)
    time_responseJson = time_response.json()

    # Extract datetime
    datetime_str    = time_responseJson["datetime"]
    datetime_object = datetime.fromisoformat(datetime_str[:-6])  # Remove the timezone offset for parsing
    dst             = time_responseJson["dst"]

    # Extract components from datetime
    day   = datetime_object.day
    month = datetime_object.month
    year  = datetime_object.year
    hour  = datetime_object.hour

    return year, month, day, hour, dst


def smhiAPI_acquire_realtime_forecast(mode):
    '''
    Acquire the real-time weather forecast of one full day from smhiAPI Forecast
    Depending on the 'mode', that can be 'today' or 'tomorrow', it gets different data
    '''
    time_range_hour = 30
    year, month, day, hour, dst = get_current_date_time_and_dst()

    # If the mode is 'tomorrow', get the date of the following date starting from the midnight
    if (mode == 'tomorrow'):
        hour = 0
        year, month, day = one_day_forward(year, month, day)

    # Save the selected date to further save the created file
    selected_date = get_date_label(year, month, day, 'hyphen')


    # Set the skeleton of the dataframe
    forecast_df_columns = ["t", "vis", "msl", "r", "gust", "ws", "wd", "tcc_mean", "lcc_mean", "mcc_mean", "hcc_mean", "pcat"]
    forecast_df         = pd.DataFrame(columns=forecast_df_columns)

    # Get the current ARN forecast (it is in UTC time, with always DST off)
    forecast_url = "https://opendata-download-metfcst.smhi.se/api/category/pmp3g/version/2/geotype/point/lon/17.8752/lat/59.5753/data.json"
    response = requests.get(forecast_url)
    responseJson = response.json()

    # The desired_valid_time will change during the day
    desired_valid_times = []
    yyyy, mm, dd, hh = year, month, day, hour

    # If DST on, move back of one hour to find the wanted times
    if (dst):
        yyyy, mm, dd, hh = one_hour_backward(yyyy, mm, dd, hh)
        
    for i in range(time_range_hour):
        datestamp = get_date_label(yyyy, mm, dd, 'hyphen')
        hourstamp = get_padded_hour(hh)

        valid_time = str(datestamp) + 'T' + str(hourstamp) + ':00:00Z'
        desired_valid_times.append(valid_time)

        yyyy, mm, dd, hh = one_hour_forward(yyyy, mm, dd, hh)


    for desired_valid_time in desired_valid_times:
        # Find the time series entry corresponding to the desired 'validTime' or set desired_time_series to "None"
        desired_time_series = next(
            (ts for ts in responseJson["timeSeries"] if ts["validTime"] == desired_valid_time),
            None
        )

        # Check if the desired 'validTime' exists in the data
        if desired_time_series:
            # Define a list of parameter names to retrieve
            parameters_to_retrieve = ["t", "vis", "msl", "r", "gust", "ws", "wd", "tcc_mean", "lcc_mean", "mcc_mean", "hcc_mean", "pcat"]

            # Extract values for the specified parameters, add them to the future df row
            new_row_attr = []
            for param_name in parameters_to_retrieve:
                param_value = next(
                    (param["values"][0] for param in desired_time_series["parameters"] if param["name"] == param_name),
                    None
                )
                new_row_attr.append(param_value)

            forecast_df.loc[len(forecast_df.index)] = new_row_attr
        else:
            print(f"No data found for validTime {desired_valid_time}")


    # Create the date and time dataframe columns
    datetime_df = pd.DataFrame(columns=['date', 'time'])

    # The file read belong the all the hours following the current hour, so set the time one hour forward
    yyyy, mm, dd, hh = one_hour_forward(year, month, day, hour)

    # Add as many row as the lenght of the forecast_df
    for i in range(len(forecast_df.index)):
        date_value = get_date_label(yyyy, mm, dd, 'hyphen')
        time_value = hh

        # Add the two values at the end of the datetime_df
        datetime_df.loc[len(datetime_df.index)] = [date_value, time_value]

        # Move the clock one hour ahead
        yyyy, mm, dd, hh = one_hour_forward(yyyy, mm, dd, hh)


    # Create the dataframe where to merge the other two
    total_df_columns = ['date', 'time', 'temperature', 'visibility', 'pressure', 'humidity', 'gusts_wind', 
            'wind_speed', 'wind_dir', 'total_cloud', 'low_cloud', 'medium_cloud', 'high_cloud', 'sort_prep']
    total_df = pd.DataFrame(columns=total_df_columns)

    # Merge both the dataframe into total_df
    total_df[['date', 'time']] = datetime_df[['date', 'time']]
    total_df[['temperature', 'visibility', 'pressure', 'humidity', 
            'gusts_wind', 'wind_speed', 'wind_dir', 'total_cloud', 
            'low_cloud', 'medium_cloud', 'high_cloud', 'sort_prep']] = forecast_df[["t", "vis", "msl", "r", 
                                                                                    "gust", "ws", "wd", "tcc_mean", 
                                                                                    "lcc_mean", "mcc_mean", "hcc_mean", "pcat"]]


    # Make pressure a categorical variable (binning)
    min_pressure = 970
    max_pressure = 1060
    num_interval = 8
    division_gap = (max_pressure - min_pressure)/num_interval

    for row in range(total_df.shape[0]):
        pressure = total_df.at[row,'pressure']

        for multiplier in range(1, num_interval):
            if pressure < min_pressure + multiplier*division_gap:
                total_df.at[row,'pressure'] = multiplier
                break;

    # Make the wind direction a categorical variable (labeling)
    wind_dir_column = []
    for row in range(total_df.shape[0]):
        # Get wind speed and shift it in the interval [-180°, +180°]
        wd = total_df.at[row, 'wind_dir']
        wd = wd - 180

        wind_dir_label = get_wind_dir_label(wd)
        wind_dir_column.append(wind_dir_label)

        #total_df.at[row, 'wind_dir'] = wind_dir_label

    total_df.drop(columns={'wind_dir'}, inplace = True)
    total_df['wind_dir'] = wind_dir_column


    # Make humidity a categorical value
    for row in range(total_df.shape[0]):
        humidity = total_df.at[row,'humidity']

        if humidity < 0.375:
            total_df.at[row,'humidity'] = 1

        elif humidity < 0.5:
            total_df.at[row,'humidity'] = 2

        elif humidity < 0.625:
            total_df.at[row,'humidity'] = 3

        elif humidity < 0.75:
            total_df.at[row,'humidity'] = 4

        elif humidity < 0.875:
            total_df.at[row,'humidity'] = 5

        elif humidity:
            total_df.at[row,'humidity'] = 6

    
    # Save the forecast dataframe in a new file (.csv)
    ts_path = "/mnt/c/Developer/University/SML/sml-project-2023-manfredi-meneghin/datasets/smhi_daily_data/"
    ts_name = 'forecast_' + selected_date + '.csv'
    ts_complete_path = os.path.join(ts_path, ts_name)

    with open(ts_complete_path, "wb") as total_df_out:
        total_df.to_csv(total_df_out, index= False)
    total_df_out.close()


smhiAPI_acquire_realtime_forecast('today')
smhiAPI_acquire_realtime_forecast('tomorrow')





