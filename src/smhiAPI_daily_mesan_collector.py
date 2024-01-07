import requests
import pygrib
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime


def get_wind_dir_label(wind_dir_degree):
  '''
  Returns the wind direction label (N, S, W, E, NE, etc...), according
  to the wind_direction_degree passed as input, in the interval [-180°, +180°].
  '''
  wind_dir_label = ''

  pi         = 180
  pi_half    = pi/2
  pi_quarter = pi/4
  pi_eighth  = pi/8

  # WEST
  if wind_dir_degree < -pi + pi_eighth:
    wind_dir_label = 'W'
  # SOUTH-WEST
  elif wind_dir_degree < -pi + pi_quarter + pi_eighth:
    wind_dir_label = 'SW'
  # SOUTH
  elif wind_dir_degree < -pi + pi_half + pi_eighth:
    wind_dir_label = 'S'
  # SOUTH-EAST
  elif wind_dir_degree < -pi + pi_half + pi_quarter + pi_eighth:
    wind_dir_label = 'SE'
  # EAST
  elif wind_dir_degree < pi_eighth:
    wind_dir_label = 'E'
  # NORTH-EAST
  elif wind_dir_degree < pi_quarter + pi_eighth:
    wind_dir_label = 'NE'
  # NORTH
  elif wind_dir_degree < pi_half + pi_eighth:
    wind_dir_label = 'N'
  # NORTH-WEST
  elif wind_dir_degree < pi_half + pi_quarter + pi_eighth:
    wind_dir_label = 'NW'
  # WEST
  else:
    wind_dir_label = 'W'

  return wind_dir_label


def one_day_backward(year, month, day):
  '''
  Return "year", "month" and "day" numbers of the day before the inserted day
  It works for all the possible years from 1592
  '''
  
  if (day == 1):
    # If this is the first day of the Year
    if month == 1:
      day = 31
      month = 12
      year = year - 1

    # If I need to return back to Febraury
    elif month == 3:
      # If the current year is a leap year
      if (year % 4 == 0):
        day = 29
      else:
        day = 28

      month = 2

    # If the previous month has 30 days
    elif month == 5 or month == 7 or month == 10 or month == 12:
      day = 30
      month = month - 1
    
    else:
       day = 31
       month = month - 1

  else:
     day = day - 1

  return year, month, day


def one_day_forward(year, month, day):
  '''
  Return "year", "month" and "day" numbers of the day after the inserted day
  It works for all the possible years from 1592
  '''
  if month == 12:
    if day == 31:
      day = 1
      month = 1
      year = year + 1
    else:
        day = day + 1

  elif month == 2:
    if (day == 28):
      if (year % 4 == 0):
        day = 29
      else:
        day = 1
        month = 3
    elif (day == 29):
      day = 1
      month = 3
    else:
      day = day + 1
    
  elif month == 4 or month == 6 or month == 9 or month == 11:
    if (day == 30):
      month = month + 1
      day = 1
    else:
      day = day + 1
  
  else:
    day = day + 1

  return year, month, day
         

def one_hour_backward(year, month, day, hour):
  '''
  Return "year", "month" and "day" and "hour" numbers of the hour before the inserted hour
  It does not consider the DST, since it is not related to any location or time-zone
  '''

  # If it is midnight, it returns one day back as well
  if (hour == 0):
      year, month, day = one_day_backward(year, month, day)
      hour = 23
  else:
    hour = hour - 1

  return year, month, day, hour


def one_hour_forward(year, month, day, hour):
  '''
  Return "year", "month" and "day" and "hour" numbers of the hour after the inserted hour
  It does not consider the DST, since it is not related to any location or time-zone
  '''
  if (hour == 23):
    year, month, day = one_day_forward(year, month, day)
    hour = 0
  else:
    hour = hour + 1

  return year, month, day, hour


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


def smhiAPI_get_hour_from_datetime(timestamp):
    '''
    Extract the hour from a string containing a datetime in the format "%Y-%m-%dT%H:%M:%SZ"
    '''
    datetime_format = "%Y-%m-%dT%H:%M:%SZ"
    datetime_ts     = datetime.strptime(timestamp, datetime_format)
    hour            = datetime_ts.hour

    return hour


def get_year_month_label(year, month, mode):
  '''
  Return the year_month in the format wanted by the different APIs file structure, by passing 
  the year, month and the mode. It pads with 0 when needed. The "mode" can be specified 
  between "hyphen", "underscore" and "empty" and it determines which divider you will find in
  the year_month_label between the different input passed (e.g. 2024-01 or 20240105)
  '''
  year_month_label = ''

  year_label = str(year)
  month_label = ''
  if month not in {10, 11, 12}:
    month_label = '0' + str(month)
  else:
    month_label = str(month)
  
  if mode == 'hyphen':
    year_month_label = year_label + '-' + month_label
  elif mode == 'underscore':
    year_month_label = year_label + '_' + month_label
  elif mode == 'empty':
    year_month_label = year_label + month_label

  return year_month_label


def get_date_label(year, month, day, mode):
  ''' 
  Return the date in the format wanted by the different APIs file structure, by passing 
  the year, month, day and the mode. It pads with 0 when needed. The "mode" can be specified 
  between "hyphen", "underscore" and "empty" and it determines which divider you will find in
  the date_label between the different input passed (e.g. 2024-01-05 or 20240105)
  '''

  date_label = ''
  year_month_label = get_year_month_label(year, month, mode)
  
  day_label = ''
  if day < 10:
      day_label = '0' + str(day)
  else:
      day_label = str(day)
  
  if mode == 'hyphen':
    date_label = year_month_label + '-' + day_label
  elif mode == 'underscore':
    date_label = year_month_label + '_' + day_label
  elif mode == 'empty':
    date_label = year_month_label + day_label

  return date_label


def get_padded_hour(hour):
  '''
  Given an hour (int) return back the hour in the format hh (str)
  (e.g. get_padded_hour(1) -> 01, get_padded_hour(15) -> 15).
  '''
  hour_label = '' 
  
  if hour < 10:
    hour_label = '0' + str(hour)
  else:
    hour_label = str(hour)

  return hour_label


def get_mesan_date_label(year, month, day, hour, mode):
  '''
  Get the date/timestamp in the format wanted by the SMHI Historical API file structure, by passing 
  the year, month, day and hour. It pads with 0 when needed.
  '''
  mesan_label = ''
  hour_label = get_padded_hour(hour) + '00'
  date_label = get_date_label(year, month, day, mode)
  
  if mode == 'hyphen':
    mesan_label = date_label + '-' + hour_label
  elif mode == 'underscore':
    mesan_label = date_label + '_' + hour_label
  elif mode == 'empty':
    mesan_label = date_label + hour_label
  
  return mesan_label


def smhiAPI_get_grib_identifier(yyyymmdd, yyyymm, hour):
    hh = get_padded_hour(hour)
    identifier = yyyymm + '/MESAN_' + yyyymmdd + hh + '00+000H00M'
    return identifier


def smhiAPI_get_daily_grib_datestamps(year, month, day, dst):
    '''
    Given a year, month, day and the DST value, return a dictionary containing pairs
    of key-value for the full inserted day, with "key" equal to the wanted stockholm_time 
    and value equal to the datestamps of the GRIB file corresponding to the wanted stockholm_time.
    It works only for the 2024, due to different DST condition year by year.
    '''

    hour_dict   = {}
    hour_keys   = []
    hour_values = []
    date_stamps = []
    hour_before = 0

    yyyybefore, mmbefore, ddbefore = one_day_backward(year, month, day)
    yyyymmbefore                   = get_year_month_label(yyyybefore, mmbefore, 'empty')
    yyyymmddbefore                 = get_date_label(yyyybefore, mmbefore, ddbefore, 'empty')
    yyyymmcurrent                  = get_year_month_label(year, month, 'empty')
    yyyymmddcurrent                = get_date_label(year, month, day, 'empty')

    # Add the hour key for midnight
    hour_keys.append(0)

    if (month == 3 and day == 31):
        hour_keys.append(2)
        hour_values.extend((23,0,1,2,3))
        hour_before = 2
    elif (month == 10 and day == 27):
        hour_keys.extend((1,2))
        hour_values.extend((22,23,1,2,3))
        hour_before = 2
    elif (dst):
        hour_keys.extend((1,2))
        hour_values.extend((22,23,0,1,2,3))
        hour_before = 2
    else:
        hour_keys.extend((1,2))
        hour_values.extend((23,0,1,2,3))
        hour_before = 1

    hour_keys.extend(range(3,24))
    hour_values.extend(range(4,22))

    if ((not dst) or (month == 10 and day == 27)):
        hour_values.append(22)


    # Get the date_stamps for each hour_value
    counter = 0
    for hour_value in hour_values:
        if (counter < hour_before):
            date_stamps.append(smhiAPI_get_grib_identifier(yyyymmddbefore, yyyymmbefore, hour_value))
        else:
            date_stamps.append(smhiAPI_get_grib_identifier(yyyymmddcurrent, yyyymmcurrent, hour_value))

        counter += 1

    hour_dict = dict(zip(hour_keys, date_stamps))

    return hour_dict


def smhiAPI_acquire_daily_mesan_historical_plugin(year, month, day, dst):
    '''
    Get the daily MESAN analysis of a specific day. It get the online GRIB file from smhiAPI OpenData
    and give as result dataframe with all the uniformized and casted data, according to the rest of the project.
    This works only in 2024, due to the difference of DST changing year by year.
    '''
    # Set the latitude and longitude limits
    target_latitude_down  = 59.575368
    target_latitude_up    = 59.600368
    target_longitude_down = 17.875208
    target_longitude_up   = 17.876174

    # Set the skeleton of the dataframe
    columns = ['date', 'time', 'temperature', 'visibility', 'pressure', 'humidity', 'gusts_wind', 'u_wind', 'v_wind', 'prep_1h', 
                        'snow_1h', 'gradient_snow', 'total_cloud', 'low_cloud', 'medium_cloud', 'high_cloud', 'type_prep', 'sort_prep']
    weather_df = pd.DataFrame(columns=columns)

    # Create new_row_attr to add future files' rows to the dataframe and set counter to 0
    new_row_attr       = []
    file_counter       = 0
    checkpoint_counter = 0

    # Select after how many files a new checkpoint is saved
    check_step = 120    #(10s each GRIB file -> 20*60s = 1200s -> ~120 files)


    # Insert the limit of wanted iteration ranges:
    mesan_year    = year
    mesan_month   = month
    mesan_day     = day
    starting_hour = 0
    ending_hour   = 23

    # Always imposed limit for query
    if (month in {4, 6, 9, 11} and day in {31}) or (month in {2} and (year % 4 == 0) and day in {30, 31}) or (month in {2} and (year % 4 != 0) and day in {29, 30, 31}):
        raise Exception('This day does not exist')
    # Requirements checking for this function
    if (year != 2024):
        raise Exception('This process works only in 2024')
    
    # Get pairs of (stockholm hour -> grib_datestamps)
    datestamp_dict = smhiAPI_get_daily_grib_datestamps(year, month, day, dst)

    for datestamp in datestamp_dict.values():
        hour_grib_url = 'https://opendata-download-grid-archive.smhi.se/data/6/' + datestamp
        hour_response = requests.get(hour_grib_url)
        print(hour_grib_url)

        file_name = 'ARN_' + new_row_date + '_' + hh
        save_path = "/mnt/c/Developer/University/SML/sml-project-2023-manfredi-meneghin/datasets/smhi_historical_data/"
        complete_name = os.path.join(save_path, file_name)



































def smhiAPI_acquire_daily_mesan(mode):
    '''
    Acquire the daily mesan analysis of one full day from smhiAPI Forecast
    Depending on the 'mode', that can be 'yesterday' or 'today'
    '''

    year, month, day, hour, dst = get_current_date_time_and_dst()

    # Set the time to midnight
    hour = 0

    # Save the selected date to further save the created file
    selected_date = get_date_label(year, month, day, 'hyphen')

    # If mode is 'yesterday', roll-back to the historical data extraction process
    if (mode == 'yesterday'):
        smhiAPI_acquire_daily_mesan_historical_plugin(year, month, day)

    # if the mode is 'today', proceed with the extraction through smhiAPI MESAN Analysis
    else:
        # Select the maximum numbers of hours that can be retrieved by the 'today' mode
        time_range_hour = 24

        # Set the skeleton of the dataframe
        mesan_df_columns = ["t", "vis", "msl", "r", "gust", "ws", "wd", "tcc_mean", "lcc_mean", "mcc_mean", "hcc_mean", "pcat"]
        mesan_df         = pd.DataFrame(columns=mesan_df_columns)

        # Get the current ARN forecast (it is in UTC time, with always DST off)
        mesan_url    = "https://opendata-download-metanalys.smhi.se/api/category/mesan1g/version/2/geotype/point/lon/17.8752/lat/59.5753/data.json"
        response     = requests.get(mesan_url)
        responseJson = response.json()


        # The desired_valid_time will change during the day
        hour_limit = 0
        desired_valid_times = []
        yyyy, mm, dd, hh = year, month, day, hour

        hour_limit = smhiAPI_get_hour_from_datetime(responseJson["timeSeries"][0]["validTime"])

        # Set the clock on hour back to match with the UCT time. If DST on, move the clock two hours back.
        yyyy, mm, dd, hh = one_hour_backward(yyyy, mm, dd, hh)
        if (dst):
            yyyy, mm, dd, hh = one_hour_backward(yyyy, mm, dd, hh)
            
        for i in range(time_range_hour):
            datestamp = get_date_label(yyyy, mm, dd, 'hyphen')
            hourstamp = get_padded_hour(hh)

            valid_time = str(datestamp) + 'T' + str(hourstamp) + ':00:00Z'
            desired_valid_times.append(valid_time)

            # Stop when the valid_times are expired
            if (hh == hour_limit):
                break
                
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
                parameters_to_retrieve = ["t", "vis", "msl", "r", "gust", "ws", "wd", "tcc", "lcc", "mcc", "hcc", "prsort"]

                # Extract values for the specified parameters, add them to the future df row
                new_row_attr = []
                for param_name in parameters_to_retrieve:
                    param_value = next(
                        (param["values"][0] for param in desired_time_series["parameters"] if param["name"] == param_name),
                        None
                    )
                    new_row_attr.append(param_value)

                mesan_df.loc[len(mesan_df.index)] = new_row_attr
            else:
                print(f"No data found for validTime {desired_valid_time}")


        # Create the date and time dataframe columns
        datetime_df = pd.DataFrame(columns=['date', 'time'])


        # The file read belong the all the hours following the current hour, so set the time one hour forward
        yyyy, mm, dd, hh = year, month, day, hour

        # Add as many row as the lenght of the forecast_df
        for i in range(len(mesan_df.index)):
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
                'low_cloud', 'medium_cloud', 'high_cloud', 'sort_prep']] = mesan_df[["t", "vis", "msl", "r", 
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

    

    ### FIX THE NAME
    # Save the forecast dataframe in a new file (.csv)
    ts_path = "/mnt/c/Developer/University/SML/sml-project-2023-manfredi-meneghin/datasets/smhi_daily_data/"
    ts_name = 'mesan_' + selected_date + '.csv'
    ts_complete_path = os.path.join(ts_path, ts_name)

    with open(ts_complete_path, "wb") as total_df_out:
        total_df.to_csv(total_df_out, index= False)
    total_df_out.close()


smhiAPI_acquire_daily_mesan('today')
smhiAPI_acquire_daily_mesan('yesterday')