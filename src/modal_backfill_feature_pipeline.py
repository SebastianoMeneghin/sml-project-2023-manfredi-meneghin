import os
import modal

stub = modal.Stub("flight_delays_backfill_feature_pipeline_daily")
image = modal.Image.debian_slim().pip_install(["hopsworks", "joblib", "seaborn","scikit-learn==1.1.1", "numpy",
                                               "pandas", "pandasql", "xgboost", "cfgrib", "eccodes", "pygrib"])

@stub.function(cpu=1.0, image=image, schedule=modal.Cron('0 2 * * *'), secret=modal.Secret.from_name("hopsworks_sml_project"))
def f():
    g()


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


def get_today_date():
    '''
    Return today's year, month and day numbers
    '''
    import json
    import requests
    from datetime import datetime


    # Get today's date through TimeAPI
    time_url = "https://worldtimeapi.org/api/timezone/Europe/Stockholm"
    time_response     = requests.get(time_url)
    time_responseJson = time_response.json()

    # Extract datetime
    datetime_str    = time_responseJson["datetime"]
    datetime_object = datetime.fromisoformat(datetime_str[:-6])  # Remove the timezone offset for parsing

    # Extract components from datetime
    day   = datetime_object.day
    month = datetime_object.month
    year  = datetime_object.year

    return year, month, day


def swedaviaAPI_flight_delay(scheduledDepTime, actualDepTime):
    '''
    Calculate the delay of a flight extracted through SwedaviaAPI
    given the scheduledDepTime and the actualDepTime
    '''
    import math
    from datetime import datetime

    # Transform the strings into datetime objects and calculate the delay in minutes
    datetime_format = "%Y-%m-%dT%H:%M:%SZ"
    scheduled_datetime = datetime.strptime(scheduledDepTime, datetime_format)
    actual_datetime = datetime.strptime(actualDepTime, datetime_format)

    delay_datetime = actual_datetime - scheduled_datetime
    delay_minutes  = math.ceil(delay_datetime.total_seconds() / 60)

    # If the flight departed beforehand, the delay is 0
    if (delay_minutes <= 0):
        return 0
    else:
        return delay_minutes
    

def swedaviaAPI_correct_UCT(time):
    '''
    Given a time in the format "%Y-%m-%dT%H:%M:%SZ" (format of SwedaviaAPI), return the
    equivalent time in Stockholm Time (+01:00 or +02:00 when DST on).
    It works only for the 2024, due to different DST condition year by year.
    '''
    import math
    from datetime import datetime

    # Destructure the received time
    datetime_format = "%Y-%m-%dT%H:%M:%SZ"
    scheduled_datetime = datetime.strptime(time, datetime_format)
    
    yyyy    = scheduled_datetime.year
    mm      = scheduled_datetime.month
    dd      = scheduled_datetime.day
    hh      = scheduled_datetime.hour
    minutes = scheduled_datetime.minute
    ss      = scheduled_datetime.second

    if (yyyy != 2024):
        raise Exception("It works only with dates belonging to 2024")
    
    # According to the time-zone differences, the clock is moved forward or backward (In Sweden 2023: +01:00)
    yyyy,mm,dd,hh = one_hour_forward(yyyy,mm,dd,hh)

    # Set the clock an additional hour ahead when the DST was active (Sweden 2024, from 31st March - 01:00 (UCT00:00) to 27th October - 01:00 (UCT00:00))
    if (mm == 3 and dd == 31 and hh >= 1) or (mm in {4,5,6,7,8,9}) or (mm == 10 and dd <= 26) or (mm == 10 and dd == 27 and hh <= 1):
        yyyy,mm,dd,hh = one_hour_forward(yyyy,mm,dd,hh)

    # Calculate now stockholm time and create a string with the required datetime_format
    stockholm_time = datetime(yyyy, mm, dd, hh, minutes, ss).strftime(datetime_format)

    return stockholm_time


def swedaviaAPI_daily_collector(mode):
    '''
    Get a full-day collection of flight info from Swedavia FlightInfoAPI
    The input mode could be "yesterday", "today" or "tomorrow", depending on the day
    of interest.
    The file is then saved in a .json format
    '''
    import os
    import re
    import json
    import requests
    import hopsworks
    import pygrib
    import math
    import pandas as pd
    from datetime import datetime

    yyyy, mm, dd = get_today_date()

    if (mode == 'yesterday'):
        yyyy, mm, dd = one_day_backward(yyyy, mm, dd)

    elif (mode == 'tomorrow'):
        yyyy, mm, dd = one_day_forward(yyyy, mm, dd)

    date_label = get_date_label(yyyy, mm, dd, 'hyphen')

    # Create the request_url, then get the subscription key from Swedavia API and set them in the header
    swedavia_url     = 'https://api.swedavia.se/flightinfo/v2/ARN/departures/' + date_label
    subscription_key = 'SWEDAVIA_API_KEY'
    headers = {
        "Ocp-Apim-Subscription-Key": subscription_key,
        "Accept": "application/json",
        "Content-Type": 'application/json',
    }

    # Make the API request for Swedavia API
    response = requests.get(swedavia_url, headers = headers)
    flights_swedavia = response.json()


    # Get the flights' info from the json file and sort the flights depending on their departure time
    flight_list_unsorted = flights_swedavia['flights']
    flight_list = sorted(flight_list_unsorted, key=lambda x: x['departureTime']['scheduledUtc'])


    # For each flight, extracted the wanted info
    flight_infos = []
    for flight in flight_list:
        # Set pattern for different values
        terminal_pattern = r'\d+'
        depDelay = None

        # If the flight info allows to calculate delay:
        if (flight.get("departureTime").get("actualUtc") != None):
            depDelay = swedaviaAPI_flight_delay(flight["departureTime"]["scheduledUtc"], flight["departureTime"]["actualUtc"])

        # If depTerminal or depGate are not present in the data, set them to 0
        # If the arrScheduledTime is not present, set it to the depScheduledTime
        flight_info = {
            "status"           : str(flight["locationAndStatus"]["flightLegStatusEnglish"]).lower(),
            "depApIataCode"    : str(flight["flightLegIdentifier"]["departureAirportIata"]).lower(),
            "depDelay"         : depDelay,
            "depScheduledTime" : swedaviaAPI_correct_UCT(flight["departureTime"]["scheduledUtc"]),
            "depApTerminal"    : int(re.findall(terminal_pattern, flight["locationAndStatus"].get("terminal", 0))[0]),
            "depApGate"        : str(flight["locationAndStatus"].get("gate", 0)).lower(),
            "arrScheduledTime" : swedaviaAPI_correct_UCT(flight.get("arrivalTime", flight["departureTime"]["scheduledUtc"])),
            "arrApIataCode"    : str(flight["flightLegIdentifier"]["arrivalAirportIata"]).lower(),
            "airlineIataCode"  : str(flight["airlineOperator"].get("iata", '0')).lower(),
            "flightIataNumber" : str(flight.get("flightId", '0')).lower(),
        }
        
        flight_infos.append(flight_info)


    # Create a json file to return as result
    result = json.dumps(flight_infos)

    return result, date_label


def swedaviaAPI_num_flight_within(interval_min, flight_df):
    '''
    Return for each row of the dataframe, the number of flights departed from the same airport
    in the interval specified in minutes (-interval_min/2, + interval_min/2).
    This works with swedaviaAPI flights info, having datetime_format "%Y-%m-%dT%H:%M:%SZ"
    '''
    import os
    import re
    import json
    import math
    import pandas as pd
    from datetime import datetime

    datetime_format = "%Y-%m-%dT%H:%M:%SZ"

    # Row range should be reduced or increased depending on the airport
    row_range = 50
    total_flight  = flight_df.shape[0]
    flight_within = []

    for row in range(total_flight):
        flight_counter = 0
        temp_df = pd.DataFrame()

        # Get departure time of the flight
        flight_dep_time = flight_df.at[row, 'depScheduledTime']
        flight_datetime = datetime.strptime(flight_dep_time, datetime_format)

        # Get an adjacent part of the full flight_df to compare with the current flight
        if row < row_range:
            temp_df = flight_df[: int(row_range/2)]
        elif row > total_flight - row_range:
            temp_df = flight_df[total_flight - int(row_range/2):]
        else:
            temp_df = flight_df[int(row - row_range/2) : int(row + row_range/2)]


        temp_df.reset_index(inplace=True)
        
        for temp_row in range(temp_df.shape[0]):
            # Get departure time of the temp flight
            flight_temp_time = temp_df.at[temp_row, 'depScheduledTime']
            temp_datetime    = datetime.strptime(flight_temp_time, datetime_format)

            # Get the delta time and add bring it to minutes
            delta_time = abs(flight_datetime - temp_datetime)
            gap_time   = math.ceil(delta_time.total_seconds()/60)

            if (gap_time < interval_min/2):
                flight_counter += 1
        
        flight_within.append(flight_counter)
    

    column_name = 'flight_within_' + str(interval_min) + 'min'
    return flight_within, column_name


def get_day_of_week(year, month, day):
    '''
    The function returns the day of the week, given a specific date
    It works only with date from 2000 to 2099 and it consider as first day (1) Monday,
    and as last day (7) Sunday.
    '''
    import os
    import re
    import json
    import requests
    import hopsworks
    import pygrib
    import math
    import pandas as pd
    import pandasql as sqldf
    from datetime import datetime

    leap_year = False

    month_val       = 0
    day_val         = day
    year_val        = year - 2000
    year_val_fourth = math.floor(year_val / 4)

    if (year % 4 == 0):
        leap_year = True

    #if (month in {4,7}) or (month == 1 and leap_year): month_val = 0
    if (month == 1 and not leap_year) or (month == 10):
        month_val = 1

    elif (month == 5):
        month_val = 2

    elif (month == 2 and leap_year) or (month == 8):
        month_val = 3

    elif (month == 2 and not leap_year) or (month in {3,11}):
        month_val = 4

    elif (month == 6):
        month_val = 5

    elif (month in {9, 12}):
        month_val = 6

    
    sum_val = year_val + year_val_fourth + month_val + day_val - 2
    day_of_week = sum_val % 7

    if(day_of_week == 0):
        day_of_week = 7

    return day_of_week


def swedaviaAPI_flight_processor(json_file, json_date, mode):
    '''
    Process the data depending on the "mode" selected:
    - 'historical' process the data in order to save them into the DB
    - 'prediction' process the data in order to use them as feature for a prediction problem
    Return a dataframe containing the processed data
    '''
    import os
    import re
    import json
    import requests
    import hopsworks
    import pygrib
    import math
    import pandas as pd
    from datetime import datetime

    datetime_format = "%Y-%m-%dT%H:%M:%SZ"

    # Get the json file
    df = pd.read_json(json_file)

    # Drop all the flights having more than one flight_number (used by different companies to sell same tickets)
    df.drop_duplicates(subset = ['depScheduledTime', 'depApIataCode', 'arrApIataCode'], inplace= True)
    df.reset_index(inplace = True)
    df.drop(columns={'index'}, inplace = True)

    # Create new columns for future df's data
    new_column_names  = ['date','time', 'month', 'trip_time', 'day_of_week']
    new_column_values = []
    row_to_remove     = []
    status_set        = {}

    # Depending on the mode:
    if (mode == 'historical'):
        # Only flight that are departed (can give me info) are considered
        status_set = {'departed'}
    else:
        # Only flight that are scheduled are considered (not other cancelled or deleted 
        # flight, e.g. summer flights during winter days)
        status_set = {'scheduled'}

    for row in range(df.shape[0]):
        if (df.at[row, 'status'] not in status_set) or (mode == 'historical' and df.at[row, 'status'] == None):
            row_to_remove.append(row)

    # Remove the not wanted rows and reset the index
    df.drop(row_to_remove, inplace=True)
    df.reset_index(inplace = True)
    df.drop(columns={'index'}, inplace = True)

    if (df.shape[0] > 0):
        # Create data for new columns
        for row in range(df.shape[0]):
            # Get year, month, day and time from the selected row
            dep_ts       = df.at[row, 'depScheduledTime']
            dep_datetime = datetime.strptime(dep_ts, datetime_format)
            arr_ts = df.at[row, 'arrScheduledTime']
            arr_datetime = datetime.strptime(arr_ts, datetime_format)

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
        flight_within, column_name = swedaviaAPI_num_flight_within(60, df)
        df[column_name] = flight_within
        df[new_column_names] = new_column_values

    # Drop useless columns
    df.drop(columns={'depScheduledTime', 'arrScheduledTime'}, inplace = True)

    return df


def get_wind_dir_label(wind_dir_degree):
  '''
  Returns the wind direction label (N, S, W, E, NE, etc...), according
  to the wind_direction_degree passed as input, in the interval [-180°, +180°].
  '''
  import os
  import re
  import json
  import requests
  import hopsworks
  import pygrib
  import math
  import pandas as pd
  import pandasql as sqldf
  from datetime import datetime

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


def get_current_date_time_and_dst():
    '''
    Return today's year, month and day numbers
    '''
    import os
    import re
    import json
    import requests
    import hopsworks
    import pygrib
    import math
    import pandas as pd
    import pandasql as sqldf
    from datetime import datetime

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

    time_response.close()
    return year, month, day, hour, dst


def smhiAPI_get_hour_from_datetime(timestamp):
    '''
    Extract the hour from a string containing a datetime in the format "%Y-%m-%dT%H:%M:%SZ"
    '''
    import math
    from datetime import datetime

    datetime_format = "%Y-%m-%dT%H:%M:%SZ"
    datetime_ts     = datetime.strptime(timestamp, datetime_format)
    hour            = datetime_ts.hour

    return hour


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
    import math
    from datetime import datetime

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
    import os
    import re
    import json
    import requests
    import hopsworks
    import pygrib
    import math
    import pandas as pd
    import numpy as np
    from datetime import datetime

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
    new_row_attr = []

    date_label = get_date_label(year, month, day, 'hyphen')

    # Always imposed limit for query
    if (month in {4, 6, 9, 11} and day in {31}) or (month in {2} and (year % 4 == 0) and day in {30, 31}) or (month in {2} and (year % 4 != 0) and day in {29, 30, 31}):
        raise Exception('This day does not exist')
    # Requirements checking for this function
    if (year != 2024):
        raise Exception('This process works only in 2024')
    
    # Get pairs of (stockholm hour -> grib_datestamps)
    datestamp_dict = smhiAPI_get_daily_grib_datestamps(year, month, day, dst)

    # Get grib data from each datestamp
    for datestamp in datestamp_dict.items():

        new_row_attr = []
        new_row_time = datestamp[0]
        new_row_attr.append(date_label)
        new_row_attr.append(new_row_time)

        hour_grib_url = 'https://opendata-download-grid-archive.smhi.se/data/6/' + datestamp[1]
        hour_response = requests.get(hour_grib_url)

        file_name = 'smhiAPI_' + date_label + '_' + get_padded_hour(datestamp[0])
        save_path = "/mnt/c/Developer/University/SML/sml-project-2023-manfredi-meneghin/datasets/smhi_historical_data/"
        complete_name = os.path.join(save_path, file_name)


        with open(complete_name, "wb") as outfile: 
            outfile.write(hour_response.content)

            grib_path = complete_name
            grib_file = pygrib.open(grib_path)

            for label in ['Temperature', 'Visibility', 'Pressure reduced to MSL', 'Relative humidity', 'Wind gusts', 'u-component of wind', 
                                'v-component of wind', '1 hour precipitation', '1 hour fresh snow cover', 
                                'Snowfall (convective + stratiform) gradient', 'Total cloud cover', 'Low cloud cover', 
                                'Medium cloud cove', 'High cloud cover', 'Type of precipitation', 'Sort of precipitation']:
                        
                #df_label = get_df_label_from_grib_label(label)
                temp_grb = grib_file.select(name=label)[0]
                part, latss, lotss = temp_grb.data(lat1=target_latitude_down,lat2=target_latitude_up,lon1=target_longitude_down,lon2=target_longitude_up)

                new_row_attr.append(part[0])


            weather_df.loc[len(weather_df.index)] = new_row_attr

        outfile.close()
        os.remove(complete_name)

    print('Processing meteorological data about:' + date_label)

    # Reset the index and drop the old column
    weather_df.reset_index(inplace = True)
    weather_df.drop(columns={'index'}, inplace = True)

    # Make humidity a categorical value
    for row in range(weather_df.shape[0]):
        humidity = weather_df.at[row,'humidity']

        if humidity < 0.375:
            weather_df.at[row,'humidity'] = 1

        elif humidity < 0.5:
            weather_df.at[row,'humidity'] = 2

        elif humidity < 0.625:
            weather_df.at[row,'humidity'] = 3

        elif humidity < 0.75:
            weather_df.at[row,'humidity'] = 4

        elif humidity < 0.875:
            weather_df.at[row,'humidity'] = 5

        elif humidity:
            weather_df.at[row,'humidity'] = 6


    # Make cloud cover categorical values (octas)
    cloud_labels = ['total_cloud', 'medium_cloud', 'high_cloud', 'low_cloud']
    for row in range(weather_df.shape[0]):
        for label in cloud_labels:
            cloud_level = weather_df.at[row, label]

            if cloud_level < 1/16:
                weather_df.at[row, label] = 0

            elif cloud_level < 3/16:
                weather_df.at[row, label] = 1

            elif cloud_level < 5/16:
                weather_df.at[row, label] = 2

            elif cloud_level < 7/16:
                weather_df.at[row, label] = 3

            elif cloud_level < 9/16:
                weather_df.at[row, label] = 4

            elif cloud_level < 11/16:
                weather_df.at[row, label] = 5

            elif cloud_level < 13/16:
                weather_df.at[row, label] = 6

            elif cloud_level < 15/16:
                weather_df.at[row, label] = 7

            else:
                weather_df.at[row, label] = 8


    # Bring visibility to km instead of m
    for row in range(weather_df.shape[0]):
        weather_df.at[row, 'visibility'] = weather_df.at[row, 'visibility'] / 1000


    # Bring pressure to hPa instead of Pa
    for row in range(weather_df.shape[0]):
        weather_df.at[row, 'pressure'] = weather_df.at[row, 'pressure'] / 100

    # Make pressure a categorical variable (binning)
    min_pressure = 970
    max_pressure = 1060
    num_interval = 8
    division_gap = (max_pressure - min_pressure)/num_interval

    for row in range(weather_df.shape[0]):
        pressure = weather_df.at[row,'pressure']

        for multiplier in range(1, num_interval):
            if pressure < min_pressure + multiplier*division_gap:
                weather_df.at[row,'pressure'] = multiplier
                break;

    # Bring temperature to Celsius instead of Kelvin
    for row in range(weather_df.shape[0]):
        weather_df.at[row, 'temperature'] = weather_df.at[row, 'temperature'] - 273.15


    # Get windspeed and wind direction from u and v components, then rename columns
    for row in range(weather_df.shape[0]):
        u_wind = weather_df.at[row, 'u_wind']
        v_wind = weather_df.at[row, 'v_wind']

        wind_speed = math.sqrt(pow(u_wind, 2) + pow(v_wind, 2))
        wind_dir   = np.arctan2(v_wind, u_wind) * (180 / np.pi)

        wind_dir_label = get_wind_dir_label(wind_dir)

        weather_df.at[row, 'u_wind'] = wind_speed
        weather_df.at[row, 'v_wind'] = wind_dir_label

    # Remove useless columns
    weather_df.rename(columns={'u_wind': 'wind_speed', 'v_wind': 'wind_dir'}, inplace= True)
    weather_df.drop(columns={'prep_1h', 'snow_1h', 'gradient_snow', 'type_prep'}, inplace=True)

    # Reset the index and drop the old column
    weather_df.reset_index(inplace = True)
    weather_df.drop(columns={'index'}, inplace = True)

    print('\n\n**Process finished**\n\n')

    # Return the calculated dataframe
    return weather_df


def smhiAPI_acquire_daily_mesan(mode):
    '''
    Acquire the daily mesan analysis of one full day from smhiAPI Forecast
    Depending on the 'mode', that can be 'yesterday' or 'today'
    '''
    import os
    import re
    import json
    import requests
    import math
    import pandas as pd
    from datetime import datetime

    year, month, day, hour, dst = get_current_date_time_and_dst()

    # Set the time to midnight
    hour = 0

    # Save the selected date to further save the created file
    selected_date = get_date_label(year, month, day, 'hyphen')

    total_df = pd.DataFrame()
    # If mode is 'yesterday', roll-back to the historical data extraction process
    if (mode == 'yesterday'):
        yearbefore, monthbefore, daybefore = one_day_backward(year, month, day)
        datebefore                         = get_date_label(yearbefore, monthbefore, daybefore, 'hyphen')

        print('Acquiring meteorological data about:' + datebefore)

        selected_date = datebefore
        total_df      = smhiAPI_acquire_daily_mesan_historical_plugin(yearbefore, monthbefore, daybefore, dst)

    # if the mode is 'today', proceed with the extraction through smhiAPI MESAN Analysis
    else:
        print('Acquiring meteorological data about:' + selected_date)

        # Select the maximum numbers of hours that can be retrieved by the 'today' mode
        time_range_hour = 24

        # Set the skeleton of the dataframe
        mesan_df_columns = ["t", "vis", "msl", "r", "gust", "ws", "wd", "tcc_mean", "lcc_mean", "mcc_mean", "hcc_mean", "pcat"]
        mesan_df         = pd.DataFrame(columns=mesan_df_columns)

        # Get the current ARN forecast (it is in UTC time, with always DST off)
        mesan_url    = "https://opendata-download-metanalys.smhi.se/api/category/mesan1g/version/2/geotype/point/lon/17.8752/lat/59.5753/data.json"
        response     = requests.get(mesan_url)
        responseJson = response.json()

        print('Processing meteorological data about:' + selected_date)

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

    # Return the processed data
    return total_df


def daily_flight_weather_dataframe_merger(flight_df, weather_df):
    import os
    import re
    import json
    import pandas as pd
    import numpy as np
    from datetime import datetime

    # Merge the two DataFrames on 'date' and 'time'
    merged_df = pd.merge(flight_df, weather_df, on=['date', 'time'], how='inner')
    merged_df = merged_df.loc[:,~merged_df.columns.duplicated()].copy()

    print('Dataset merged created!')
    return merged_df


def dataset_normalizer(dataset_df):
    '''
    Given a dataset with the columns names extracted from the APIs data, return a dataset (dataframe)
    with the name of the columns according to the feature group on Hopsworks
    '''
    import math
    import pandas as pd

    dataset_df.rename(columns={'depApIataCode' : 'dep_ap_iata_code', 'depDelay' : 'dep_delay', 'depApTerminal': 'dep_ap_terminal',
                                'depApGate': 'dep_ap_gate', 'arrApIataCode' : 'arr_ap_iata_code', 'airlineIataCode':'airline_iata_code',
                                'flightIataNumber':'flight_iata_number'}, inplace= True)
    
    return dataset_df


def collect_yesterday_flight_weather_info():
    '''
    Collect yesterday's flight and weather info
    Return a dataframe insertable on Hopsworks in 'flight_weather_dataset'
    '''
    import os
    import json
    import pandas as pd

    # Collect from SwedaviaAPI raw information about yesterday's departed flights
    row_yesterday_flight_json, selected_date = swedaviaAPI_daily_collector('yesterday')

    # Process raw information into a dataframe with wanted columns and data
    yesterday_flight_df = swedaviaAPI_flight_processor(row_yesterday_flight_json, selected_date, 'historical')

    # Collect from the SmhiAPI the yesterday's wheather measurements (detailed by hour)
    yesterday_wheather_df = smhiAPI_acquire_daily_mesan('yesterday')

    # Merge the two dataframes
    yesterday_fw_df = daily_flight_weather_dataframe_merger(yesterday_flight_df, yesterday_wheather_df)

    # Normalize the unified dataframe
    yesterday_fw_normalized_df = dataset_normalizer(yesterday_fw_df)

    return yesterday_fw_normalized_df


def g():
    import os
    import json
    import hopsworks
    import pandas as pd

    hopsworks_api_key = os.environ['HOPSWORKS_API_KEY']
    project = hopsworks.login(api_key_value = hopsworks_api_key)

    fs = project.get_feature_store()
    flight_weather_fg = fs.get_feature_group(name = 'flight_weather_dataset', version = 1)

    yesterday_fw_info_df = collect_yesterday_flight_weather_info()
    flight_weather_fg.insert(yesterday_fw_info_df)