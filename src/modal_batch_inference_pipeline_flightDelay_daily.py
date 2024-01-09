import os
import modal

stub = modal.Stub("flight_delays_batch_inference_pipeline_daily")
image = modal.Image.debian_slim().pip_install(["hopsworks", "joblib", "seaborn","scikit-learn==1.1.1", "numpy",
                                               "pandas", "pandasql", "xgboost", "cfgrib", "eccodes", "pygrib"])

@stub.function(cpu=1.0, image=image, schedule=modal.Cron('15 0 * * *'), secret=modal.Secret.from_name("hopsworks_sml_project"))
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
  import os
  import re
  import json
  import requests
  import hopsworks
  import joblib
  import math
  import pandas as pd
  import numpy as np
  from datetime import datetime

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
  import os
  import re
  import json
  import requests
  import hopsworks
  import joblib
  import math
  import pandas as pd
  import numpy as np
  from datetime import datetime

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
    import os
    import re
    import json
    import requests
    import hopsworks
    import joblib
    import math
    import pandas as pd
    import numpy as np
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
    import os
    import re
    import json
    import requests
    import hopsworks
    import joblib
    import math
    import pandas as pd
    import numpy as np
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
    import os
    import re
    import json
    import requests
    import hopsworks
    import joblib
    import math
    import pandas as pd
    import numpy as np
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
    import joblib
    import math
    import pandas as pd
    import numpy as np
    from datetime import datetime

    yyyy, mm, dd = get_today_date()

    if (mode == 'yesterday'):
        yyyy, mm, dd = one_day_backward(yyyy, mm, dd)

    elif (mode == 'tomorrow'):
        yyyy, mm, dd = one_day_forward(yyyy, mm, dd)

    date_label = get_date_label(yyyy, mm, dd, 'hyphen')

    # Create the request_url, then get the subscription key from Swedavia API and set them in the header
    swedavia_url     = 'https://api.swedavia.se/flightinfo/v2/ARN/departures/' + date_label
    subscription_key =  os.environ['SWEDAVIA_API_KEY']
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
    import requests
    import hopsworks
    import joblib
    import math
    import pandas as pd
    import numpy as np
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
    import joblib
    import math
    import pandas as pd
    import numpy as np
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
    Return a dataframe containing the processed data and a dataframe containing hh:MM for each flight
    '''
    import os
    import re
    import json
    import requests
    import hopsworks
    import joblib
    import math
    import pandas as pd
    import numpy as np
    from datetime import datetime

    datetime_format = "%Y-%m-%dT%H:%M:%SZ"

    # Get the json file
    df = pd.read_json(json_file)

    # Drop all the flights having more than one flight_number (used by different companies to sell same tickets)
    df.drop_duplicates(subset = ['depScheduledTime', 'depApIataCode', 'arrApIataCode'], inplace= True)
    df.reset_index(inplace = True)
    df.drop(columns={'index'}, inplace = True)

    # Create new columns for future df's data
    new_column_names  = ['time', 'month', 'trip_time', 'day_of_week']
    new_column_values = []
    new_date_names    = 'date'
    new_date_values   = []
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

    flight_hh = []
    flight_MM = []

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
            dep_MM   = dep_datetime.minute

            # Get additional information from the departure datetime
            dep_date_label  = get_date_label( dep_yyyy, dep_mm, dep_dd, 'hyphen')
            day_of_the_week = get_day_of_week(dep_yyyy, dep_mm, dep_dd)
            # Save now: date_label, hour, month, trip_time, day_of_the_week

            new_column_values.append([dep_hh, dep_mm, trip_time, day_of_the_week])
            new_date_values.append(dep_date_label)

            # In another lists, save hour and minute of each flight
            flight_hh.append(dep_hh)
            flight_MM.append(dep_MM)


        # Add the column "flight_within_60min" and calculate these values for each flight
        flight_within, column_name = swedaviaAPI_num_flight_within(60, df)
        df[column_name] = flight_within
        df[new_column_names] = new_column_values
        df[new_date_names] = new_date_values

        # Create another dataframe containing hh:MM for each flight
        hhMM_df = pd.DataFrame()
        hhMM_df['hh'] = flight_hh
        hhMM_df['MM'] = flight_MM

    # Drop useless columns
    df.drop(columns={'depScheduledTime', 'arrScheduledTime'}, inplace = True)

    return df, hhMM_df


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


def get_padded_minute(minute):
  '''
  Given an minute (int) return back the minute in the format MM (str)
  (e.g. get_padded_minute(1) -> 01, get_padded_minute(15) -> 15).
  '''
  minute_label = '' 
  
  if minute < 10:
    minute_label = '0' + str(minute)
  else:
    minute_label = str(minute)

  return minute_label


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
  import joblib
  import math
  import pandas as pd
  import numpy as np
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
    import joblib
    import math
    import pandas as pd
    import numpy as np
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

    return year, month, day, hour, dst


def smhiAPI_acquire_realtime_forecast(mode):
    '''
    Acquire the real-time weather forecast of one full day from smhiAPI Forecast
    Depending on the 'mode', that can be 'today' or 'tomorrow', it gets different data
    '''
    import os
    import re
    import json
    import requests
    import hopsworks
    import joblib
    import math
    import pandas as pd
    import numpy as np
    from datetime import datetime

    time_range_hour = 30
    year, month, day, hour, dst = get_current_date_time_and_dst()

    # If the mode is 'tomorrow', get the date of the following date starting from the midnight
    if (mode == 'tomorrow'):
        hour = 0
        year, month, day = one_day_forward(year, month, day)

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

    # Return processed forecast
    return total_df


def daily_flight_weather_dataframe_merger(flight_df, weather_df):
    
    import os
    import re
    import json
    import requests
    import hopsworks
    import joblib
    import math
    import pandas as pd
    import numpy as np
    from datetime import datetime


    flight_df.reset_index(inplace=True)
    flight_df.rename(columns={'index':'pointer'}, inplace = True)

    # Merge the two DataFrames on 'date' and 'time'
    merged_df = pd.merge(flight_df, weather_df, on=['date', 'time'], how='inner')
    
    removed_rows = flight_df.loc[~flight_df['pointer'].isin(merged_df['pointer']), 'pointer']
    removed_rows_list = removed_rows.tolist()
    print(removed_rows_list)

    merged_df.drop(columns={'pointer'}, inplace = True)
    merged_df = merged_df.loc[:,~merged_df.columns.duplicated()].copy()

    return merged_df, removed_rows_list


def dataset_normalizer(dataset_df):
    '''
    Given a dataset with the columns names extracted from the APIs data, return a dataset (dataframe)
    with the name of the columns according to the feature group on Hopsworks, and a dataframe with
    arr_ap_iata_code and flight_iata_number for every flight
    '''
    import os
    import re
    import json
    import requests
    import hopsworks
    import joblib
    import math
    import pandas as pd
    import numpy as np
    from datetime import datetime

    dataset_df.rename(columns={'depApIataCode' : 'dep_ap_iata_code', 'depDelay' : 'dep_delay', 'depApTerminal': 'dep_ap_terminal',
                                'depApGate': 'dep_ap_gate', 'arrApIataCode' : 'arr_ap_iata_code', 'airlineIataCode':'airline_iata_code',
                                'flightIataNumber':'flight_iata_number'}, inplace= True)
    
    df = dataset_df.drop(columns={'trip_time', 'dep_ap_gate', 'airline_iata_code', 'flight_iata_number', 
                     'arr_ap_iata_code', 'status','dep_ap_iata_code', 'date', 'high_cloud', 
                     'medium_cloud', 'low_cloud', 'gusts_wind'})

    # Some data should be casted to int64
    convert_column = ['pressure','total_cloud', 'sort_prep','humidity']
    for col in convert_column:
        df = df.astype({col: 'int64'})

    # Make wind_dir a categorical feature with numbers and not string labels
    dir_dict = {'SW':0,'S':1,'SE':2,'E':3,'NE':4,'N':5,'NW':6,'W':7}
    direction_list = []
    for row in range(df.shape[0]):
        direction = df.at[row, 'wind_dir']
        number = dir_dict.get(direction)
        direction_list.append(number)
    df.drop(columns={'wind_dir'}, inplace = True)
    df['wind_dir'] = direction_list

    # Gets residual iata's information and return
    iata_df = dataset_df[['arr_ap_iata_code', 'flight_iata_number']]

    return df, iata_df


def get_hour_minute_timetable_label(hour, minute):
    '''
    Given hour (int) and minute (int) return them with format "hh:MM" (str)
    '''
    label_hh = get_padded_hour(hour)
    label_MM = get_padded_minute(minute)

    label = str(label_hh) + ":" + str(label_MM)

    return label


def get_ontime_timetable_label(hour, minute):
    '''
    Given hour (int) and minute (int) return them with format "hh:MM" (str)
    '''
    timestamp = get_hour_minute_timetable_label(hour, minute)
    return timestamp


def get_delayed_timetable_label(hour, minute, delay):
    '''
    Given hour (int), minute (int) and delay in minutes (int), return the delayed timestamp
    in the format "hh:MM" (str)
    '''
    import os
    import re
    import json
    import requests
    import hopsworks
    import joblib
    import math
    import pandas as pd
    import numpy as np
    from datetime import datetime

    hours_of_delay = math.floor(delay/60)
    mins_of_delay  = delay - hours_of_delay*60

    hours_of_delay =+ math.floor((minute + mins_of_delay)/60)

    hour_delayed   = (hour   + hours_of_delay) % 24
    minute_delayed = (minute + mins_of_delay ) % 60

    timestamp_delayed = get_hour_minute_timetable_label(hour_delayed, minute_delayed)

    return timestamp_delayed


def get_timetable_labels(timetable_df):
    '''
    Given a timetable dataframe, calculate ontime and delayed timetable labels.
    Return a dataframe with columns {'ontime', 'delayed', 'airport', 'flight_number'}, drop
    the columns {'hh', 'MM', 'delay'}
    '''
    import os
    import re
    import json
    import requests
    import hopsworks
    import joblib
    import math
    import pandas as pd
    import numpy as np
    from datetime import datetime

    # Calculate today timetable's ontime hh:MM and delayed hh:MM, then drop 'hh', 'MM' and 'delay' columns
    ontime_labels  = []
    delayed_labels = []
    for row in range(timetable_df.shape[0]):
        hour_label, minute_label, delay_amount = timetable_df.at[row, 'hh'], timetable_df.at[row, 'MM'], timetable_df.at[row, 'delay']

        ontime_label  = get_ontime_timetable_label(hour_label, minute_label)
        delayed_label = get_delayed_timetable_label(hour_label, minute_label, delay_amount)

        ontime_labels.append(ontime_label)
        delayed_labels.append(delayed_label)
    
    # Add the new columns, drop the old ones
    timetable_df['ontime']  = ontime_labels
    timetable_df['delayed'] = delayed_labels
    timetable_df.drop(columns={'hh', 'MM', 'delay'}, inplace = True)

    return timetable_df


def collect_timetable_attributes(hhMM_df, iata_df, removed_rows):
    
    hhMM_df.drop(removed_rows, inplace = True)
    hhMM_df.reset_index(inplace = True)
    hhMM_df.drop(columns={'index'}, inplace = True)

    # Collect timetable information to visualize
    timetable_df                  = hhMM_df
    timetable_df['airport']       = iata_df['arr_ap_iata_code']
    timetable_df['flight_number'] = iata_df['flight_iata_number']

    return timetable_df


def collect_today_flight_weather_info():
    '''
    Collect today's flight and weather info
    Return a dataframe insertable on Hopsworks in 'flight_weather_dataset' and a dataframe containing hh:MM for each flight
    It is possible to make prediction holding this data (dep_delay has to be predicted)
    '''
    import os
    import re
    import json
    import requests
    import hopsworks
    import joblib
    import math
    import pandas as pd
    import numpy as np
    from datetime import datetime

    # Collect from SwedaviaAPI raw information about today's departing flights
    row_today_flight_json, selected_date = swedaviaAPI_daily_collector('today')

    # Process raw information into a dataframe with wanted columns and data
    today_flight_df, today_hhMM_df = swedaviaAPI_flight_processor(row_today_flight_json, selected_date, 'prediction')

    # Collect from the SmhiAPI the today's wheather measurements (detailed by hour)
    today_wheather_df = smhiAPI_acquire_realtime_forecast('today')

    # Merge the two dataframes
    today_fw_df, removed_rows = daily_flight_weather_dataframe_merger(today_flight_df, today_wheather_df)

    # Normalize the unified dataframe
    today_fw_normalized_df, today_iata_df = dataset_normalizer(today_fw_df)

    # Collect timetable information to visualize
    today_timetable_attr_df = collect_timetable_attributes(today_hhMM_df, today_iata_df, removed_rows)
                            
    return today_fw_normalized_df, today_timetable_attr_df


def collect_tomorrow_flight_weather_info():
    '''
    Collect tomorrow's flight and weather info
    Return a dataframe insertable on Hopsworks in 'flight_weather_dataset' and a dataframe containing hh:MM for each flight
    It is possible to make prediction holding this data (dep_delay has to be predicted)
    '''
    import os
    import re
    import json
    import requests
    import hopsworks
    import joblib
    import math
    import pandas as pd
    import numpy as np
    from datetime import datetime

    # Collect from SwedaviaAPI raw information about tomorrow's departing flights
    row_tomorrow_flight_json, selected_date = swedaviaAPI_daily_collector('tomorrow')

    # Process raw information into a dataframe with wanted columns and data
    tomorrow_flight_df, tomorrow_hhMM_df = swedaviaAPI_flight_processor(row_tomorrow_flight_json, selected_date, 'prediction')

    # Collect from the SmhiAPI the tomorrow's wheather measurements (detailed by hour)
    tomorrow_wheather_df = smhiAPI_acquire_realtime_forecast('tomorrow')

    # Merge the two dataframes
    tomorrow_fw_df, removed_rows = daily_flight_weather_dataframe_merger(tomorrow_flight_df, tomorrow_wheather_df)

    # Normalize the unified dataframe
    tomorrow_fw_normalized_df, tomorrow_iata_df = dataset_normalizer(tomorrow_fw_df)

    # Collect timetable information to visualize
    tomorrow_timetable_attr_df = collect_timetable_attributes(tomorrow_hhMM_df, tomorrow_iata_df, removed_rows)
                            
    return tomorrow_fw_normalized_df, tomorrow_timetable_attr_df


def get_timetable_predictions(project):
    
    import os
    import re
    import json
    import requests
    import hopsworks
    import joblib
    import math
    import pandas as pd
    import numpy as np
    from datetime import datetime

    today_flight_weather_df,    today_timetable_attr_df    = collect_today_flight_weather_info()
    tomorrow_flight_weather_df, tomorrow_timetable_attr_df = collect_tomorrow_flight_weather_info()

    # Download the pre-trained model and load it
    mr        = project.get_model_registry()
    model     = mr.get_model("flight_weather_delay_model", version=1)
    model_dir = model.download()
    model     = joblib.load(model_dir + "/flight_weather_delay_model.pkl")


    # Predict today and tomorrow delays
    X_today         = today_flight_weather_df.drop(columns={'dep_delay'})
    X_tomorrow      = tomorrow_flight_weather_df.drop(columns={'dep_delay'})
    y_pred_today    = model.predict(X_today)
    y_pred_tomorrow = model.predict(X_tomorrow)


    # Add to the flight_weather dataframe the predicated delays
    today_timetable_attr_df['delay']    = y_pred_today.astype(int)
    tomorrow_timetable_attr_df['delay'] = y_pred_tomorrow.astype(int)

    today_timetable_attr_df    = get_timetable_labels(today_timetable_attr_df)
    tomorrow_timetable_attr_df = get_timetable_labels(tomorrow_timetable_attr_df)

    return today_timetable_attr_df, tomorrow_timetable_attr_df


def get_dataframe_padded(dataframe, padding_column_number):
  '''
  Given a dataframe and a padding_measure, return the same dataframe a number of
  new padding column containing only the value "padding_column_number"
  '''

  padding_column = [padding_column_number] * dataframe.shape[0]

  for number in range(padding_column_number):
    numbered_name = 'padding' + str(number)
    dataframe[numbered_name] = padding_column

  return dataframe


def save_timetable_predictions_on_hopsworks(project, today_df, tomorrow_df):
    
    import os
    import re
    import json
    import requests
    import hopsworks
    import joblib
    import math
    import pandas as pd
    import numpy as np
    from datetime import datetime

    # Get access to hopsworks memory
    dataset_api = project.get_dataset_api()

    today_df_padded    = get_dataframe_padded(today_df,    50)
    tomorrow_df_padded = get_dataframe_padded(tomorrow_df, 50)

    with open("today_timetable_prediction.csv", "w") as today_outfile:
        today_df_padded.to_csv(today_outfile, index = False)

        today_pred_path = os.path.abspath("today_timetable_prediction.csv")
        dataset_api.upload(today_pred_path, "Resources/today_timetable_prediction", overwrite=True)
    today_outfile.close()
    os.remove("today_timetable_prediction.csv")

    with open("tomorrow_timetable_prediction.csv", "w") as tomorrow_outfile:
        tomorrow_df_padded.to_csv(tomorrow_outfile, index = False)

        tomorrow_pred_path = os.path.abspath("tomorrow_timetable_prediction.csv")
        dataset_api.upload(tomorrow_pred_path, "Resources/tomorrow_timetable_prediction", overwrite=True)
    tomorrow_outfile.close()
    os.remove("tomorrow_timetable_prediction.csv")

    print('Timetable predictions saved on hopsworks')





def g():
  import os
  import json
  import hopsworks
  import pandas as pd
  
  hopsworks_api_key = os.environ['HOPSWORKS_API_KEY']
  project = hopsworks.login(api_key_value = hopsworks_api_key)

  # Get today and tomorrow timetable predictions
  today_timetable_prediction, tomorrow_timetable_prediction = get_timetable_predictions(project)

  # Save the files on Hopsworks
  save_timetable_predictions_on_hopsworks(project, today_timetable_prediction, tomorrow_timetable_prediction)



