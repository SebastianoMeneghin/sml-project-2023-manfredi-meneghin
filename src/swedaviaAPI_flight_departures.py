import os
import pandas as pd
import numpy as np
import json
import requests
import math
import re
from datetime import datetime
from io import StringIO
from utils import one_hour_backward, one_hour_forward, get_date_label, one_day_forward, one_day_backward

def get_today_date():
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
    yyyy, mm, dd = get_today_date()

    if (mode == 'yesterday'):
        yyyy, mm, dd = one_day_backward(yyyy, mm, dd)

    elif (mode == 'tomorrow'):
        yyyy, mm, dd = one_day_forward(yyyy, mm, dd)

    date_label = get_date_label(yyyy, mm, dd, 'hyphen')

    # Create the request_url, then get the subscription key from Swedavia API and set them in the header
    swedavia_url     = 'https://api.swedavia.se/flightinfo/v2/ARN/departures/' + date_label
    subscription_key = 'a9042dc249f34e02b9d7512a1d85aa70'
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
    '''
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

    # Save the new dataframe in a new file (.csv)
    ts_path = "/mnt/c/Developer/University/SML/sml-project-2023-manfredi-meneghin/datasets/swedaviaAPI_flights/"
    ts_name = mode + '_flight_data_' + json_date + '.csv'
    ts_complete_path = os.path.join(ts_path, ts_name)

    with open(ts_complete_path, "wb") as df_out:
        df.to_csv(df_out, index= False)
    df_out.close()




result, selected_date = swedaviaAPI_daily_collector('yesterday')
swedaviaAPI_flight_processor(result, selected_date, 'historical')



