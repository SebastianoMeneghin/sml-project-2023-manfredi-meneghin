import re
import os
import requests
import json
import pandas as pd
import numpy as np
import math
from datetime import datetime

# Global values used by different functions
ZRAD = math.pi / 180
ZRADI = 1 / ZRAD


# Function to devide the different parts of datatime
def get_data(x):
    year   = '' 
    month  = ''
    day    = ''
    hour   = 0
    minute = 0
    second = 0
    datetime_pattern = re.compile(r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z')
    match = datetime_pattern.match(x)

    if match:
      year, month, day, hour, minute, second = map(str, match.groups())
    
    my_date = year + '-' + month + '-' + day
    return (my_date)


# Function to get the month from the datatime
def get_month(x):
    month = 0
    datetime_pattern = re.compile(r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z')
    match = datetime_pattern.match(x)

    if match:
      year, month, day, hour, minute, second = map(int, match.groups())
    
    my_month = month
    return (my_month)


# Create the url for flightLabsAPI requests
def flight_lab_url(mode, type, access_key, code, dep_iataCode, arr_iataCode, date_from, date_to, airline_iata, flight_num): 
    if access_key == '':
        raise Exception("You must indicate your access_key for each request!")
    
    initial_url = "https://goflightlabs.com/"

    if mode == "historical":
        if type == '' or code == '':
            raise Exception("You must indicate both a type (dep or arr) and a code (airport code) for each historical request!")
        
        if date_from == '': 
            raise Exception('You must select a date_from if you are making an historical request!')
        
        initial_url = initial_url + 'historical/' + date_from + "?access_key=" + access_key + "&code=" + code + "&type=" + type

        if dep_iataCode != '' and type == 'arrival':
            initial_url = initial_url + '&dep_iataCode=' + dep_iataCode

        if arr_iataCode != '' and type == 'departure':
            initial_url = initial_url + '&arr_iataCode=' + arr_iataCode
        
        if date_to != '':
            initial_url = initial_url + '&date_to=' + date_to

        if airline_iata != '':
            initial_url = initial_url + '&airline_iata=' + airline_iata
        
        if flight_num != '':
            initial_url = initial_url + '&flight_num=' + flight_num
    
    # To be defined
    # if mode == 'current':

    return initial_url


# Create the url for flightLabsAPI requests
def zylaAPI_url(type, code, dep_iataCode, arr_iataCode, date_from, date_to, airline_iata, flight_num): 
    
    initial_url = "https://zylalabs.com/api/1020/historical+flights+information+api/890/historical?"

    if type == '' or code == '':
        raise Exception("You must indicate both a type (dep or arr) and a code (airport code) for each historical request!")
    
    if date_from == '': 
        raise Exception('You must select a date_from if you are making an historical request!')
    
    initial_url = initial_url + "code=" + code + "&type=" + type + "&date=" + date_from

    if dep_iataCode != '' and type == 'arrival':
        initial_url = initial_url + '&dep_iataCode=' + dep_iataCode

    if arr_iataCode != '' and type == 'departure':
        initial_url = initial_url + '&arr_iataCode=' + arr_iataCode
    
    if date_to != '':
        initial_url = initial_url + '&date_to=' + date_to

    if airline_iata != '':
        initial_url = initial_url + '&airline_iata=' + airline_iata
    
    if flight_num != '':
        initial_url = initial_url + '&flight_num=' + flight_num

    return initial_url


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



# Class used to work with GRIB Files
class Point:
  def __init__(self, lat, long):
    self.lat = lat
    self.lon = long
  
  def getLat(self):
    return self.lat
  
  def getLon(self):
    return self.lon
  
  def setLat(self, x):
     self.lat = x
  
  def setLon(self, x):
    self.lon = x


# Rotate a classical lon/lat point to a point rotated according to SMHI GRIB Files
def regularToRotatedPoint(regularPoint, polePoint):

  rotatedPoint = Point(0.0, 0.0)

  zsycen = math.sin(ZRAD * (polePoint.lat + 90.0));
  zcycen = math.cos(ZRAD * (polePoint.lat + 90.0));
  zxmxc = ZRAD * (regularPoint.lon - polePoint.lon);
  zsxmxc = math.sin(zxmxc);
  zcxmxc = math.cos(zxmxc);
  zsyreg = math.sin(ZRAD * regularPoint.lat);
  zcyreg = math.cos(ZRAD * regularPoint.lat);
  zsyrot = zcycen * zsyreg - zsycen * zcyreg * zcxmxc;
  zsyrot = math.max(zsyrot, -1.0);
  zsyrot = math.min(zsyrot, +1.0);

  rotatedPoint.setLat(math.asin(zsyrot) * ZRADI)

  zcyrot = math.cos(rotatedPoint.lat * ZRAD);
  zcxrot = (zcycen * zcyreg * zcxmxc + zsycen * zsyreg) / zcyrot;
  zcxrot = math.max(zcxrot, -1.0);
  zcxrot = math.min(zcxrot, +1.0);
  zsxrot = zcyreg * zsxmxc / zcyrot;

  rotatedPoint.setLon(math.acos(zcxrot) * ZRADI);

  if (zsxrot < 0.0):
    rotatedPoint.setLon(-1.0 * rotatedPoint.lon)

  return rotatedPoint


# Translate column label found in GRIB files with compatible dataframe column's label.
def get_df_label_from_grib_label(grib_label):
  df_label = ''

  if grib_label == 'Temperature':
    df_label = 'temperature'
  elif grib_label == 'Wind gusts':
    df_label = 'gusts_wind'
  elif grib_label == 'u-component of wind':
    df_label = 'u_wind'
  elif grib_label == 'v-component of wind':
    df_label = 'v_wind'
  elif grib_label == 'Relative humidity':
    df_label = 'humidity'
  elif grib_label == '1 hour precipitation':
    df_label = 'prep_1h'
  elif grib_label == '1 hour fresh snow cover':
    df_label = 'snow_1h'
  elif grib_label == 'Visibility':
    df_label = 'visibility'
  elif grib_label == 'Pressure reduced to MSL':
    df_label = 'pressure'
  elif grib_label == 'Total cloud cover':
    df_label = 'total_cloud'
  elif grib_label == 'Low cloud cover':
    df_label = 'low_cloud'
  elif grib_label == 'Medium cloud cove':
    df_label = 'medium_cloud'
  elif grib_label == 'High cloud cover':
    df_label = 'high_cloud'
  elif grib_label == 'Type of precipitation':
    df_label = 'type_prep'
  elif grib_label == 'Sort of precipitation':
    df_label = 'sort_prep'
  elif grib_label == 'Snowfall (convective + stratiform) gradient':
    df_label = 'gradient_snow'

  return df_label


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


def select_zylaAPI_flight_infos(zylaAPI_file_path):
    '''
    From an input zylaAPI's json file containing all the flights of a day,
    extract the flight_info for each flight and savem them in a list, with
    a flat format.
    '''
    with open(zylaAPI_file_path, 'r') as file:
        full_day_data = json.load(file)

    flight_infos = []

    if (full_day_data.get("data") != None):
        for flight in full_day_data['data']:
            
            # If depTerminal or depGate are not present in the data, set them to 0
            flight_info = {
                "status": flight["status"],
                "depApIataCode" : flight["departure"]["iataCode"],
                "depDelay" : flight["departure"].get("delay", 0),
                "depScheduledTime": flight["departure"]["scheduledTime"],
                "depApTerminal": flight["departure"].get("terminal", 0),
                "depApGate": flight["departure"].get("gate", '0'),
                "arrScheduledTime": flight["arrival"]["scheduledTime"],
                "arrApIataCode": flight["arrival"]["iataCode"],
                "airlineIataCode": flight["airline"].get("iataCode", '0'),
                "flightIataNumber": flight["flight"].get("iataNumber", '0'),
            }

            if flight_info["airlineIataCode"] == "": 
                flight_info["airlineIataCode"] = '0'
            if flight_info["flightIataNumber"] == "": 
                flight_info["flightIataNumber"] = '0'

            flight_infos.append(flight_info)
    
    return flight_infos


def merge_and_extract_zylaAPI_flight_infos(directory_path):
    '''
    From an input director full of zylaAPI's json file, each of them containing all the 
    flights of a specific day, from a specific airport (in this project Stockholm Arlanda,
    IATACode: ARN), get the list of all the flights of all the day with the structure specified
    in function "select_zylaAPI_flight_infos".
    '''
    flight_infos = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            flight_info = select_zylaAPI_flight_infos(file_path)
            flight_infos = flight_infos + flight_info

    return flight_infos
  

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


def zylaAPI_num_flight_within(interval_min, flight_df):
    '''
    Return for each row of the dataframe, the number of flights departed from the same airport
    in the interval specified in minutes (-interval_min/2, + interval_min/2).
    This works with zylaAPI flights info, having datetime_format "%Y-%m-%dT%H:%M:%S.%f"
    '''
    # Row range should be reduced or increased depending on the airport
    row_range = 50
    total_flight  = flight_df.shape[0]
    flight_within = []

    for row in range(total_flight):
        flight_counter = 0
        temp_df = pd.DataFrame()

        # Get departure time of the flight
        flight_dep_time = flight_df.at[row, 'depScheduledTime']
        flight_datetime = datetime.strptime(flight_dep_time, "%Y-%m-%dT%H:%M:%S.%f")

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
            temp_datetime    = datetime.strptime(flight_temp_time, "%Y-%m-%dT%H:%M:%S.%f")

            # Get the delta time and add bring it to minutes
            delta_time = abs(flight_datetime - temp_datetime)
            gap_time   = math.ceil(delta_time.total_seconds()/60)

            if (gap_time < interval_min/2):
                flight_counter += 1
        
        flight_within.append(flight_counter)
    

    column_name = 'flight_within_' + str(interval_min) + 'min'
    return flight_within, column_name


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


def set_model_last_version_number(project, last_version_number):
    '''
    Given the Hopsworks Project "project", set the latest version number of the dataset to "version_number"
    '''
    # Get dataset API to save the version number on Hopsworks
    dataset_api = project.get_dataset_api()

    # Create the skeleton of a .json file big enough to be saved by Hopsworks
    last_version_number = {'last_version_number': last_version_number}
    version_number_list   = [last_version_number] * 1000
    last_version_number_json = json.dumps(version_number_list)

    # Create a file .json, save it in local and then save it on hopsworks. When finished, delete the json file locally
    with open("last_version_number.json", "w") as outfile:
        outfile.write(last_version_number_json)

        last_version_number_path = os.path.abspath('last_version_number.json')
        dataset_api.upload(last_version_number_path, "Resources/dataset_version", overwrite=True)
    os.remove("last_version_number.json")
    
    
def get_model_last_version_number(project):
    '''
    Given the Hopsworks Project "project", get the last_version_number of the dataset
    '''
    # Get dataset API to download the last_version_number from Hopsworks
    dataset_api = project.get_dataset_api()
    dataset_api.download("Resources/dataset_version/last_version_number.json")

    # Open JSON file, return it as a dictionary
    json_file = open('last_version_number.json')
    json_data = json.load(json_file)

    last_version_number = json_data[0]['last_version_number']

    return last_version_number


def dataset_normalizer(dataset_df):
    '''
    Given a dataset with the columns names extracted from the APIs data, return a dataset (dataframe)
    with the name of the columns according to the feature group on Hopsworks
    '''
    dataset_df.rename(columns={'depApIataCode' : 'dep_ap_iata_code', 'depDelay' : 'dep_delay', 'depApTerminal': 'dep_ap_terminal',
                                'depApGate': 'dep_ap_gate', 'arrApIataCode' : 'arr_ap_iata_code', 'airlineIataCode':'airline_iata_code',
                                'flightIataNumber':'flight_iata_number'}, inplace= True)
    
    return dataset_df

