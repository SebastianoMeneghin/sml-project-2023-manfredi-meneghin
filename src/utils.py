import re
import json
import pandas as pd
import math

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


# Get the date/timestamp in the format wanted by the SMHI Historical API file structure, by passing 
# the year, month, day and hour. It pads with 0 when needed.
def get_mesan_date_label(year, month, day, hour, mode):
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