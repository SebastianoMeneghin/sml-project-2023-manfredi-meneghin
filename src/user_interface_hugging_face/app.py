import gradio as gr
import pandas as pd
import hopsworks
from datetime import datetime
import requests
import json
import math
import os

hopsworks_today_path    = "Resources/today_timetable_prediction/today_timetable_prediction.csv"
hopsworks_tomorrow_path = "Resources/tomorrow_timetable_prediction/tomorrow_timetable_prediction.csv"

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

    
def get_name_of_cities():
    yyyy, mm, dd = get_today_date()
    yyyy1, mm1, dd1 = one_day_forward(yyyy, mm, dd)

    date_label = get_date_label(yyyy, mm, dd, 'hyphen')
    date_label1 = get_date_label(yyyy1, mm1, dd1, 'hyphen')

    # Create the request_url, then get the subscription key from Swedavia API and set them in the header
    swedavia_url     = 'https://api.swedavia.se/flightinfo/v2/ARN/departures/' + date_label
    swedavia_url1     = 'https://api.swedavia.se/flightinfo/v2/ARN/departures/' + date_label1

    subscription_key = os.environ['SWEDAVIA_API_KEY']
    headers = {
        "Ocp-Apim-Subscription-Key": subscription_key,
        "Accept": "application/json",
        "Content-Type": 'application/json',
    }

    # Make the API request for Swedavia API
    response = requests.get(swedavia_url, headers = headers)
    response1 = requests.get(swedavia_url, headers = headers)
    flights_swedavia = response.json()
    flights_swedavia1 = response1.json()


    # Load JSON data into a Python dictionary
    arrival_airports_info = [{
        'ArrivalAirportIata': flight.get('flightLegIdentifier', {}).get('arrivalAirportIata'),
        'ArrivalAirportEnglish': flight.get('arrivalAirportEnglish')} 
        for flight in flights_swedavia.get('flights', [])]
    df = pd.DataFrame(arrival_airports_info)
    arrival_airports_info1 = [{
        'ArrivalAirportIata': flight.get('flightLegIdentifier', {}).get('arrivalAirportIata'),
        'ArrivalAirportEnglish': flight.get('arrivalAirportEnglish')} 
        for flight in flights_swedavia1.get('flights', [])]
    df1 = pd.DataFrame(arrival_airports_info1)

    total_df = pd.DataFrame({'ArrivalAirportEnglish': pd.concat([df1['ArrivalAirportEnglish'], df['ArrivalAirportEnglish']]).drop_duplicates().reset_index(drop=True),'ArrivalAirportIata': pd.concat([df1['ArrivalAirportIata'], df['ArrivalAirportIata']]).drop_duplicates().reset_index(drop=True)})
    total_df.sort_values('ArrivalAirportEnglish', inplace=True)

    return total_df


def create_single_dataframe_from(dataframe):
    df = get_name_of_cities()
    df['ArrivalAirportIata'] = df['ArrivalAirportIata'].str.lower()
    merged_df = pd.merge(df, dataframe, left_on='ArrivalAirportIata', right_on='airport', how='inner')
    # Drop the duplicate 'ArrivalAirportIata' column
    merged_df = merged_df.drop('ArrivalAirportIata', axis=1)

    return merged_df

def get_dataframe(online_dataframe_path):
    # Connect to Hopsworks File System
    project     = hopsworks.login(api_key_value = os.environ['HOPSWORKS_API_KEY'])
    dataset_api = project.get_dataset_api()

    # Download online dataframe and get path
    dataframe_path = os.path.abspath(dataset_api.download(online_dataframe_path, overwrite = True))

    # Read dataframe from local path, drop duplicates, return
    dataframe = pd.read_csv(dataframe_path)
    dataframe.drop_duplicates(inplace=True)

    dataframe = create_single_dataframe_from(dataframe)
    return dataframe

def get_tomorrow_dataframe():
    return get_dataframe(hopsworks_today_path)

def get_today_dataframe():
    return get_dataframe(hopsworks_tomorrow_path)

def get_metrics():
    # Connect to Hopsworks File System
    dataframe = hopsworks.login(api_key_value = os.environ['HOPSWORKS_API_KEY'])
    dataframe = dataframe.get_feature_store()
    dataframe = dataframe.get_feature_group(name = 'model_performance', version = 1)
    dataframe = dataframe.read(dataframe_type = 'pandas')
    dataframe = dataframe.sort_values('timestamp')
    dataframe = dataframe[['timestamp', 'mae', 'dateset_size']].rename(columns={'dateset_size':'Dataset Size', 'mae':'Mean Absolute Error', 'timestamp':'Date'})
    dataframe = dataframe.sort_values(['Date'], ascending = False)
    return dataframe

selected_columns = ['destination', 'airport code', 'flight number', 'ontime', 'delayed']
ciccio, pasticcio  = pd.DataFrame(), pd.DataFrame()
cities_datafram    = get_name_of_cities()
ciccio             = get_today_dataframe()
ciccio             = ciccio.rename(columns={'airport':'airport code' , 'ArrivalAirportEnglish':'destination', 'flight_number':'flight number'})
today_dataframe    = ciccio[selected_columns]
pasticcio          = get_tomorrow_dataframe()
pasticcio          = pasticcio.rename(columns={'airport':'airport code','ArrivalAirportEnglish':'destination', 'flight_number':'flight number'})
performance_metric = get_metrics()


def get_possible_destinations():
    global today_dataframe, tomorrow_dataframe
    today_df, tomorrow_df = today_dataframe, tomorrow_dataframe
    total_df = pd.DataFrame({'destination': pd.concat([today_df['destination'], tomorrow_df['destination']]).drop_duplicates().reset_index(drop=True).sort_values()})
    total_dest = (total_df['destination']).tolist()
    return total_dest

def get_dataframe_of(day):
    global cities_datafram, today_dataframe, tomorrow_dataframe
    if   (day.lower() == 'today'):
        return today_dataframe
    elif (day.lower() == 'tomorrow'):
        return tomorrow_dataframe

def get_specific_flights(day, max_delay, departure_hour, ampm, weather, destinations, yes):
    df = get_dataframe_of(day)

    if ('Select all' in destinations):
        destinations = get_possible_destinations()
    
    # Remove unwanted destinations
    destinations = [dest for dest in destinations if dest not in ["That's a reason why I travel alone...", "I prefer not to say", 'Select all']]

    # Select only flight during the same departure hour
    df['departure_hour'] = df['ontime'].str.split(':').str[0].astype(int)
    df = df[df['departure_hour'] == departure_hour].drop(columns=['departure_hour'])

    # Convert time columns to datetime objects
    df['ontime'] = pd.to_datetime(df['ontime'], format='%H:%M')
    df['delayed'] = pd.to_datetime(df['delayed'], format='%H:%M')

    # Get flight with less delay than the given and from the destinations selected, of the right day
    df['delay'] = (df['delayed'] - df['ontime']).dt.total_seconds() / 60
    filtered_df = df.loc[(df['delay'] < max_delay) & (df['destination'].isin(destinations)), ['destination', 'flight number', 'ontime', 'delayed']]
 
    # Convert the string to datetime, then the datetime column to HH:MM
    filtered_df['ontime'] = pd.to_datetime(filtered_df['ontime'])
    filtered_df['ontime'] = filtered_df['ontime'].dt.strftime('%H:%M')
    filtered_df['delayed'] = pd.to_datetime(filtered_df['delayed'])
    filtered_df['delayed'] = filtered_df['delayed'].dt.strftime('%H:%M')
    
    return filtered_df

def full_day_departure(day):
    dataframe = get_dataframe_of(day)
    dataframe.drop(columns=['airport code'], inplace=True)
    return dataframe

def get_performance():
    global performance_metric
    return performance_metric

specific_flights = gr.Interface(
    get_specific_flights,
    [
        gr.Radio(["today", "tomorrow"], type="value", label="Day", info="When do you have the plane?"),
        gr.Slider(0, 50, value=20, label="Possible Delay", info="How unfortunate do you wanna be?"),
        gr.Number(precision=0, minimum=0, maximum=23, label="Departure Time"),
        gr.Radio(["am", "pm"], type="index", info="It's the same, no worries!", label = "Am or Pm?"),
        gr.CheckboxGroup(["Yes, it's cloudy", "I am not in Stockholm"], label="Weather", info="Is it a typical Stockholm day?"),
        gr.Dropdown(get_possible_destinations() + ["That's a reason why I travel alone...", "I prefer not to say", "Select all"], 
                    type = "value", multiselect=True, label="Destination", value=["That's a reason why I travel alone..."],
                    info="Are you just curious or you are actually going somewhere? Where? With who?"),
        gr.Radio(["Yes", "Yes", "Yes"], type="index", label="Let's guess?", info="We know that you'll say yes!"),
    ],
    "dataframe",
)

total_departure = gr.Interface(
    full_day_departure,
    [
        gr.Radio(["Today", "Tomorrow"], type="value", label="Departure", info="When are you departing?"),
    ],
    "dataframe",
)

metrics = gr.Interface(fn = get_performance, inputs=None, outputs='dataframe', allow_flagging="never")

#flights.launch()

interface = gr.TabbedInterface([specific_flights, total_departure, metrics], {"Full Day Departure", "Specific Flights", "Model Performances"})
interface.launch()