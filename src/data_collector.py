import requests
import pandas as pd
import json
import re

# Get some data from the SHMI OpenData API
# response2 = requests.get("https://opendata-download-metanalys.smhi.se/api/category/mesan1g/version/2/geotype/point/lon/17.893379/lat/59.597679/data.json")
# responseJson2 = response2.json()
# print(responseJson2)


# Get the subscription key from Swedavia API and set up it as header for the requests
subscription_key = 'a9042dc249f34e02b9d7512a1d85aa70'
headers = {
    "Ocp-Apim-Subscription-Key": subscription_key,
    "Accept": "application/json",
    "Content-Type": 'application/json',
}

# Make the API request for Swedavia API
response3 = requests.get("https://api.swedavia.se/flightinfo/v2/ARN/departures/2023-12-02", headers = headers)
responseJson3 = response3.json()
#print(responseJson3)

# Get the flights from the Json file, then flatten them to get nested-attributes and convert it to a DataFrame
flight_str = responseJson3['flights']
flight_df = pd.json_normalize(flight_str)

# Select the wanted data from the DataFrame, creating another_df
flight_df = flight_df.rename(columns={'departureTime.actualUtc': 'actualDepTime', 'departureTime.scheduledUtc': 'scheduledDepTime', 'locationAndStatus.terminal' : 'terminal', 'diIndicator' : 'flightAreaType', 'airlineOperator.icao' : 'airline', 'locationAndStatus.flightLegStatus' : 'flightStatus', 'flightLegIdentifier.callsign' : 'flightIdentifier', 'flightLegIdentifier.flightDepartureDateUtc' : 'departureDate', 'flightLegIdentifier.departureAirportIcao' : 'departureAirport', 'flightLegIdentifier.arrivalAirportIcao' : 'arrivalAirport'})
another_df = flight_df[["flightIdentifier", "airline", "departureDate", "flightAreaType", "arrivalAirport", "departureAirport", "terminal", "flightStatus", "scheduledDepTime", "actualDepTime"]]

# Get the month from the datatime (as future input for the ML algorithm) and the data, as key for the Database
data_key_column = another_df['departureDate'].map(lambda x : get_data(x))
month_column    = another_df['departureDate'].map(lambda x : get_month(x))

dep_df = another_df
dep_df['departureDate'] = data_key_column
dep_df['month']         = month_column


dep_df.T







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

