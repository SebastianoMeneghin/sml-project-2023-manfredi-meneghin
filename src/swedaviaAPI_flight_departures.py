import requests
import pandas as pd
import json
from utils import get_data, get_month

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


print(dep_df.head().T)



### For OAG API
########### Python 3.2 #############
import urllib.request, json

try:
    url = "https://api.oag.com/flight-instances/?DepartureDateTime=2022-12-04&DepartureAirport=ESSA&Content=Status&CodeType=ICAO&Limit=100&version=v2"

    hdr ={
    # Request headers
    'Cache-Control': 'no-cache',
    'Subscription-Key': '1ba6f1f8d11143b698bb576269232c52',
    }

    req = urllib.request.Request(url, headers=hdr)

    req.get_method = lambda: 'GET'
    response = urllib.request.urlopen(req)
    print(response.getcode())
    print(response.read())
except Exception as e:
    print(e)
####################################
