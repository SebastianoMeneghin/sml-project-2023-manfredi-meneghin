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
response = requests.get("https://api.swedavia.se/flightinfo/v2/ARN/departures/2023-12-02", headers = headers)
flights_swedavia = response.json()
print(flights_swedavia)

# Get the flights from the Json file, then flatten them to get nested-attributes and convert it to a DataFrame
flight_str = flights_swedavia['flights']
flight_df = pd.json_normalize(flight_str)

# Select the wanted data from the DataFrame, creating another_df
#flight_df = flight_df.rename(columns={'departureTime.actualUtc': 'actualDepTime', 'departureTime.scheduledUtc': 'scheduledDepTime', 'locationAndStatus.terminal' : 'terminal', 'diIndicator' : 'flightAreaType', 'airlineOperator.icao' : 'airline', 'locationAndStatus.flightLegStatus' : 'flightStatus', 'flightLegIdentifier.callsign' : 'flightIdentifier', 'flightLegIdentifier.flightDepartureDateUtc' : 'departureDate', 'flightLegIdentifier.departureAirportIcao' : 'departureAirport', 'flightLegIdentifier.arrivalAirportIcao' : 'arrivalAirport'})
#another_df = flight_df[["flightIdentifier", "airline", "departureDate", "flightAreaType", "arrivalAirport", "departureAirport", "terminal", "flightStatus", "scheduledDepTime", "actualDepTime"]]





