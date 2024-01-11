import requests
import json
from utils import zylaAPI_url


headers = {
    'Authorization': 'Bearer 3109|eaLTjs0WyoNM4J5rV2VzkHvVH1k1zd75X3GLV92Q',
    "Cache-Control": 'no-cache',
}

# Some variables can be specified with the request
type         = 'departure'  # only departure-from or arrival-to can be asked in the same airport
code         = 'ARN'    
dep_iataCode = ''           # you can speficy the departure airport if you are requesting arrivals
arr_iataCode = ''           # you can speficy the arrival airport if you are requesting departures
date_from    = '2024-01-02'
date_to      = '2024-01-02'           # APPARENTLY THIS IS NOT WORKING
airline_iata = ''
flight_num   = ''

url = zylaAPI_url(type, code, dep_iataCode, arr_iataCode, date_from, date_to, airline_iata, flight_num)
print(url)

# Make the API request for Zyla API
response = requests.get(url, headers=headers)
responseJson = response.json()
