import requests
import json
from utils import flight_lab_url

headers = {
    "Accept": "application/json",
    "Content-Type": 'application/json',
    "Cache-Control": 'no-cache',
}
access_key = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiI0IiwianRpIjoiMDI1ODUxMTJhYTg2YWU5NWQwYWFmN2RiZDQxYWJmZTc4MjE5OGQyYTMyZWM4Yzg4MmJlYjM1NjMxYTRiYzZiNmUxMDNjMDFiMjQ3NmM4YTEiLCJpYXQiOjE3MDM4NjY4NDUsIm5iZiI6MTcwMzg2Njg0NSwiZXhwIjoxNzM1NDg5MjQ1LCJzdWIiOiIyMjAyOSIsInNjb3BlcyI6W119.s09egb0XKHwS9Ox0OOcFLf9m6PqKb1ziZvP8UsGoBo9ECZMCANpb8ahScix3gGJwHbJBCh41K7vUmxrWa11rDg'


# Some variables can be specified with the request
mode         = 'historical' # here select "historical" or "current"
type         = 'departure'  # only departure-from or arrival-to can be asked in the same airport
code         = 'ARN'    
dep_iataCode = ''           # you can speficy the departure airport if you are requesting arrivals
arr_iataCode = ''           # you can speficy the arrival airport if you are requesting departures
date_from    = '2023-01-01'
date_to      = '2023-01-15'
airline_iata = ''
flight_num   = ''

url = flight_lab_url(mode, type, access_key, code, dep_iataCode, arr_iataCode, date_from, date_to, airline_iata, flight_num)
# Make the API request for FlightLabs API
response = requests.get(url, headers = headers)
responseJson = response.json()

print(responseJson)




        
        
