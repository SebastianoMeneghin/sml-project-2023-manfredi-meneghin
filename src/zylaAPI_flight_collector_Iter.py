import requests
import json
import os
from utils import zylaAPI_url, get_date_label
        
headers = {
    'Authorization': 'Bearer 3109|eaLTjs0WyoNM4J5rV2VzkHvVH1k1zd75X3GLV92Q',
    "Cache-Control": 'no-cache',
}

# Some variables can be specified with the request
type         = 'departure'  # only departure-from or arrival-to can be asked in the same airport
code         = 'ARN'    
dep_iataCode = ''           # you can speficy the departure airport if you are requesting arrivals
arr_iataCode = ''           # you can speficy the arrival airport if you are requesting departures
date_from    = ''           # this value will be updated in the following code
date_to      = ''           # this value will be updated in the following code
airline_iata = ''
flight_num   = ''
    
# Insert the limit of your wanted range here:
# 1 is Jan, 2 is Feb, ...
year           = 2023
starting_month = 1
ending_month   = 12
starting_day   = 1
ending_day     = 31

for month in range(starting_month, ending_month + 1):
    for day in range(starting_day, ending_day + 1):
        compile = True
        if (month in {1} and day in {1,2,3,4,5,6}) or (month in {4, 6, 9, 11} and day in {31}) or (month in {2} and day in {29, 30, 31}):
            compile = False

        if compile == True:
            date_label = get_date_label(year, month, day, 'hyphen')
            print(date_label)

            date_from = date_label
            date_to   = date_label

            url = zylaAPI_url(type, code, dep_iataCode, arr_iataCode, date_from, date_to, airline_iata, flight_num)
            print(url)

            # Make the API request for Zyla API
            response = requests.get(url, headers=headers)
            responseJson = response.json()

            file_name = "flight_" + date_label + ".json"
            save_path = "/mnt/c/Developer/University/SML/sml-project-2023-manfredi-meneghin/datasets/zylaAPI_flights/"
            complete_name = os.path.join(save_path, file_name)

            with open(complete_name, "w") as outfile:
                json.dump(responseJson, outfile)
            outfile.close()
