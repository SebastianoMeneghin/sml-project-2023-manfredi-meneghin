import os
import pandas as pd
import numpy as np
import json
import requests
import math
import re
from datetime import datetime
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

        # If the flight info contains also the delay:
        if (flight.get("departureTime").get("actualUtc") != None):

            # If depTerminal or depGate are not present in the data, set them to 0
            # If the arrScheduledTime is not present, set it to the depScheduledTime
            flight_info = {
                "status"           : str(flight["locationAndStatus"]["flightLegStatusEnglish"]).lower(),
                "depApIataCode"    : str(flight["flightLegIdentifier"]["departureAirportIata"]).lower(),
                "depDelay"         : swedaviaAPI_flight_delay(flight["departureTime"]["scheduledUtc"], flight["departureTime"]["actualUtc"]),
                "depScheduledTime" : swedaviaAPI_correct_UCT(flight["departureTime"]["scheduledUtc"]),
                "depApTerminal"    : int(re.findall(terminal_pattern, flight["locationAndStatus"].get("terminal", 0))[0]),
                "depApGate"        : str(flight["locationAndStatus"].get("gate", 0)).lower(),
                "arrScheduledTime" : swedaviaAPI_correct_UCT(flight.get("arrivalTime", flight["departureTime"]["scheduledUtc"])),
                "arrApIataCode"    : str(flight["flightLegIdentifier"]["arrivalAirportIata"]).lower(),
                "airlineIataCode"  : str(flight["airlineOperator"].get("iata", '0')).lower(),
                "flightIataNumber" : str(flight.get("flightId", '0')).lower(),
            }
            
            flight_infos.append(flight_info)



    # Create then a new file and save there the flight_infos from all the files of the directory
    file_name     = 'swedaviaAPI_' + date_label + '.json'
    file_path     = '/mnt/c/Developer/University/SML/sml-project-2023-manfredi-meneghin/datasets/swedaviaAPI_flights/'
    complete_path = os.path.join(file_path, file_name)

    with open(complete_path, 'w') as output_file:
        print(json.dumps(flight_infos, indent=2), file=output_file)
    output_file.close()


swedaviaAPI_daily_collector('yesterday')



