import re
import json
import pandas as pd

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
