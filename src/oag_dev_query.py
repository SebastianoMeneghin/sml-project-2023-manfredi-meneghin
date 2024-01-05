import requests
import pandas as pd
import json
import re
from utils import get_data, get_month

subscription_key = '6433087898dd4bc8a814883c1a1f53e9'
headers = {
    "Subscription-Key": subscription_key,
    "Accept": "application/json",
    "Content-Type": 'application/json',
    "Cache-Control": 'no-cache',
}


# Make the API request for Swedavia API
response = requests.get("https://api.oag.com/flight-instances/?version=v2", headers = headers)
responseJson = response.json()

try:
    url = "https://api.oag.com/flight-instances/?version=v2"

    hdr ={
    # Request headers
    'Cache-Control': 'no-cache',
    'Subscription-Key': '••••••••••••••••••••••••••••••••',
    }

    req = urllib.request.Request(url, headers=hdr)

    req.get_method = lambda: 'GET'
    response = urllib.request.urlopen(req)
    print(response.getcode())
    print(response.read())
except Exception as e:
    print(e)