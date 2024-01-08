import os
from sklearn import metrics
from sklearn.metrics import get_scorer_names
import pandas as pd
import pandasql as sqldf
import hopsworks
import time
import json
from hsml.client.exceptions import RestAPIError


def set_model_last_version_number(project, last_version_number):
    '''
    Given the Hopsworks Project "project", set the latest version number of the dataset to "version_number"
    '''
    # Get dataset API to save the version number on Hopsworks
    dataset_api = project.get_dataset_api()

    # Create the skeleton of a .json file big enough to be saved by Hopsworks
    last_version_number = {'last_version_number': last_version_number}
    version_number_list   = [last_version_number] * 1000
    last_version_number_json = json.dumps(version_number_list)

    # Create a file .json, save it in local and then save it on hopsworks. When finished, delete the json file locally
    with open("last_version_number.json", "w") as outfile:
        outfile.write(last_version_number_json)

        last_version_number_path = os.path.abspath('last_version_number.json')
        dataset_api.upload(last_version_number_path, "Resources/dataset_version", overwrite=True)
    os.remove("last_version_number.json")

    
def get_model_last_version_number(project):
    '''
    Given the Hopsworks Project "project", get the last_version_number of the dataset
    '''
    # Get dataset API to download the last_version_number from Hopsworks
    dataset_api = project.get_dataset_api()
    dataset_api.download("Resources/dataset_version/last_version_number.json")

    # Open JSON file, return it as a dictionary
    json_file = open('last_version_number.json')
    json_data = json.load(json_file)

    last_version_number = json_data[0]['last_version_number']
    os.remove('last_version_number.json')

    return last_version_number


def get_model_last_version(project):
    last_version_number = get_model_last_version_number(project)

    mr    = project.get_model_registry()
    model = mr.get_model('flight_weather_delay', version = last_version_number)

    return model





###### Body of function ######

hopsworks_api_key = os.environ['HOPSWORKS_API_KEY']
project = hopsworks.login(api_key_value = hopsworks_api_key)

set_model_last_version_number(project, 0)
time.sleep(5)
print(get_model_last_version_number(project))










