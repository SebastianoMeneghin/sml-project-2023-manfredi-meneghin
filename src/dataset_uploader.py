import os
import json
import pandas as pd
import hopsworks

def dataset_normalizer(dataset_df):
    '''
    Given a dataset with the columns names extracted from the APIs data, return a dataset (dataframe)
    with the name of the columns according to the feature group on Hopsworks
    '''
    dataset_df.rename(columns={'depApIataCode' : 'dep_ap_iata_code', 'depDelay' : 'dep_delay', 'depApTerminal': 'dep_ap_terminal',
                                'depApGate': 'dep_ap_gate', 'arrApIataCode' : 'arr_ap_iata_code', 'airlineIataCode':'airline_iata_code',
                                'flightIataNumber':'flight_iata_number'}, inplace= True)
    
    return dataset_df


def dataset_uploader(project, normalized_dataset_df):
    '''
    Given the Hopsworks project and a normalized dataset, upload it to Hopsworks
    '''
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="flight_weather_delay_model",
        version=1,
        primary_key=['flight_iata_number', 'dep_ap_iata_code', 'arr_ap_iata_code'], 
        description="Daily updated flight-weather info dataset")
    fg.insert(normalized_dataset_df)


hopsworks_api_key = os.environ['HOPSWORKS_API_KEY']
project = hopsworks.login(api_key_value = hopsworks_api_key)

dataset_df            = pd.read_csv('/mnt/c/Developer/University/SML/sml-project-2023-manfredi-meneghin/datasets/join_dataset_smhi_zyla.csv')
normalized_dataset_df = dataset_normalizer(dataset_df)

dataset_uploader(project, normalized_dataset_df)
