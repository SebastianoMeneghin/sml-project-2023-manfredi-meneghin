import json
import os
import joblib
import pandas as pd
import numpy as np
import xgboost
import hopsworks
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error


##### PARAMETERS SETTING #####
# Set true if you want:
MODEL_SELECTION  = True  #perform model selection
MODEL_EVALUATION = True  #perform model evaluation


def uniform_dataframe_for_training(df):
    
    '''
    Given a dataset with the columns names extracted from the APIs data, return a dataset (dataframe)
    uniformed in order to be possible training a model on that
    '''
    
    '''
    df.rename(columns ={'status': 'status', 'depApIataCode': 'dep_ap_iata_code', 'depDelay': 'dep_delay', 'depApTerminal': 'dep_ap_terminal', 
                        'depApGate': 'dep_ap_gate', 'arrApIataCode': 'arr_ap_iata_code', 'airlineIataCode': 'airline_iata_code', 
                        'flightIataNumber': 'flight_iata_number', 'flight_within_60min': 'flight_within_60min', 'date': 'date', 
                        'time': 'time', 'month': 'month', 'trip_time': 'trip_time', 'day_of_week': 'day_of_week', 'temperature': 'temperature', 
                        'visibility': 'visibility', 'pressure': 'pressure', 'humidity': 'humidity', 'gusts_wind': 'gusts_wind', 
                        'wind_speed': 'wind_speed', 'wind_dir': 'wind_dir', 'total_cloud': 'total_cloud', 'low_cloud': 'low_cloud', 
                        'medium_cloud': 'medium_cloud', 'high_cloud': 'high_cloud', 'sort_prep': 'sort_prep'}, inplace=True)
    '''
    df.drop(  columns ={'trip_time', 'dep_ap_gate', 'flight_iata_number', 'status', 'dep_ap_terminal', 'date', 'high_cloud', 
                        'medium_cloud', 'total_cloud', 'wind_speed','pressure', 'dep_ap_iata_code', 'humidity', 'flight_within_60min'}, inplace=True)

    all_the_columns  =['airline_iata_code', 'arr_ap_iata_code', 'day_of_week', 'dep_delay', 'gusts_wind', 'low_cloud', 'month', 'sort_prep',
                       'temperature', 'time', 'visibility', 'wind_dir']
    
    # Remove outliners in delay (dep_delay > 80)
    row_list = []
    for row in range(df.shape[0]):
        if (df.at[row, 'dep_delay'] > 80 - (80-53)*2/3):
            row_list.append(row)
    df.drop(row_list, inplace = True)
    df.reset_index(inplace = True)
    df.drop(columns={'index'}, inplace = True)
    
    # Transform Wind Direction into three dummy variables: wind from SE; wind from NE, other wind
    new_columns_wind = ['se_wind','ne_wind','other_wind']
    dir_dict = {'SW':'other_wind','S':'other_wind','SE':'se_wind','E':'ne_wind','NE':'ne_wind','N':'ne_wind','NW':'other_wind','W':'other_wind'}
    direction_list = []
    for row in range(df.shape[0]):
        new_row = []
        direction = df.at[row, 'wind_dir']
        string = dir_dict.get(direction)

        if (string == 'se_wind'):
            new_row = [1,0,0]
        elif (string == 'ne_wind'):
            new_row = [0,1,0]
        else:
            new_row = [0,0,1]
        direction_list.append(new_row)
    df.drop(columns={'wind_dir'}, inplace = True)
    df[new_columns_wind] = direction_list

    # Transform visibility into two dummy variables: low_visibility and normal_visibility
    new_columns_vis = ['low_visibility','normal_visibility']
    first_octile_vis = np.percentile(df['visibility'], 12.5)
    visibility_label = []
    for row in range(df.shape[0]):
        new_row = []
        visibility = df.at[row, 'visibility']

        if (visibility <= first_octile_vis):
            new_row = [1,0]
        else:
            new_row = [0,1]
        visibility_label.append(new_row)
    df.drop(columns={'visibility'}, inplace = True)
    df[new_columns_vis] = visibility_label

    # Transform temperature into three dummy variables: below_zero, normal_temperature, over_twenty_degree
    new_columns_temp = ['below_zero', 'normal_temperature', 'over_twenty_degree']
    temperature_label = []
    for row in range(df.shape[0]):
        new_row = []
        temperature = df.at[row, 'temperature']
        if (temperature <= 0):
            new_row = [1,0,0]
        elif (temperature > 20):
            new_row = [0,0,1]
        else:
            new_row = [0,1,0]
        temperature_label.append(new_row)
    #df.drop(columns={'temperature'}, inplace = True)
    df[new_columns_temp] = temperature_label

    # Transform wind gusts speed into two dummy variables: normal_wind and strong_wind
    new_columns_vis = ['normal_wind','strong_wind']
    last_octile = np.percentile(df['gusts_wind'], 85)
    gusts_wind_label = []
    for row in range(df.shape[0]):
        new_row = []
        gusts_wind = df.at[row, 'gusts_wind']
        if (gusts_wind <= last_octile):
            new_row = [1,0]
        else:
            new_row = [0,1]
        gusts_wind_label.append(new_row)
    df.drop(columns={'gusts_wind'}, inplace = True)
    df[new_columns_vis] = gusts_wind_label

    # Reduce sort_prep number of categories, from 6 to 2: 'snow' and 'rain'
    new_columns_vis = ['snow','rain']
    weather_label = []
    for row in range(df.shape[0]):
        new_row = []
        condition = df.at[row, 'sort_prep']
        if (condition == 1):
            new_row = [1,0]
        elif(condition == 3):
            new_row = [0,1]
        else:
            new_row = [0,0]
        weather_label.append(new_row)
    df.drop(columns={'sort_prep'}, inplace = True)
    df[new_columns_vis] = weather_label

    new_columns_months = ['up_months', 'down_months', 'other_months']
    month_labels = []
    for row in range(df.shape[0]):
        new_row = []
        month = df.at[row, 'month']
        if (month in {1,7,12}):
            new_row = [1,0,0]
        elif(month in {2,3,4,5,10}):
            new_row = [0,1,0]
        else:
            new_row = [0,0,1]
        month_labels.append(new_row)
    #df.drop(columns={'month'}, inplace = True)
    df[new_columns_months] = month_labels

    # Get few data from arr_ap_iata_code and airline_iata_code
    new_columns_flights = ['local_flight', 'outer_flight']
    flight_labels = []
    for row in range(df.shape[0]):
        new_row = []
        operator = df.at[row, 'arr_ap_iata_code']
        destination = df.at[row, 'airline_iata_code']
        if ((operator in {'hp','n9'}) or (destination in {'lla','krf','mhq','oer'})):
            new_row = [1,0]
        else:
            new_row = [0,1]
        flight_labels.append(new_row)
    df.drop(columns={'arr_ap_iata_code','airline_iata_code'}, inplace = True)
    df[new_columns_flights] = flight_labels

    # Move from 'day_of_week' to three dummy variables: mid_week, lord_day, extreme_week
    new_columns_day = ['mid_week', 'lord_day', 'extreme_week']
    day_labels = []
    for row in range(df.shape[0]):
        new_row = []
        day = df.at[row, 'day_of_week']
        if (day in {2,3,4}):
            new_row = [1,0,0]
        elif (day in {7}):
            new_row = [0,1,0]
        else:
            new_row = [0,0,1]
        day_labels.append(new_row)
    df.drop(columns={'day_of_week'}, inplace = True)
    df[new_columns_day] = day_labels

    # Save weather or not there is a bright blue sky or a sky full of clouds: perfect_sky, stockholm_sky
    new_cloud_columns = ['perfect_sky', 'stockholm_sky']
    cloud_labels = []
    for row in range(df.shape[0]):
        new_row = []
        cloud_level = df.at[row, 'low_cloud']
        if (cloud_level == 0):
            new_row = [1,0]
        elif (cloud_level == 8):
            new_row = [0,1]
        else:
            new_row = [0,0]
        cloud_labels.append(new_row)
    df.drop(columns={'low_cloud'}, inplace = True)
    df[new_cloud_columns] = cloud_labels

    # Get five label for the time of the day (hour):
    new_hour_columns = ['night','morning','late_morning','afternoon','evening']
    hour_labels = []
    for row in range(df.shape[0]):
        new_row = []
        hour_level = df.at[row, 'time']
        if (hour_level in {0,1,2,3,4,5,21,22,23}):
            new_row = [1,0,0,0,0]
        elif (hour_level in {6,7,8,9,10}):
            new_row = [0,1,0,0,0]
        elif (hour_level in {11,12,13,14}):
            new_row = [0,0,1,0,0]
        elif (hour_level in {15,16,17}):
            new_row = [0,0,0,1,0]
        else:
            new_row = [0,0,0,0,1]
        hour_labels.append(new_row)
    df.drop(columns={'time'}, inplace = True)
    df[new_hour_columns] = hour_labels

    # Convert all the data in int64, delete some of the feature which are not needed
    df = df.astype('int32')
    df.drop(columns ={'normal_visibility','late_morning','stockholm_sky','lord_day','outer_flight',
                      'other_months','rain','normal_temperature','other_wind'}, inplace=True)

    return df


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



##### DATA PRE-PROCESSING #####
hopsworks_api_key = os.environ['HOPSWORKS_API_KEY']
project = hopsworks.login(api_key_value = hopsworks_api_key)
fs = project.get_feature_store()

fg = fs.get_feature_group(
        name="flight_weather_dataset",
        version=1,
    )
df = fg.read(dataframe_type = 'pandas')
df = uniform_dataframe_for_training(df)  


##### MODEL EVALUATION AND SELECTION #####
train, test = train_test_split(df, test_size=0.2)
Xtrain = train.drop(columns={'dep_delay'})
ytrain = train['dep_delay']
Xtest  = test.drop(columns={'dep_delay'})
ytest  = test['dep_delay']


if MODEL_SELECTION:

    clf = xgboost.XGBRegressor()
    gbc = GridSearchCV(clf, param_grid = [])

    if not MODEL_EVALUATION:
        clf = xgboost.XGBRegressor(eval_metric='rmse', early_stopping_rounds=10)
        params = {'n_estimators': np.arange(3,40,5), 
                  'max_depth'   : np.arange(3,15,2), 
                  'eta'         : np.arange(0.1, 1.5, 0.2),
                  'subsample'   : np.arange(0.6, 1.0, 0.1)}
    
    elif MODEL_EVALUATION:
        train, eval = train_test_split(train, test_size = 0.125)
        Xtrain = train.drop(columns={'dep_delay'})
        ytrain = train['dep_delay']
        Xeval  = eval.drop(columns={'dep_delay'})
        yeval  = eval['dep_delay']
        eval_set = [(Xeval, yeval)]

        clf = xgboost.XGBRegressor()
        params = {'n_estimators': np.arange(3,40,5), 
                  'max_depth'   : np.arange(3,15,2), 
                  'eta'         : np.arange(0.1, 1.5, 0.2),
                  'subsample'   : np.arange(0.6, 1.0, 0.1)}

    gbc = GridSearchCV(clf, param_grid = params, cv = 3, n_jobs=-1, verbose=3, scoring='neg_root_mean_squared_error')
    gbc.fit(Xtrain, ytrain, verbose = 1)
    cv = pd.DataFrame(gbc.cv_results_)

    print(cv.sort_values(by = 'rank_test_score').T)
    print(gbc.best_params_)



  