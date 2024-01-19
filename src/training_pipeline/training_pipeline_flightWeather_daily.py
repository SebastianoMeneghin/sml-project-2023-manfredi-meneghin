import os
import modal

stub = modal.Stub("flight_delays_training_pipeline_daily")
image = modal.Image.debian_slim().pip_install(["hopsworks", "joblib", "seaborn","scikit-learn==1.1.1", "numpy",
                                               "pandas", "pandasql", "xgboost", "cfgrib", "eccodes", "pygrib"])

@stub.function(cpu=1.0, image=image, schedule=modal.Cron('1 0 * * *'), secret=modal.Secret.from_name("hopsworks_sml_project"))
def f():
    g()


def uniform_dataframe_for_training(df):
    import hopsworks
    import numpy as np
    import pandas as pd
    import math
    
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


def create_last_model_performance_dataframe_row(size, model_metrics):
    
    import pandas as pd
    import numpy as np
    from datetime import datetime

    today_current_datetime   = datetime.now()
    today_formatted_datetime = today_current_datetime.strftime("%Y-%m-%d_%H:%M:%S")
    mae = model_metrics.get('mae')
    mse = model_metrics.get('mse')

    columns = ['timestamp','dateset_size','mae','mse']
    row_df = pd.DataFrame(columns=columns)
    row_df.loc[0] = [today_formatted_datetime, size, mae, mse]

    return row_df


def training_pipeline_feature_collect():
    import json
    import os
    import joblib
    import pandas as pd
    import numpy as np
    import xgboost
    import hopsworks
    from datetime import datetime

    # Read data into dataframe and preprocess the dataset 
    project       = hopsworks.login(api_key_value = os.environ['HOPSWORKS_API_KEY'])
    feature_store = project.get_feature_store()
    feature_group = feature_store.get_feature_group(name = 'flight_weather_dataset', version=1)
    df            = feature_group.read(dataframe_type='pandas')
    df            = uniform_dataframe_for_training(df)

    return df


def replace_model_on_hopsworks(local_file_name, hopsworks_file_name, hopsworks_directory):

    import hopsworks
    import pandas as pandas
    import os
    import json
    from hopsworks.client.exceptions import RestAPIError

    # Login in hopsworks
    project = hopsworks.login(api_key_value = os.environ['HOPSWORKS_API_KEY'])
    dataset_api = project.get_dataset_api()

    # Get the model local path
    file_path      = os.path.abspath(local_file_name)

    # Upload the new model
    dataset_api.upload(file_path, hopsworks_directory, overwrite=True)


def replace_file_on_hopsworks(local_file_name, hopsworks_file_name, hopsworks_directory):

    import hopsworks
    import pandas as pandas
    import os
    import json
    from hopsworks.client.exceptions import RestAPIError

    # Login in hopsworks
    project = hopsworks.login(api_key_value = os.environ['HOPSWORKS_API_KEY'])
    dataset_api = project.get_dataset_api()

    # Get the model local path
    file_path      = os.path.abspath(local_file_name)

    '''
    # Remove model from Hopsworks
    try:
        dataset_api.remove(hopsworks_directory + local_file_name)
    except RestAPIError:
        print('I was not able to remove the file')
    
    # Remove model from Hopsworks
    try:
        dataset_api.remove(hopsworks_directory + local_file_name)
    except RestAPIError:
        print('I was not able to remove the directory')
    '''
    # Remove model from Hopsworks
    try:
        dataset_api.mkdir(hopsworks_directory)
    except RestAPIError:
        print('I was not able to remove the file')

    # Upload the new model
    print('Now I try to upload the model!')
    dataset_api.upload(file_path, hopsworks_directory, overwrite=True)


def replace_file_on_hopsworks_Iter(local_file_name, hopsworks_file_name, hopsworks_directory, model_bool):

    import hopsworks
    import pandas as pandas
    import os
    import json
    from hopsworks.client.exceptions import RestAPIError

    OPERATION_DONE = False
    while(not OPERATION_DONE):
        try:
            if model_bool:
                replace_model_on_hopsworks(local_file_name, hopsworks_file_name, hopsworks_directory)
            else:
                replace_file_on_hopsworks(local_file_name, hopsworks_file_name, hopsworks_directory)
            OPERATION_DONE = True
            print('\n**** File ' + local_file_name + ' replaced successfully! ****\n')

        except RestAPIError as e:
            print(e)
            print('\n\n**** Another temptative started now ****\n\n')


def training_pipeline_model_training_and_saving(df, save_also_schema):
    import json
    import os
    import joblib
    import pandas as pd
    import numpy as np
    import xgboost
    import hopsworks
    import time
    from datetime import datetime
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from hopsworks.client.exceptions import RestAPIError

    # Set the hopsworks file directory
    hopsworks_directory = '/Projects/SMLFinalProject3/Models/flight_weather_delay_model/10/'


    ##### MODEL TRAINING #####
    model = xgboost.XGBRegressor(eta= 0.1, max_depth= 7, n_estimators= 38, subsample= 0.8)
    train, test = train_test_split(df, test_size=0.2)
    Xtrain = train.drop(columns={'dep_delay'})
    ytrain = train['dep_delay']
    Xtest  = test.drop(columns={'dep_delay'})
    ytest  = test['dep_delay']

    # Train and test the model
    model.fit(Xtrain, ytrain)
    y_pred = model.predict(Xtest)
    model_metrics = {'mae' :mean_absolute_error(ytest, y_pred), 'mse': mean_squared_error(ytest, y_pred)}


    # Set the model local file name and save it locally
    model_file_name  = 'flight_weather_delay_model.pkl'
    joblib.dump(model, model_file_name)

    # Iteratively try to replace the model on Hopsworks with this new version
    replace_file_on_hopsworks_Iter(model_file_name, model_file_name, hopsworks_directory, False)

    if save_also_schema:
        # Specify the schema of the models' input/output using the features (Xtrain) and labels (ytrain)
        input_schema  = Schema(Xtrain)
        output_schema = Schema(ytrain)
        schema        = ModelSchema(input_schema, output_schema)
        
        # Set the schema local file name and save it locally
        schema_file_name = 'model_schema.json'
        with open(schema_file_name, 'w+') as file:
            json.dump(schema, file, default=lambda o: getattr(o, "__dict__", o), sort_keys=True, indent=2)

        # Iteratively try to replace the model schema on Hopsworks with this new version
        replace_file_on_hopsworks_Iter(schema_file_name, schema_file_name, hopsworks_directory, True)
        os.remove(schema_file_name)

    # Remove the model file from the memory
    os.remove(model_file_name)

    # Return the model metrics in order to save them in the Model Metrics feature group
    return model_metrics


def training_pipeline_save_model_performances(dataset_size, model_metrics):
    import json
    import os
    import pandas as pd
    import hopsworks
    from datetime import datetime
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Save the new model performances in the dedicated feature group
    project           = hopsworks.login(api_key_value = os.environ['HOPSWORKS_API_KEY'])
    feature_store     = project.get_feature_store()
    performance_fg     = feature_store.get_feature_group(name = 'model_performance', version = 1)
    performance_df_row = create_last_model_performance_dataframe_row(dataset_size, model_metrics)
    performance_fg.insert(performance_df_row)


def g():
    import json
    import os
    import joblib
    import pandas as pd
    import numpy as np
    import xgboost
    import hopsworks
    from datetime import datetime
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    df            = training_pipeline_feature_collect()
    model_metrics = training_pipeline_model_training_and_saving(df, True)

    # Save the new model performances in the dedicated feature store
    training_pipeline_save_model_performances(df.shape[0], model_metrics)