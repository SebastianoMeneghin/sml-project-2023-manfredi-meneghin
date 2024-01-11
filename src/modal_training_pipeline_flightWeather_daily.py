import os
import modal

stub = modal.Stub("flight_delays_training_pipeline_daily")
image = modal.Image.debian_slim().pip_install(["hopsworks", "joblib", "seaborn","scikit-learn==1.1.1", "numpy",
                                               "pandas", "pandasql", "xgboost", "cfgrib", "eccodes", "pygrib"])

@stub.function(cpu=1.0, image=image, schedule=modal.Cron('1 0 * * *'), secret=modal.Secret.from_name("hopsworks_sml_project"))
def f():
    g()


def uniform_dataframe_for_training(df):
    '''
    Given a dataset with the columns names extracted from the APIs data, return a dataset (dataframe)
    uniformed in order to be possible training a model on that
    '''
    import pandas as pd
    import numpy as np

    df.drop(columns={'trip_time', 'dep_ap_gate', 'airline_iata_code', 'flight_iata_number', 
                     'arr_ap_iata_code', 'status','dep_ap_iata_code', 'date', 'high_cloud', 
                     'medium_cloud', 'low_cloud', 'gusts_wind'}, inplace=True)

    # Some data should be casted to int64
    convert_column = ['pressure','total_cloud', 'sort_prep','humidity']
    for col in convert_column:
        df = df.astype({col: 'int64'})

    # Remove outliners in delay (dep_delay > 120)
    row_list = []
    for row in range(df.shape[0]):
        if (df.at[row, 'dep_delay'] > 120):
            row_list.append(row)
    df.drop(row_list, inplace = True)
    df.reset_index(inplace = True)
    df.drop(columns={'index'}, inplace = True)

    # Make wind_dir a categorical feature with numbers and not string labels
    dir_dict = {'SW':0,'S':1,'SE':2,'E':3,'NE':4,'N':5,'NW':6,'W':7}
    direction_list = []
    for row in range(df.shape[0]):
        direction = df.at[row, 'wind_dir']
        number = dir_dict.get(direction)
        direction_list.append(number)
    df.drop(columns={'wind_dir'}, inplace = True)
    df['wind_dir'] = direction_list

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

    # Remove model from Hopsworks
    try:
        dataset_api.remove(hopsworks_directory + hopsworks_file_name)
    except RestAPIError:
        print('I was not able to remove the file')
    #Create new directory for model
    try:
        dataset_api.mkdir(hopsworks_directory)
    except RestAPIError:
        print("The folder already exists")

    # Upload the new model
    dataset_api.upload(file_path,  hopsworks_directory, overwrite=True)


def replace_file_on_hopsworks_Iter(local_file_name, hopsworks_file_name, hopsworks_directory):

    import hopsworks
    import pandas as pandas
    import os
    import json
    from hopsworks.client.exceptions import RestAPIError

    OPERATION_DONE = False
    while(not OPERATION_DONE):
        try:
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
    hopsworks_directory = '/Projects/SMLFinalProject3/Models/flight_weather_delay_model/2/'


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
    replace_file_on_hopsworks_Iter(model_file_name, model_file_name, hopsworks_directory)

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
        replace_file_on_hopsworks_Iter(schema_file_name, schema_file_name, hopsworks_directory)
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
    model_metrics = training_pipeline_model_training_and_saving(df, False)

    # Save the new model performances in the dedicated feature store
    training_pipeline_save_model_performances(df.shape[0], model_metrics)