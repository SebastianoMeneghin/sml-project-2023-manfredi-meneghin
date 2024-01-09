import json
import os
import joblib
import pandas as pd
import numpy as np
import xgboost
import hopsworks
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


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




##### DATA PRE-PROCESSING #####
hopsworks_api_key = os.environ['HOPSWORKS_API_KEY']
project = hopsworks.login(api_key_value = hopsworks_api_key)
fs = project.get_feature_store()

fg = fs.get_feature_group(
        name="flight_weather_dataset",
        version=1,
    )
df = fg.read(dataframe_type = 'pandas')


# Due to a not optimal flight API, some data as "DepApGate", "TimeTrip" cannot be calculated in new data
# Furthermore, the following columns are dropped:
df.drop(columns={'trip_time', 'dep_ap_gate'}, inplace = True)

# Due to a disproportion between categories and flights features (too mant for too few), the future columns are dropped:
# When there will be more data, it will be interesting to add them to the model
df.drop(columns={'airline_iata_code', 'flight_iata_number', 'arr_ap_iata_code'}, inplace = True)

# Some data are used as a key, but are not made to be variables of our model, furthermore are dropped:
# If there will be data coming from different airports, it will be interesting to add 'depApIataCode' as variable.
df.drop(columns={'status','dep_ap_iata_code', 'date'}, inplace = True)

# Some data should be casted to int64
convert_column = ['pressure','total_cloud', 'high_cloud', 'medium_cloud', 'low_cloud', 'sort_prep','humidity']
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

# Since total_cloud can summarize the others:
df.drop(columns={'high_cloud', 'medium_cloud', 'low_cloud'}, inplace = True)

# Since wind_speed can summarize gusts_wind:
df.drop(columns={'gusts_wind'}, inplace = True)

# Make wind_dir a categorical feature with numbers and not string labels
dir_dict = {'SW':0,'S':1,'SE':2,'E':3,'NE':4,'N':5,'NW':6,'W':7}
direction_list = []
for row in range(df.shape[0]):
    direction = df.at[row, 'wind_dir']
    number = dir_dict.get(direction)
    direction_list.append(number)
df.drop(columns={'wind_dir'}, inplace = True)
df['wind_dir'] = direction_list
    


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
model_metrics = [mean_absolute_error(ytest, y_pred), mean_squared_error(ytest, y_pred)]
print(f'\nTrained model metrics: {model_metrics}\n')



##### MODEL SAVING #####
mr = project.get_model_registry()
model_dir="flight_weather_delay_dir"

if os.path.isdir(model_dir) == False:
    os.mkdir(model_dir)
# Save the model
joblib.dump(model, model_dir + "/flight_weather_delay_model.pkl")


# Specify the schema of the models' input/output using the features (Xtrain) and labels (ytrain)
input_schema = Schema(Xtrain)
output_schema = Schema(ytrain)
model_schema = ModelSchema(input_schema, output_schema)

# Since the model cannot be overwritten, use the get_model/set_model workaround to create a new version of the model
# and set is as the most updated
flight_weather_delay_model = mr.python.create_model(
    name="flight_weather_delay_model", 
    metrics={"mean_absolute_error" : model_metrics[0]},
    model_schema=model_schema,
    version = int(get_model_last_version_number(project)) + 1,
    description="XGBoost Regression model for flight departure delays, trained on flight info and weather info"
)


# Upload the model to the model registry
flight_weather_delay_model.save(model_dir)
set_model_last_version_number(project, get_model_last_version_number + 1)



  