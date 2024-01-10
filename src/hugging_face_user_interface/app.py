import gradio as gr
import pandas as pd
import hopsworks
import math
import os

def download_data(path):
    os.path.abspath(path)
    df = pd.read_csv(path)
    wanted_df = df[['airport', 'flight_number', 'ontime', 'delayed']].copy()
    return wanted_df

def access_online_dataframe():
    # Get data from online source
    hopsworks_api_key  = os.environ['HOPSWORKS_API_KEY']
    project            = hopsworks.login(api_key_value = hopsworks_api_key)
    dataset_api        = project.get_dataset_api()
    today_path         = dataset_api.download("Resources/today_timetable_prediction/today_timetable_prediction.csv", overwrite=True)
    tomorrow_path      = dataset_api.download("Resources/tomorrow_timetable_prediction/tomorrow_timetable_prediction.csv", overwrite=True)
    today_df           = download_data(today_path)
    tomorrow_df        = download_data(tomorrow_path)
    return today_df, tomorrow_df


def get_possible_destinations():
    today_df, tomorrow_df = access_online_dataframe()
    total_df = pd.DataFrame({'airport': pd.concat([today_df['airport'], tomorrow_df['airport']]).drop_duplicates().reset_index(drop=True)})
    total_dest = (total_df['airport']).tolist()
    return total_dest


def get_specific_flights(day, max_delay, departure_hour, ampm, weather, destinations, yes):
    today_df, tomorrow_df = access_online_dataframe()
    df = pd.DataFrame()
    if(day == 'today'):
        df = today_df
    else:
        df = tomorrow_df

    # Remove unwanted destinations
    destinations = [dest for dest in destinations if dest not in ["That's a reason why I travel alone...", "I prefer not to say"]]

    # Select only flight during the same departure hour
    df['departure_hour'] = df['ontime'].str.split(':').str[0].astype(int)
    df = df[df['departure_hour'] == departure_hour].drop(columns=['departure_hour'])

    # Convert time columns to datetime objects
    df['ontime'] = pd.to_datetime(df['ontime'], format='%H:%M')
    df['delayed'] = pd.to_datetime(df['delayed'], format='%H:%M')

    # Get flight with less delay than the given and from the destinations selected, of the right day
    df['delay'] = (df['delayed'] - df['ontime']).dt.total_seconds() / 60
    filtered_df = df.loc[(df['delay'] < max_delay) & (df['airport'].isin(destinations)), ['airport', 'flight_number', 'ontime', 'delayed']]
 
    # Convert the string to datetime, then the datetime column to HH:MM
    filtered_df['ontime'] = pd.to_datetime(filtered_df['ontime'])
    filtered_df['ontime'] = filtered_df['ontime'].dt.strftime('%H:%M')
    filtered_df['delayed'] = pd.to_datetime(filtered_df['delayed'])
    filtered_df['delayed'] = filtered_df['delayed'].dt.strftime('%H:%M')
    
    return filtered_df



flights = gr.Interface(
    get_specific_flights,
    [
        gr.Radio(["today", "tomorrow"], type="value", label="Day", info="When do you have the plane?"),
        gr.Slider(0, 50, value=20, label="Possible Delay", info="How unfortunate do you wanna be?"),
        gr.Number(precision=0, minimum=0, maximum=23, label="Departure Time"),
        gr.Radio(["am", "pm"], type="index", info="It's the same, no worries!", label = "Am or Pm?"),
        gr.CheckboxGroup(["Yes, it's cloudy", "I am not in Stockholm"], label="Weather", info="Is it a typical Stockholm day?"),
        gr.Dropdown(get_possible_destinations() + ["That's a reason why I travel alone...", "I prefer not to say"], 
                    type = "value", multiselect=True, label="Destination", value=["That's a reason why I travel alone..."],
                    info="Are you just curious or you are actually going somewhere? Where? With who?"),
        gr.Radio(["Yes", "Yes", "Yes"], type="index", label="Let's guess?", info="We know that you'll say yes!"),
    ],
    "dataframe",
)

flights.launch()
