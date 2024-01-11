import requests
import pandas as pd
import json
import os
from src.other.utils import get_date_label, get_year_month_label, get_mesan_date_label, get_padded_hour
import pygrib


# Set the latitude and longitude limits
target_latitude_down  = 59.575368
target_latitude_up    = 59.600368
target_longitude_down = 17.875208
target_longitude_up   = 17.876174


# Set the skeleton of the dataframe
columns = ['date', 'time', 'temperature', 'visibility', 'pressure', 'humidity', 'gusts_wind', 'u_wind', 'v_wind', 'prep_1h', 
                     'snow_1h', 'gradient_snow', 'total_cloud', 'low_cloud', 'medium_cloud', 'high_cloud', 'type_prep', 'sort_prep']
weather_df = pd.DataFrame(columns=columns)

# Create new_row_attr to add future files' rows to the dataframe and set counter to 0
new_row_attr       = []
file_counter       = 0
checkpoint_counter = 0

# Select after how many files a new checkpoint is saved
check_step = 120    #(10s each GRIB file -> 20*60s = 1200s -> ~120 files)


# Insert the limit of wanted iteration ranges:
year           = 2023
starting_month = 12
ending_month   = 12
starting_day   = 30
ending_day     = 31
starting_hour  = 0
ending_hour    = 23


for month in range(starting_month, ending_month + 1):
    for day in range(starting_day, ending_day + 1):
        compile = True

        # Always imposed limit for query
        if (month in {4, 6, 9, 11} and day in {31}) or (month in {2} and (year % 4 == 0) and day in {30, 31}) or (month in {2} and (year % 4 != 0) and day in {29, 30, 31}):
                compile = False
        
        if compile:
            new_row_date = get_date_label(year, month, day, 'hyphen')

            for hour in range(starting_hour, ending_hour + 1):
                file_counter += 1
                # Empty the list representing the future df element and append the needed information of the df
                new_row_attr = []
                new_row_time = hour
                new_row_attr.append(new_row_date)
                new_row_attr.append(new_row_time)

                # Get some data with specific formatting to put in url and file name
                yyyymm = get_year_month_label(year, month, 'empty')
                yyyymmddhhhh = get_mesan_date_label(year, month, day, hour, 'empty')
                hh = get_padded_hour(hour)

                # Create the url depending on the SMHI Historical Data filesystem requirements
                url = 'https://opendata-download-grid-archive.smhi.se/data/6/' + yyyymm + '/MESAN_' + yyyymmddhhhh + '+000H00M'
                response = requests.get(url)

                print(url)

                file_name = 'ARN_' + new_row_date + '_' + hh
                save_path = "/mnt/c/Developer/University/SML/sml-project-2023-manfredi-meneghin/datasets/smhi_historical_data/"
                complete_name = os.path.join(save_path, file_name)


                with open(complete_name, "wb") as outfile: 
                    outfile.write(response.content)

                    grib_path = complete_name
                    grib_file = pygrib.open(grib_path)                    

                    for label in ['Temperature', 'Visibility', 'Pressure reduced to MSL', 'Relative humidity', 'Wind gusts', 'u-component of wind', 
                                'v-component of wind', '1 hour precipitation', '1 hour fresh snow cover', 
                                'Snowfall (convective + stratiform) gradient', 'Total cloud cover', 'Low cloud cover', 
                                'Medium cloud cove', 'High cloud cover', 'Type of precipitation', 'Sort of precipitation']:
                        
                        #df_label = get_df_label_from_grib_label(label)
                        temp_grb = grib_file.select(name=label)[0]
                        part, latss, lotss = temp_grb.data(lat1=target_latitude_down,lat2=target_latitude_up,lon1=target_longitude_down,lon2=target_longitude_up)

                        new_row_attr.append(part[0])


                    weather_df.loc[len(weather_df.index)] = new_row_attr
                    #print(weather_df.T)

                outfile.close()
                os.remove(complete_name)

                # Save in local memory every "check_step" read files
                if (file_counter % check_step == 0):
                    checkpoint_path = "/mnt/c/Developer/University/SML/sml-project-2023-manfredi-meneghin/datasets/smhi_historical_data/"
                    checkpoint_name = 'checkpoint_' + get_padded_hour(checkpoint_counter) + '.csv'
                    checkpoint_complete_path = os.path.join(checkpoint_path, checkpoint_name)

                    print(checkpoint_complete_path)

                    with open(checkpoint_complete_path, "wb") as df_out:
                        weather_df.to_csv(df_out, index = False)

                    df_out.close()

                    checkpoint_counter += 1