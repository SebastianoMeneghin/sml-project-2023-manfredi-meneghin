import requests
import pandas as pd
import json
import re
import os
from utils import get_data, get_month, zylaAPI_url, get_date_label, get_df_label_from_grib_label
import xarray as xr
import cfgrib
import pygrib

columns = ['date', 'time', 'temperature', 'visibility', 'pressure', 'humidity', 'gusts_wind', 'u_wind', 'v_wind', 'prep_1h', 
                     'snow_1h', 'gradient_snow', 'total_cloud', 'low_cloud', 'medium_cloud', 'high_cloud', 'type_prep', 'sort_prep']
weather_df = pd.DataFrame(columns=columns)
print(weather_df)

new_row_attr = []

# Add create information for the new dataframe row
new_row_date = '2021-06-25'
new_row_time = '23'
new_row_attr.append(new_row_date)
new_row_attr.append(new_row_time)

url = "https://opendata-download-grid-archive.smhi.se/data/6/202106/MESAN_202106250000+000H00M"
response = requests.get(url)
file_name = "ARN_20210625_00"
save_path = "/mnt/c/Developer/University/SML/sml-project-2023-manfredi-meneghin/datasets/smhi_weather/"
complete_name = os.path.join(save_path, file_name)


with open(complete_name, "wb") as outfile: 
    outfile.write(response.content)

    grib_path = complete_name
    grib_file = pygrib.open(grib_path)

    # With them, I am capturing all the information needed, from two different sensors
    target_latitude_down  = 59.575368
    target_latitude_up    = 59.600368
    target_longitude_down = 17.875208
    target_longitude_up   = 17.876174

    for label in ['Temperature', 'Visibility', 'Pressure reduced to MSL', 'Relative humidity', 'Wind gusts', 'u-component of wind', 
                  'v-component of wind', '1 hour precipitation', '1 hour fresh snow cover', 
                  'Snowfall (convective + stratiform) gradient', 'Total cloud cover', 'Low cloud cover', 
                  'Medium cloud cove', 'High cloud cover', 'Type of precipitation', 'Sort of precipitation']:
        
        df_label = get_df_label_from_grib_label(label)
        temp_grb = grib_file.select(name=label)[0]
        part, latss, lotss = temp_grb.data(lat1=target_latitude_down,lat2=target_latitude_up,lon1=target_longitude_down,lon2=target_longitude_up)

        new_row_attr.append(part[0])


    weather_df.loc[len(weather_df.index)] = new_row_attr
    print(weather_df.T)


