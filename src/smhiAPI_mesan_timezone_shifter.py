import requests
import pandas as pd
import json
import os
from utils import get_data, get_month, zylaAPI_url, get_date_label, get_df_label_from_grib_label, get_year_month_label, get_mesan_date_label, get_padded_hour, one_hour_forward, one_hour_backward, one_day_backward, one_day_forward, get_wind_dir_label
import numpy as np
from datetime import datetime

######################################################################################################
# Here a smhiAPI historical mesan file is processed by adjusting the date and the time according to 
# the wanted Time Zone (in this project case is +01:00 ) and to the DST (Daylight Saving Time). This
# is done since SMHI OpenData provides data that are all around the year saved according to UCT, with 
# no DST activation. This works only on one year and one place, since the logic depends on the time-zone
# and on the date of DST activation/deactivation. Data are then saved in a new file.
######################################################################################################

# Read a checkpoint or a file (.csv) and load it on a dataframe
file_name = 'smhiAPI_mesan_historical_data.csv'
file_path = '/mnt/c/Developer/University/SML/sml-project-2023-manfredi-meneghin/datasets/smhi_historical_forecast/'
complete_name = file_path + file_name
df = pd.read_csv(complete_name)

row_to_drop = []

# For each row of the dataframe, adjust date and time columns
for row in range(df.shape[0]):
    # Get year, month, day and time from the selected row
    date_string = df.at[row, 'date']
    date_object = datetime.strptime(date_string, "%Y-%m-%d")
    yyyy = date_object.year
    mm   = date_object.month
    dd   = date_object.day
    hh   = df.at[row, 'time']

    # Set when the DST was active (this refers to Sweden 2023, from 26th March - 01:00 (UCT00:00) to 29th October - 01:00 (UCT00:00))
    if (mm == 3 and dd == 26 and hh >= 1) or (mm == 3 and dd >= 27) or (mm in {4,5,6,7,8,9}) or (mm == 10 and dd <= 28) or (mm == 10 and dd == 29 and hh == 0):
        yyyy,mm,dd,hh = one_hour_forward(yyyy,mm,dd,hh)

    # Set when the DST goes from summer time to winter time, since one our computed is lost (In Sweden 2023: 29th Oct - 01:00 (UCT00:00))
    if (mm == 10 and dd == 29 and hh == 1):
        row_to_drop.append(row)
        break

    # According to the time-zone differences, the clock is moved forward or backward (In Sweden 2023: +01:00)
    yyyy,mm,dd,hh = one_hour_forward(yyyy,mm,dd,hh)

    # The old values are then replace by the just shifted date and time
    df.at[row, 'date'] = get_date_label(yyyy, mm, dd, 'hyphen')
    df.at[row, 'time'] = hh

# Drop the DST deactivation's hour
print(row_to_drop)
df.drop(row_to_drop, inplace= True)


# Save the new dataframe in a new file (.csv)
# Save clean dataframe in a new file (.csv)
ts_path = "/mnt/c/Developer/University/SML/sml-project-2023-manfredi-meneghin/datasets/time_shifted/"
ts_name = 'smhiAPI_mesan_historical_data_TS.csv'
ts_complete_path = os.path.join(ts_path, ts_name)

with open(ts_complete_path, "wb") as df_out:
    df.to_csv(df_out, index= False)
df_out.close()

    
    