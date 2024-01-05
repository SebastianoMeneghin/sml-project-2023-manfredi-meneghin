import pandas as pd
import json
import xarray as xr
import cfgrib
import pygrib

grib1_path = "/mnt/c/Developer/University/SML/sml-project-2023-manfredi-meneghin/datasets/smhi_weather/MESAN_202206250000+000H00M"
grib2_path = "/mnt/c/Developer/University/SML/sml-project-2023-manfredi-meneghin/datasets/smhi_weather/MESAN_202206252200+000H00M"
grib3_path = "/mnt/c/Developer/University/SML/sml-project-2023-manfredi-meneghin/datasets/smhi_weather/MESAN_202312012300+000H00M"

grbs1 = pygrib.open(grib1_path)
grbs2 = pygrib.open(grib2_path)
grbs3 = pygrib.open(grib3_path)

for grb in grbs3: print(grb)


# With them, I am capturing all the information needed, from two different sensors
target_latitude_down  = 59.575368
target_latitude_up    = 59.600368
target_longitude_down = 17.875208
target_longitude_up   = 17.876174

# If not enough values with the previous sensor:
lat2down = 59.62532
lon2down = 17.879502

print('\n\n\n\n')
for temp_grb in pygrib.open(grib2_path):
    print(temp_grb)
    part, latss, lotss = temp_grb.data(lat1=target_latitude_down,lat2=target_latitude_up,lon1=target_longitude_down,lon2=target_longitude_up)
    print(part, latss, lotss, part.shape)