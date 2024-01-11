import os
import pandas as pd
import numpy as np
import json
from src.other.utils import merge_and_extract_zylaAPI_flight_infos

# Confirm that the script has started, since it requires some time
print('*Script started\n')

# Set the directoty from which take the files
directory_path    = '/mnt/c/Developer/University/SML/sml-project-2023-manfredi-meneghin/datasets/zylaAPI_flights_raw'

# Set the directory where to save the data and set the new name
save_path 		  = '/mnt/c/Developer/University/SML/sml-project-2023-manfredi-meneghin/datasets/zylaAPI_flights/'
file_to_save_name = 'historical_flight_data.json'
complete_name     = save_path + file_to_save_name

print('*Files will be extracted from: \n' + directory_path + '\n\n' + '*Flight info will be saved in file: \n' + file_to_save_name + '\n' + '*in the directory: \n' + save_path)

# Create then a new file and save there the flight_infos from all the files of the directory
with open(complete_name, 'w') as output_file:
	print(json.dumps(merge_and_extract_zylaAPI_flight_infos(directory_path), indent=2), file=output_file)
output_file.close()

print('*Saved')