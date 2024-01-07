import os
import pandas as pd
import pandasql as sqldf

# Load the two historical datasets into dataframes
smhi_file_name     = 'historical_data_time_shifted.csv'
smhi_directory     = '/mnt/c/Developer/University/SML/sml-project-2023-manfredi-meneghin/datasets/smhi_historical_data/'
smhi_complete_path = os.path.join(smhi_directory, smhi_file_name)

zyla_file_name     = 'historical_flight_data_processed.csv'
zyla_directory     = '/mnt/c/Developer/University/SML/sml-project-2023-manfredi-meneghin/datasets/zylaAPI_flights/'
zyla_complete_path = os.path.join(zyla_directory, zyla_file_name)

smhi_df = pd.read_csv(smhi_complete_path)
zyla_df = pd.read_csv(zyla_complete_path)

# Create a query to join the two datasets, according to date and time
q = """
    select
        *
    from
        zyla_df z
    inner join
        smhi_df s
            on z.date = s.date and z.time = s.time
"""

# Join the two datasets according and remove duplicated columns
merged_df = sqldf.sqldf(q)
merged_df = merged_df.loc[:,~merged_df.columns.duplicated()].copy()


# Save the new dataframe in a new file (.csv)
merged_path = "/mnt/c/Developer/University/SML/sml-project-2023-manfredi-meneghin/datasets/"
merged_name = 'join_dataset_smhi_zyla.csv'
merged_complete_path = os.path.join(merged_path, merged_name)

with open(merged_complete_path, "wb") as df_out:
    merged_df.to_csv(df_out, index= False)
df_out.close()