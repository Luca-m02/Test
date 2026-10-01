#

import pytz
import os
import pandas as pd

def concatenate_csv(source_path='./dataset'):
    key = 'timestamp'
    sep = ';' # ';'
    files_list = sorted(f for f in os.listdir(source_path) if f.endswith('.csv'))
    to_zone = pytz.timezone('Europe/Rome')
    days = []

    #concateno tutti i file in uno unico cronologico
    for name in files_list:
        chain2_data = pd.read_csv(os.path.join(source_path, name), sep=sep)
        chain2_data[key] = pd.to_datetime(chain2_data[key])
        chain2_data = chain2_data.set_index(key)
        chain2_data.index = chain2_data.index.tz_convert(to_zone)
        days.append(chain2_data[['P_ex', 'P_pr']])

    data = pd.concat(days, axis=0).sort_index()
    data = data[~data.index.duplicated(keep='last')]
    return data  

PATH = os.path.dirname(os.path.abspath(__file__))
DESTINATION_PATH = os.path.join(PATH, 'data')
os.makedirs(DESTINATION_PATH, exist_ok=True)
for i in (1, 2, 3):
    data = concatenate_csv(os.path.join(PATH, f'CASA_{i}'))
    data.to_csv(os.path.join(DESTINATION_PATH, f'casa_{i}_full.csv'))