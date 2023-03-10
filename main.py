from pathlib import Path
import os
import pandas as pd
import plotly.express as px
from datetime import datetime as dt
import matplotlib.pyplot as plt
from scipy import integrate
from sklearn.linear_model import LinearRegression

data_folder = os.path.join(Path(os.getcwd()).parent, 'data')
csvfiles = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
save_to_excel = False
for f in csvfiles[1:2]: # case 1
    data = pd.read_csv(f, low_memory=False, index_col=1)
    if save_to_excel:
        data.to_excel(f[:-3]+'xlsx')

data1 = data.iloc[2:,:]
data1 = data1.iloc[:-1, :]
data1.loc[:, 'date'] = [dt.fromtimestamp(float(x)/1000) for x in data1['timestamp']]

input_cols = ['date', # date
'rt_temp', # room temperature
'ambient_temp', # ambient temperature
'rt_sp_heating', # setpoint
'status_3wv'] # status_3wv==1 then domestic hot water is active
# status_3wv==0 then space heating is active
output_cols = [col for col in data1.columns if 'input_kw' in col]
data_ml = data1[input_cols+output_cols]
data_ml = data_ml.set_index('date').astype(float)

data_ml_resampled = data_ml.resample('24H').mean()
data_ml_resampled = data_ml_resampled.dropna()
data_ml_resampled['2012-03-06':'2022-03-10'].plot()