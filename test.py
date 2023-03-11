from pathlib import Path
import os
import pandas as pd
import plotly.express as px
from datetime import datetime as dt
import matplotlib.pyplot as plt
from scipy import integrate
from sklearn.linear_model import LinearRegression
from calendar import monthrange

CONVERSION_FACTOR_PRICE = 0.40 # 40 cent per KWH
CONVERSION_FACOTOR_EMISSION = 0.14 # 0,14 kg CO2 per KWH
AVERAGE_FAMILY_EMISSION_YEAR = 2690 # 2690 kg CO2 average annual emission of family
HOURSE_IN_DAY = 24

def change_power_to_price(x_test, y_serie):
    for current_index in range(len(x_test)):
        year = x_test[current_index][0:4]
        month = x_test[current_index][5:7]
        print(type(y_serie[current_index]))
        print(int(y_serie[current_index]))
        y_serie[current_index] = y_serie[current_index] * float(CONVERSION_FACTOR_PRICE * HOURSE_IN_DAY)
    return y_serie

data_folder = os.path.join(Path(os.getcwd()).parent, 'data')
csvfiles = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
save_to_excel = False
index_file_to_read = 0 #  case 1


data = pd.read_csv(csvfiles[index_file_to_read], low_memory=False)
if save_to_excel:
    data.to_excel(f[:-3]+'xlsx')

data1 = data.copy()
data1.loc[:, 'date'] = [dt.fromtimestamp(float(x)/1000) for x  in data1['timestamp']]

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

data_ml_resampled['2022-03-06':'2022-03-10'].plot()
# create train test partition
data_train = data_ml_resampled['2022-03-06':'2022-03-10']
data_test  = data_ml_resampled['2022-03-16':]

power_to_estimate = 'hp'
power_ind = 0 if power_to_estimate=='buh' else 1 if power_to_estimate=='hp' else 0

x_train, y_train = data_train.iloc[:, [0,1,3]].values, data_train[output_cols[power_ind]].values
x_test, y_test = data_train.iloc[:,[0,1,3]].values, data_train[output_cols[power_ind]].values

model = LinearRegression().fit(x_train, y_train)

r_2 = model.score(x_train, y_train)

y_pred = model.predict(x_test)

str_x = data_test.index[0:len(y_test)].astype(str)
fig, ax = plt.subplots()
ax.plot(str_x, y_test)
ax.plot(str_x, y_pred)
ax.set_ylabel('Average daily power')
ax.set_xlabel('Day')
#ax.set_xticklabels(ax.get_xticks(), rotation = 45)
ax.legend(['real', 'estimated'])
ax.set_title('Estimation of Power base on indoor and ambient temperatures ')


fig, ax = plt.subplots()
y_test = change_power_to_price(str_x, y_test)
y_pred = change_power_to_price(str_x, y_pred)
ax.plot(str_x, y_test)
ax.plot(str_x, y_pred)
ax.set_ylabel('Average Price per Month (€)')
ax.set_xlabel('Day')
ax.legend(['real', 'estimated'])
ax.set_title('Estimation of Price per month (€) on indoor and ambient temperatures ')






plt.show() 


