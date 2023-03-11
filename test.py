from pathlib import Path
import os
import pandas as pd
import plotly.express as px
from datetime import datetime as dt
import matplotlib.pyplot as plt
from scipy import integrate
from sklearn.linear_model import LinearRegression

def load_data():
    data_folder = os.path.join(Path(os.getcwd()).parent, 'data')
    csvfiles = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
    save_to_excel = False
    index_file_to_read = 0 #  case 1

    DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    DAY0 = 6

    data = pd.read_csv(csvfiles[index_file_to_read], low_memory=False)
    if save_to_excel:
        data.to_excel(f[:-3]+'xlsx')
    return data


def convert_data(data):
    data1 = data.copy()
    data1.loc[:, 'date'] = [dt.fromtimestamp(float(x)/1000) for x  in data1['timestamp']]
    data1.loc[:, 'day'] = [(date+ DAY0)%7 for date in range(len(data1['timestamp']))]
    print(data1.columns)

    input_cols = ['date', # date 
                'rt_temp', # room temperature
                'ambient_temp', # ambient temperature
                'rt_sp_heating', # setpoint
                'status_3wv', #] # status_3wv==1 then domestic hot water is active 
                                # status_3wv==0 then space heating is active 
                'day'] # name of the day

    output_cols = [col for col in data1.columns if 'input_kw' in col]  


    data_ml = data1[input_cols+output_cols]
    data_ml = data_ml.set_index('date').astype(float)


    print(data_ml.head(10))
    print('-'*20)
    print(data_ml.tail(10))

    return data_ml, input_cols, output_cols

def resample_24h(data_ml):
    data_ml_resampled = data_ml.resample('24H').mean()

    data_ml_resampled = data_ml_resampled.dropna()

    data_ml_resampled['2022-03-06':'2022-03-10'].plot()

    # create train test partition
    data_train = data_ml_resampled['2022-03-06':'2022-03-10']
    data_test  = data_ml_resampled['2022-03-16':]
    print('-'*20)
    print(data_train.head())

    print('-'*20)
    print(data_test.head())
    return data_train, data_test

def analyse_hp_power_consumption(data_train, output_cols):
    power_to_estimate = 'hp'
    power_ind = 0 if power_to_estimate=='buh' else 1 if power_to_estimate=='hp' else 0

    x_train, y_train = data_train.iloc[:, [0,1,3]].values, data_train[output_cols[power_ind]].values
    x_test, y_test = data_train.iloc[:,[0,1,3]].values, data_train[output_cols[power_ind]].values

    print('here')
    print(output_cols[power_ind])
    return x_train, y_train, x_test, y_test

def data_to_model(x_train, y_train):
    model = LinearRegression().fit(x_train, y_train)

    r_2 = model.score(x_train, y_train)
    print(f"coefficient of determination: {r_2}")
    return model, r_2

def predict(model, x_test):
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
    plt.show()

data = load_data()
data_ml, input_cols, output_cols = convert_data(data)
data_train, data_test = resample_24h(data_ml)
x_train, y_train, x_test, y_test = analyse_hp_power_consumption(data_train, output_cols)
model, r_2 = data_to_model(x_train, y_train)
predict(model, x_test)