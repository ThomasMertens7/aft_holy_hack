from pathlib import Path
import os
import pandas as pd
import plotly.express as px
from datetime import datetime as dt
import matplotlib.pyplot as plt
from scipy import integrate
from sklearn.linear_model import LinearRegression
from calendar import monthrange
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor



DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
DAY0 = 6
CONVERSION_FACTOR_PRICE = 0.40 # 40 cent per KWH
CONVERSION_FACOTOR_EMISSION = 0.14 # 0,14 kg CO2 per KWH
AVERAGE_FAMILY_EMISSION_YEAR = 2690 # 2690 kg CO2 average annual emission of family
HOURSE_IN_DAY = 24



def change_power_to_price(x_test, y_serie):
    for current_index in range(len(x_test)):
        print(type(y_serie[current_index]))
        print(int(y_serie[current_index]))
        y_serie[current_index] = y_serie[current_index] * float(CONVERSION_FACTOR_PRICE * HOURSE_IN_DAY)
    return y_serie

def load_data():
    data_folder = os.path.join(Path(os.getcwd()).parent, 'data')
    csvfiles = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
    save_to_excel = False
    index_file_to_read = 2 #  case 1

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
    data_ml = data1[input_cols+output_cols]
    data_ml = data_ml.set_index('date').astype(float)

    return data_ml, input_cols, output_cols

def get_first_and_last_date(data):
    fisrt_entry = data['timestamp'][0]
    last_entry = data['timestamp'][len(data['timestamp'])-1]

    first_date = dt.fromtimestamp(float(fisrt_entry) / 1000)
    last_date = dt.fromtimestamp(float(last_entry) / 1000)

    return str(first_date.date()), str(last_date.date())


def resample_24h(data_ml_train, data_ml_test, train_start_date = '2022-03-06',train_end_date = '2022-03-10', test_start_date='2022-03-16',test_end_date='2022-03-20' ):
    data_ml_resampled_train = data_ml_train.resample('24H').mean()
    data_ml_resampled_test = data_ml_test.resample('24H').mean()

    data_ml_resampled_train = data_ml_resampled_train.dropna()
    data_ml_resampled_test = data_ml_resampled_test.dropna()

    data_ml_resampled_train[train_start_date:train_end_date].plot()
    data_ml_resampled_test[test_start_date:test_end_date].plot()

    # create train test partition
    data_train = data_ml_resampled_train[train_start_date:train_end_date]
    data_test  = data_ml_resampled_test[test_start_date:test_end_date]

    print('-'*20)
    print(data_train.head())

    print('-'*20)
    print(data_test.head())
    return data_train, data_test

def analyse_hp_power_consumption(data_train,data_test, output_cols):
    power_to_estimate = 'hp'
    power_ind = 0 if power_to_estimate=='buh' else 1 if power_to_estimate=='hp' else 0
    print("////////")
    import numpy as np
    print(data_test.head())
    print("////////")
    x_train, y_train = data_train.iloc[:, [0,1,3]].values, data_train[output_cols[power_ind]].values
    x_test, y_test = data_test.iloc[:,[0,1,3]].values, data_test[output_cols[power_ind]].values

    print('here')
    print(output_cols[power_ind])
    return x_train, y_train, x_test, y_test

def data_to_model(x_train, y_train):
    model = LinearRegression().fit(x_train, y_train)

    r_2 = model.score(x_train, y_train)
    print(f"coefficient of determination: {r_2}")
    return model, r_2

def data_to_nonlinear_model(x_train_linear, x_train_categorical, y_train):
    # separate the categorical and linear columns
    x_train_linear = x_train_linear.drop('day_of_week', axis=1)
    x_data_nonlinear_categorical = x_train_categorical[['day_of_week']]

    # convert the day of the week column to categorical datatype
    x_data_nonlinear_categorical['day_of_week'] = x_data_nonlinear_categorical['day_of_week'].astype('category')

    # create a decision tree model for the categorical variables
    model_categorical = DecisionTreeRegressor(random_state=0)

    # use bagging to create an ensemble of decision tree models for the categorical variables
    model_categorical_bagging = BaggingRegressor(base_estimator=model_categorical, n_estimators=10, random_state=0)
    model_categorical_bagging.fit(x_data_nonlinear_categorical, y_train)

    # get the predicted values for the categorical variables
    y_train_categorical = model_categorical_bagging.predict(x_data_nonlinear_categorical)

    # concatenate the predicted categorical values with the linear data
    x_train = pd.concat([x_train_linear, pd.DataFrame(y_train_categorical, columns=['day_of_week_prediction'])], axis=1)
    
    # fit the linear model
    model_linear = LinearRegression().fit(x_train, y_train)

    # use bagging to create an ensemble of linear models
    model_linear_bagging = BaggingRegressor(base_estimator=model_linear, n_estimators=10, random_state=0)
    model_linear_bagging.fit(x_train, y_train)

    r_2 = model_linear_bagging.score(x_train, y_train)
    print(f"coefficient of determination: {r_2}")
    return model_linear_bagging, model_categorical_bagging, r_2

if False:
    data = load_data()
    data_ml, input_cols, output_cols = convert_data(data)
    data_train, data_test = resample_24h(data_ml)
    x_train, y_train, x_test, y_test = analyse_hp_power_consumption(data_train,data_test, output_cols)
    model, r_2 = data_to_model(x_train, y_train)

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


