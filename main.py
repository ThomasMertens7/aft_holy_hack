from test import *
from weather import *
data = load_data()
beginning, end = get_first_and_last_date(data)
data_ml, input_cols, output_cols = convert_data(data)
data_train, data_test = resample_24h(data_ml, data_ml, train_start_date=beginning, train_end_date=end, test_start_date=beginning, test_end_date=end)
quit()

