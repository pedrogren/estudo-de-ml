import pandas as pd
import numpy as np

chicago_taxi_data = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")
training_df = chicago_taxi_data.loc[:,('TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE')]
print(format(len(training_df)))
print(training_df.describe(include='all'))