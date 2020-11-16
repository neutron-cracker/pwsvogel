import numpy as np
import pandas as pd
import tensorflow as tf
# from pandas import DataFrame as dtf

class Data_processing:
    def process_to_requested_format(raw_data_csv): #requested input: collums: date (YYYYMMDD), wind_veloctity (m/s), wind_rotation(degree), temperature(degree_C), rainfall(mm)
        raw_data_dataframe = pd.read_csv(raw_data_csv)
        
        #get data series
        timestamp = pd.to_datetime(raw_data_dataframe.pop('date'), format='%Y-%m-%d')
        timestamp_day_of_year = timestamp.dt.dayofyear
        is_leap_year = timestamp.is_leap_year
        wind_rotation_degrees = raw_data_dataframe.pop('wind_rotation')
        wind_velocity = raw_data_dataframe.pop('wind_velocity')
        
        #transform weather series
        wind_rotation_radian = wind_rotation_degrees * 2 * np.pi / 360
        wind_rotation_x = wind_velocity * np.cos(wind_rotation_radian)
        wind_rotation_y = wind_velocity * np.sin(wind_rotation_radian)

        #transform time series
        


class Create_model:
    def __init__(self):
        self.model = ()

    model = tf.keras.Sequential()
    # the layers of the model

# add class for requesting weather-data from knmi
