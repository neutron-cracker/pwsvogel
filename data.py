import numpy as np
import pandas as pd
import tensorflow as tf
import datetime


class Data_processing:
    def process_time_data(date_dataframe):
        seconds_in_year = 365.2425 * 24 * 60 * 60
        date_time = pd.to_datetime(
            date_dataframe.pop('date'), format='%Y-%m-%d')
        timestamp_seconds_since_epoch = date_time.map(
            datetime.datetime.timestamp)
        moment_of_year_cos = np.sin(
            timestamp_seconds_since_epoch * (2 * np.pi / seconds_in_year))
        moment_of_year_sin = np.cos(
            timestamp_seconds_since_epoch * (2 * np.pi / seconds_in_year))
        date_dataframe['moment_of_year_cos'] = moment_of_year_cos
        date_dataframe['moment_of_year_sin'] = moment_of_year_sin
        return date_dataframe

    def process_weather_data(weather_dataframe):
        wind_rotation_degrees = weather_dataframe.pop('wind_rotation')
        wind_velocity = weather_dataframe.pop('wind_velocity')
        wind_rotation_radian = wind_rotation_degrees * 2 * np.pi / 360
        wind_x = wind_velocity * np.cos(wind_rotation_radian)
        wind_y = wind_velocity * np.sin(wind_rotation_radian)
        weather_dataframe['wind_x'] = wind_x
        weather_dataframe['wind_y'] = wind_y
        return weather_dataframe

    def convert_pandas_dataframe_to_tf_dataset(pandas_dataframe):
        tf_dataset = tf.data.Dataset.from_tensor_slices(pandas_dataframe.values)
        return tf_dataset

class Get_data_from_knmi ():
    pass
