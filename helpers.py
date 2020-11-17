from numpy.lib.function_base import delete
from pandas.core.frame import DataFrame
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import datetime
from datetime import timedelta
from datetime import timedelta


class Data_processing:
    # requested input: collums: date (YYYYMMDD), wind_veloctity (m/s), wind_rotation(degree), temperature(degree_C), rainfall(mm)
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


class Create_model:
    def __init__(self): #initialiser, dont look at it lol
        self.model = ()

    def window_generator():
        pass
    # the layers of the model

# add class for requesting weather-data from knmi
