import numpy as numpy
import pandas as pandas
import tensorflow as tensorflow
import datetime


class Data_processing:
    def process_time_data(date_dataframe):
        seconds_in_year = 365.2425 * 24 * 60 * 60
        date_time = pandas.to_datetime(
            date_dataframe.pop('DATE'), format='%Y-%m-%d')
        timestamp_seconds_since_epoch = date_time.map(
            datetime.datetime.timestamp)
        moment_of_year_cos = numpy.sin(
            timestamp_seconds_since_epoch * (2 * numpy.pi / seconds_in_year))
        moment_of_year_sin = numpy.cos(
            timestamp_seconds_since_epoch * (2 * numpy.pi / seconds_in_year))
        date_dataframe['moment_of_year_cos'] = moment_of_year_cos
        date_dataframe['moment_of_year_sin'] = moment_of_year_sin
        return date_dataframe

    def process_weather_data(weather_dataframe):
        wind_rotation_degrees = weather_dataframe.pop('DDVEC')
        wind_velocity = weather_dataframe.pop('FHVEC')
        wind_rotation_radian = wind_rotation_degrees * 2 * numpy.pi / 360
        wind_x = wind_velocity * numpy.cos(wind_rotation_radian)
        wind_y = wind_velocity * numpy.sin(wind_rotation_radian)
        weather_dataframe['wind_x'] = wind_x
        weather_dataframe['wind_y'] = wind_y
        return weather_dataframe

    def convert_pandas_dataframe_to_tf_dataset(pandas_dataframe):
        tf_dataset = tensorflow.data.Dataset.from_tensor_slices(
            pandas_dataframe.values)
        return tf_dataset


class transform_data_frame():

    #! removes last 4 rows
    def transform_data(dataframe_with_weather_data_for_1_day_per_row):
        dataframe = dataframe_with_weather_data_for_1_day_per_row
        temp_dataframe = dataframe_with_weather_data_for_1_day_per_row
        series_with_all_collums = temp_dataframe.columns
        for b in range(-1, -6, -1):
            days_back = -1 * b
            for a in range(0, 5):
                column = temp_dataframe.iloc[:, a]
                column_name = series_with_all_collums[a]
                temp_column_name = f'{column_name}_{b}'
                temp_column = transform_data_frame.transform_column(  # fix it
                    column, days_back)
                temp_dataframe[temp_column_name] = temp_column
                temp_dataframe[f'{temp_column_name}'] = temp_column
        dataframe_with_weather_data_for_5_days_per_row = temp_dataframe
        return dataframe_with_weather_data_for_5_days_per_row

    def transform_column(old_column, days_back):
        for x in (0, days_back):
            temp_column = old_column.drop(index=x)
        temp_column.reset_index()
        new_column = temp_column
        return new_column


class Get_data_from_knmi():
    pass
