import numpy as numpy
import pandas as pandas
import tensorflow as tensorflow
import datetime


class Get_data:
    # --------------------------------------------------------------------------------------------------------
    # Read CSV file for weather and parse dates into pandas dataframe.
    # --------------------------------------------------------------------------------------------------------
    def raw_weather_data(path_to_csv):
        raw_dataframe = pandas.read_csv(path_to_csv)
        return raw_dataframe

    def transform_weather_data(raw_weather_dataframe):
        raw_weather_dataframe.info()
        raw_weather_dataframe = Convert_data.time_data(
            raw_weather_dataframe)
        clean_dataframe = Convert_data.weather_data(
            raw_weather_dataframe)
        transformed_data_frame = Transform_data.dataframe(
            clean_dataframe)
        return transformed_data_frame
    # --------------------------------------------------------------------------------------------------------
    # Read CSV file for birds and parse dates into pandas dataframe.
    # --------------------------------------------------------------------------------------------------------

    def bird_data(path_to_csv):
        raw_dataframe = pandas.read_csv(path_to_csv)
        raw_dataframe.info()
        Convert_data.process_bird_migration_data(raw_dataframe)  # TODO
        return raw_dataframe

    def filtered_weather_data(weather_data, bird_data):
        all_bird_dates = bird_data['DATE']
        all_weather_dates = weather_data['DATE']
        # print(all_bird_dates)
        # print(all_weather_dates)

        deleteRows = []

        for date in all_weather_dates:

            if not (all_bird_dates == date).any():
                deleteRows.append(date)
            else:
                pass

        return weather_data


class Convert_data:  # to sin, cosin

    # --------------------------------------------------------------------------------------------------------
    # Process and convert time data into cosin and sin.
    # --------------------------------------------------------------------------------------------------------
    def time_data(date_dataframe):
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
    # --------------------------------------------------------------------------------------------------------
    # Process and convert weather data into vectors.
    # --------------------------------------------------------------------------------------------------------

    def weather_data(weather_dataframe):
        wind_rotation_degrees = weather_dataframe.pop('DDVEC')
        wind_velocity = weather_dataframe.pop('FHVEC')
        wind_rotation_radian = wind_rotation_degrees * 2 * numpy.pi / 360
        wind_x = wind_velocity * numpy.cos(wind_rotation_radian)
        wind_y = wind_velocity * numpy.sin(wind_rotation_radian)
        weather_dataframe['wind_x'] = wind_x
        weather_dataframe['wind_y'] = wind_y
        return weather_dataframe
    # --------------------------------------------------------------------------------------------------------
    # Process and convert bird migration data.
    # --------------------------------------------------------------------------------------------------------

    def process_bird_migration_data(bird_migration_dataframe):
        # get date of flying bird
        # specie = birdMigration_dataframe.pop(
        #     'specie')  # get specie of flying birds
        # avg_amount = birdMigration_dataframe.pop(
        #     'average_amount')  # get average amount of flying birds
        # bird_migration_dataframe['date'] = date_of_flight
        return bird_migration_dataframe
    # --------------------------------------------------------------------------------------------------------
    # Convert pandas dataframe to tensorflow dataset
    # --------------------------------------------------------------------------------------------------------

    def dataframe_to_tf_dataset(pandas_dataframe_input, pandas_dataframe_target):
        if (pandas_dataframe_target.shape[0] != pandas_dataframe_input.shape[0]):
            raise ValueError(
                "shape input dataframe has to be equal to output dataframe")
        tf_dataset = tensorflow.data.Dataset.from_tensor_slices(
            (pandas_dataframe_input.values, pandas_dataframe_target.values))
        return tf_dataset
    # --------------------------------------------------------------------------------------------------------
    # Transform the dataframe.
    # --------------------------------------------------------------------------------------------------------


class Transform_data:

    #! removes last 4 rows
    def dataframe(dataframe_with_weather_data_for_1_day_per_row):
        dataframe = dataframe_with_weather_data_for_1_day_per_row
        series_with_all_collums = dataframe.columns
        for days_back in range(-1, -5, -1):
            for a in range(0, 6):
                column = dataframe.iloc[:, a]
                column_name = series_with_all_collums[a]
                temp_column_name = f'{column_name}_{days_back}'
                temp_column = Transform_data.column(
                    column, days_back)
                dataframe[temp_column_name] = temp_column
                dataframe[f'{temp_column_name}'] = temp_column
        return dataframe

    def column(old_column, days_back):
        for x in range(0, -1*days_back, 1):
            old_column = old_column.drop(index=x)
        old_column.reset_index(drop=True, inplace=True)
        new_column = old_column
        return new_column
    # --------------------------------------------------------------------------------------------------------
    # Get data form KNMI API TODO: It's too tedious for someone to ask API_KEY and request large database.
    # --------------------------------------------------------------------------------------------------------


class Get_data_from_knmi():
    # I think this is not going to happen.
    pass
