import os
import data
import model as m
import pandas as pd
import tensorflow as tf
from data import Data_processing as dp, transform_data_frame
# main functions


def get_data(path_to_csv):
    raw_dataframe = pd.read_csv(path_to_csv)
    raw_weather_dataframe = dp.process_time_data(raw_dataframe)
    clean_dataframe = dp.process_weather_data(
        raw_weather_dataframe)
    transformed_data_frame = data.transform_data_frame.transform_data(clean_dataframe)
    dataset = dp.convert_pandas_dataframe_to_tf_dataset(transformed_data_frame)
    return dataset

path_to_csv = os.path.join(os.getcwd(), "data/weather_data.csv")
weather_data = get_data(path_to_csv)
model = m()
trained_model = train_model(model, train_data)
