import os
import pandas as pd
import tensorflow as tf
import data
from data import Data_processing as dp, transform_data_frame
from model import Create_model as cm
# main functions


def get_data(path_to_csv):
    raw_dataframe = pd.read_csv(path_to_csv)
    raw_weather_dataframe = dp.process_time_data(raw_dataframe)
    clean_dataframe = dp.process_weather_data(
        raw_weather_dataframe)
    transformed_data_frame = data.transform_data_frame.transform_data(clean_dataframe)
    dataset = dp.convert_pandas_dataframe_to_tf_dataset(transformed_data_frame)
    return dataset


def build_model():
    input_shape = (30)
    return model


def train_model(model, train_data):
    # add stuff
    return model


def predict_with_model(model, data):
    # add stuff
    return predicted_data
    pass

b = -1
days_back = -1 * b

path_to_csv = os.path.join(os.getcwd(), "data/weather_data.csv")
weather_data = get_data(path_to_csv)
model = build_model()
trained_model = train_model(model, train_data)
