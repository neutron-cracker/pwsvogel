import os
import pandas as pd
import tensorflow as tf
from data import Data_processing as dp
from model import Create_model as cm
# main functions


def get_data(path_to_csv):
    raw_dataframe = pd.read_csv(path_to_csv)
    raw_weather_dataframe = dp.process_time_data(raw_dataframe)
    clean_dataframe = dp.process_weather_data(
        raw_weather_dataframe)
    dataset = dp.convert_pandas_dataframe_to_tf_dataset(clean_dataframe)
    del raw_dataframe
    del raw_weather_dataframe
    # del clean_dataframe
    return dataset


def build_model():
    model = Create_model
    return model


def train_model(model, train_data):
    # add stuff
    return model


def predict_with_model(model, data):
    # add stuff
    return predicted_data
    pass


path_to_csv = os.path.join(os.getcwd(), "data/weather_data.csv")
weather_data = get_data(path_to_csv)
model = build_model()
trained_model = train_model(model, train_data)
